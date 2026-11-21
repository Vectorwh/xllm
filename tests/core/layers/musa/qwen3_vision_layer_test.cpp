/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "layers/musa/qwen3_vision_layer.h"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/dense_mlp.h"
#include "core/layers/common/lora/lora_column_parallel_linear.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "layers/qwen2_5_vision_layer.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kHiddenSize = 32;
constexpr int64_t kIntermediateSize = 64;
constexpr int64_t kNumHeads = 4;
constexpr int64_t kHeadDim = 8;

std::map<std::string, std::vector<int64_t>> ParameterShapes(
    const torch::nn::Module& module) {
  std::map<std::string, std::vector<int64_t>> shapes;
  for (const auto& parameter : module.named_parameters()) {
    shapes.emplace(parameter.key(), parameter.value().sizes().vec());
  }
  return shapes;
}

torch::Tensor Parameter(torch::nn::Module& module, const std::string& name) {
  for (const auto& parameter : module.named_parameters()) {
    if (parameter.key() == name) {
      return parameter.value();
    }
  }
  return torch::Tensor();
}

std::string CheckpointKey(const std::string& name) {
  const std::vector<std::pair<std::string, std::string>> replacements = {
      {"self_attn.qkv_proj.", "attn.qkv."},
      {"self_attn.proj.", "attn.proj."},
      {"mlp.gate_up_proj.", "mlp.linear_fc1."},
      {"mlp.down_proj.", "mlp.linear_fc2."},
  };
  for (const auto& replacement : replacements) {
    if (name.rfind(replacement.first, 0) == 0) {
      return replacement.second + name.substr(replacement.first.size());
    }
  }
  return name;
}

class Qwen3VisionLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "MUSA device is required";
    }
    device_ = torch::Device(Platform::type_torch(), 0);
    Device xllm_device(device_);
    xllm_device.set_device();
    xllm_device.init_device_context();
    xllm_device.set_seed(7);
    options_ = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    ModelArgs model_args;
    model_args.mm_hidden_size() = kHiddenSize;
    model_args.mm_intermediate_size() = kIntermediateSize;
    model_args.mm_num_attention_heads() = kNumHeads;
    model_args.mm_head_dim() = kHeadDim;
    model_args.mm_hidden_act() = "gelu";
    model_args.rms_norm_eps() = 1e-6f;
    process_group_ = std::make_unique<ProcessGroup>(0, 1, device_);
    parallel_args_ = ParallelArgs(0, 1, process_group_.get());
    parallel_args_.tp_group_ = process_group_.get();
    context_ = ModelContext(parallel_args_, model_args, QuantArgs(), options_);
  }

  std::unordered_map<std::string, torch::Tensor> checkpoint_weights(
      const torch::nn::Module& module) const {
    std::unordered_map<std::string, torch::Tensor> weights;
    for (const auto& parameter : module.named_parameters()) {
      weights.emplace(CheckpointKey(parameter.key()),
                      torch::randn(parameter.value().sizes(), options_) * 0.02);
    }
    weights.emplace(
        "mlp.linear_fc1.weight",
        torch::randn({kIntermediateSize, kHiddenSize}, options_) * 0.02);
    weights.emplace("mlp.linear_fc1.bias",
                    torch::randn({kIntermediateSize}, options_) * 0.02);
    weights.emplace(
        "mlp.linear_fc2.weight",
        torch::randn({kHiddenSize, kIntermediateSize}, options_) * 0.02);
    weights.emplace("mlp.linear_fc2.bias",
                    torch::randn({kHiddenSize}, options_) * 0.02);
    return weights;
  }

  torch::Device device_ = torch::kCPU;
  torch::TensorOptions options_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  std::unique_ptr<ProcessGroup> process_group_;
  ModelContext context_;
};

TEST_F(Qwen3VisionLayerTest, MatchesQwen3StyleAssembly) {
  Qwen3_VisionLayer musa_layer(context_);
  Qwen2_5_VisionLayer qwen3_style(context_, /*is_qwen3_style=*/true);
  Qwen2_5_VisionLayer qwen2_style(context_, /*is_qwen3_style=*/false);

  EXPECT_EQ(ParameterShapes(*musa_layer), ParameterShapes(*qwen3_style));
  const auto* qwen2_gate_up = qwen2_style->named_modules()["mlp.gate_up_proj"]
                                  ->as<LoRAColumnParallelLinear>();
  const auto* musa_gate_up = musa_layer->named_modules()["mlp.gate_up_proj"]
                                 ->as<LoRAColumnParallelLinear>();
  ASSERT_NE(qwen2_gate_up, nullptr);
  ASSERT_NE(musa_gate_up, nullptr);
  EXPECT_GT(qwen2_gate_up->weight().numel(), musa_gate_up->weight().numel());
  EXPECT_TRUE(Parameter(*musa_layer, "norm1.bias").defined());
  EXPECT_FALSE(Parameter(*qwen2_style, "norm1.bias").defined());
}

TEST_F(Qwen3VisionLayerTest, LoadsQwen3CheckpointKeys) {
  Qwen3_VisionLayer layer(context_);
  const auto weights = checkpoint_weights(*layer);
  for (const auto& parameter : layer->named_parameters()) {
    const std::string key = CheckpointKey(parameter.key());
    const bool kept_norm_name =
        key == parameter.key() && (parameter.key().rfind("norm1.", 0) == 0 ||
                                   parameter.key().rfind("norm2.", 0) == 0);
    ASSERT_TRUE(key != parameter.key() || kept_norm_name)
        << "No Qwen3 checkpoint name for " << parameter.key();
  }

  layer->load_state_dict(StateDict(weights));
  for (const auto& parameter : layer->named_parameters()) {
    const torch::Tensor& expected = weights.at(CheckpointKey(parameter.key()));
    EXPECT_TRUE(torch::allclose(parameter.value(), expected));
  }
  const auto modules = layer->named_modules();
  modules["mlp"]->as<DenseMLP>()->verify_loaded_weights("mlp.");
  EXPECT_TRUE(torch::equal(
      modules["mlp.gate_up_proj"]->as<LoRAColumnParallelLinear>()->weight(),
      weights.at("mlp.linear_fc1.weight")));
}

TEST_F(Qwen3VisionLayerTest, ForwardKeepsTheQwen3StyleShape) {
  Qwen3_VisionLayer musa_layer(context_);
  Qwen2_5_VisionLayer qwen3_style(context_, /*is_qwen3_style=*/true);
  const StateDict state_dict(checkpoint_weights(*musa_layer));
  musa_layer->load_state_dict(state_dict);
  const auto reference_modules = qwen3_style->named_modules();
  reference_modules["self_attn"]->as<Qwen2VisionAttention>()->load_state_dict(
      state_dict.get_dict_with_prefix("attn."));
  reference_modules["mlp"]->as<DenseMLP>()->load_state_dict(
      state_dict.get_dict_with_prefix("mlp."), {"linear_fc1."}, "linear_fc2.");
  for (const std::string name : {"norm1", "norm2"}) {
    reference_modules[name]->as<RMSNorm>()->load_state_dict(
        state_dict.get_dict_with_prefix(name + "."));
  }
  torch::Tensor hidden = torch::randn({4, kHiddenSize}, options_);
  torch::Tensor cos = torch::randn({4, kHeadDim}, options_);
  torch::Tensor sin = torch::randn({4, kHeadDim}, options_);
  torch::Tensor cu_seq = torch::tensor({0, 4}, options_.dtype(torch::kInt32));
  std::vector<int32_t> cu_seq_vec = {0, 4};

  torch::Tensor actual =
      musa_layer->forward(hidden, cos, sin, cu_seq, cu_seq_vec, /*node_id=*/0);
  torch::Tensor expected =
      qwen3_style->forward(hidden, cos, sin, cu_seq, cu_seq_vec, /*node_id=*/0);
  EXPECT_EQ(actual.sizes(), expected.sizes());
  EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
  EXPECT_TRUE(torch::allclose(actual, expected, /*rtol=*/1e-4, /*atol=*/1e-4));
}

}  // namespace
}  // namespace layer
}  // namespace xllm
