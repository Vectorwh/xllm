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

#include <framework/core/MLUStream.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model_context.h"
#include "framework/model_loader.h"
#include "framework/parallel_state/process_group.h"
#include "framework/state_dict/state_dict.h"
#include "models/vlm/qwen3_5.h"
#include "platform/device.h"

namespace xllm {
namespace {

constexpr int64_t kHiddenSize = 128;

using Weights = std::unordered_map<std::string, torch::Tensor>;

TEST(Qwen35MluTextTest, TextOnlyPositionsAdvanceOnAllThreeAxes) {
  const std::vector<int32_t> tokens{37, 52, 0};
  const auto [positions, delta] =
      Qwen3VLMPositionGenerator().generate(tokens, MMData{}, ModelArgs{});
  EXPECT_EQ(positions.scalar_type(), torch::kInt32);
  EXPECT_TRUE(torch::equal(positions,
                           torch::tensor({{0, 1, 2}, {0, 1, 2}, {0, 1, 2}})));
  EXPECT_EQ(delta, 0);
}

class InMemoryModelLoader final : public ModelLoader {
 public:
  explicit InMemoryModelLoader(std::vector<Weights> shards) {
    states_.reserve(shards.size());
    for (auto& shard : shards) {
      states_.emplace_back(std::make_unique<StateDict>(std::move(shard)));
    }
  }

  std::unique_ptr<Tokenizer> tokenizer() const override {
    LOG(FATAL) << "Tokenizer construction is not used by weight loading tests";
    return nullptr;
  }

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return states_;
  }

  std::string model_weights_path() const override { return "in-memory"; }

 private:
  std::vector<std::unique_ptr<StateDict>> states_;
};

class Qwen35MluWeightLoadingTest : public ::testing::Test {
 protected:
  ModelContext make_context() {
    ParallelArgs parallel(/*rank=*/1, /*world_size=*/2, &tp_group_);
    parallel.tp_group_ = &tp_group_;
    parallel.lm_head_group_ = &head_group_;
    ModelArgs args;
    args.model_type("qwen3_5_text")
        .n_layers(0)
        .hidden_size(8)
        .vocab_size(8)
        .rms_norm_eps(1e-6)
        .tie_word_embeddings(true);
    return ModelContext(parallel, args, QuantArgs(), options_);
  }

  const torch::TensorOptions options_ =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  ProcessGroup tp_group_{/*rank=*/1, /*world_size=*/2, torch::kCPU};
  ProcessGroup head_group_{/*rank=*/2, /*world_size=*/4, torch::kCPU};
};

TEST_F(Qwen35MluWeightLoadingTest,
       LoadsTiedHeadFromNestedShardsWithOwnTpLayout) {
  Qwen3_5ForCausalLMImpl model(make_context());
  const torch::Tensor embedding = torch::arange(64, options_).reshape({8, 8});
  std::vector<Weights> shards;
  shards.reserve(2);
  shards.emplace_back(Weights{
      {"model.language_model.norm.weight", torch::ones({8}, options_)}});
  shards.emplace_back(
      Weights{{"model.language_model.embed_tokens.weight", embedding},
              {"embed_tokens.weight", torch::full({8, 8}, -1.0f, options_)}});

  model.load_model(std::make_unique<InMemoryModelLoader>(std::move(shards)));

  ASSERT_TRUE(model.get_lm_head()->is_weight_loaded());
  EXPECT_TRUE(torch::equal(model.get_lm_head()->weight(),
                           embedding.slice(/*dim=*/0, /*start=*/4, /*end=*/6)));
  ASSERT_TRUE(model.get_word_embedding()->is_weight_loaded());
  EXPECT_TRUE(torch::equal(model.get_word_embedding()->weight(),
                           embedding.slice(/*dim=*/1, /*start=*/4, /*end=*/8)));
}

TEST_F(Qwen35MluWeightLoadingTest, PreservesRootEmbeddingFallback) {
  Qwen3_5ForCausalLMImpl model(make_context());
  const torch::Tensor embedding = torch::arange(64, options_).reshape({8, 8});
  std::vector<Weights> shards;
  shards.reserve(1);
  shards.emplace_back(
      Weights{{"model.language_model.norm.weight", torch::ones({8}, options_)},
              {"embed_tokens.weight", embedding}});

  model.load_model(std::make_unique<InMemoryModelLoader>(std::move(shards)));

  ASSERT_TRUE(model.get_lm_head()->is_weight_loaded());
  EXPECT_TRUE(torch::equal(model.get_lm_head()->weight(),
                           embedding.slice(/*dim=*/0, /*start=*/4, /*end=*/6)));
}

TEST_F(Qwen35MluWeightLoadingTest, RejectsMissingHeadAfterAllShards) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Qwen3_5ForCausalLMImpl model(make_context());
  std::vector<Weights> shards;
  shards.reserve(2);
  shards.emplace_back(Weights{
      {"model.language_model.norm.weight", torch::ones({8}, options_)}});
  shards.emplace_back();

  EXPECT_DEATH(model.load_model(
                   std::make_unique<InMemoryModelLoader>(std::move(shards))),
               "lm_head weight was not loaded");
}

bool has_safe_visible_device() {
  const char* visible = std::getenv("MLU_VISIBLE_DEVICES");
  if (visible == nullptr) {
    return false;
  }
  const std::string devices(visible);
  if (devices.find(',') != std::string::npos) {
    return false;
  }
  int32_t physical = 0;
  const auto [end, error] = std::from_chars(
      devices.data(), devices.data() + devices.size(), physical);
  return error == std::errc{} && end == devices.data() + devices.size() &&
         physical > 0;
}

struct SyncGatherObserved {};
struct AsyncGatherObserved {};

class GatherProbeGroup final : public ProcessGroup {
 public:
  explicit GatherProbeGroup(const torch::Device& device)
      : ProcessGroup(/*rank=*/0, /*world_size=*/2, device) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++sync_calls_;
    input_rows_ = input.size(0);
    torch_mlu::getCurrentMLUStream(input.device().index()).synchronize();
    throw SyncGatherObserved{};
  }

  c10::intrusive_ptr<c10d::Work> allgather_base_async(
      const torch::Tensor& /*input*/,
      torch::Tensor& /*output*/) override {
    ++async_calls_;
    throw AsyncGatherObserved{};
  }

  int32_t sync_calls() const { return sync_calls_; }
  int32_t async_calls() const { return async_calls_; }
  int64_t input_rows() const { return input_rows_; }

 private:
  int32_t sync_calls_ = 0;
  int32_t async_calls_ = 0;
  int64_t input_rows_ = 0;
};

TEST(Qwen35MluForwardTest, MoeForwardUsesPatchedDpCounts) {
  if (!has_safe_visible_device()) {
    GTEST_SKIP() << "Set MLU_VISIBLE_DEVICES to a healthy physical card ID "
                    "excluding 0";
  }
  const torch::Device device("mlu:0");
  Device xllm_device(device);
  xllm_device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto ints = options.dtype(torch::kInt32);

  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/1, device);
  GatherProbeGroup dp_group(device);
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/2,
                             /*dp_size=*/2,
                             /*cp_size=*/1,
                             /*process_group=*/nullptr,
                             /*ep_size=*/2);
  parallel_args.tp_group_ = &tp_group;
  parallel_args.moe_tp_group_ = &tp_group;
  parallel_args.dp_local_process_group_ = &dp_group;
  parallel_args.moe_ep_group_ = &dp_group;

  ModelArgs args;
  args.model_type("qwen3_5")
      .n_layers(1)
      .layer_types({"full_attention"})
      .hidden_size(kHiddenSize)
      .vocab_size(16)
      .n_heads(2)
      .n_kv_heads(1)
      .head_dim(64)
      .partial_rotary_factor(0.25f)
      .rope_scaling_mrope_section({3, 3, 2})
      .max_position_embeddings(32)
      .rope_theta(10000.0f)
      .rms_norm_eps(1e-6)
      .n_routed_experts(2)
      .num_experts_per_tok(1)
      .moe_intermediate_size(64)
      .n_shared_experts(0)
      .decoder_sparse_step(1)
      .n_group(1)
      .topk_group(1)
      .routed_scaling_factor(1.0f)
      .scoring_func("softmax")
      .hidden_act("silu");
  ModelContext context(parallel_args, args, QuantArgs(), options);
  Qwen3_5ModelImpl model(context);

  // The probe stops in MoE gather, before the gate or expert weights are read.
  // Load finite weights for every preceding embedding, norm and attention op.
  std::unordered_map<std::string, torch::Tensor> weights;
  const auto add_weight =
      [&](const std::string& name, torch::IntArrayRef shape, float value) {
        weights.emplace(name, torch::full(shape, value, options));
      };
  add_weight("embed_tokens.weight", {16, kHiddenSize}, 0.25f);
  add_weight("norm.weight", {kHiddenSize}, 1.0f);
  add_weight("layers.0.input_layernorm.weight", {kHiddenSize}, 1.0f);
  add_weight("layers.0.post_attention_layernorm.weight", {kHiddenSize}, 1.0f);
  add_weight(
      "layers.0.self_attn.q_proj.weight", {kHiddenSize, kHiddenSize}, 0.01f);
  add_weight("layers.0.self_attn.k_proj.weight", {64, kHiddenSize}, 0.01f);
  add_weight("layers.0.self_attn.v_proj.weight", {64, kHiddenSize}, 0.01f);
  add_weight(
      "layers.0.self_attn.o_proj.weight", {kHiddenSize, kHiddenSize}, 0.01f);
  add_weight("layers.0.self_attn.q_norm.weight", {64}, 1.0f);
  add_weight("layers.0.self_attn.k_norm.weight", {64}, 1.0f);
  model.load_state_dict(StateDict(std::move(weights)));

  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 1;
  params.parallel.dp_global_token_nums = {1, 0};
  params.parallel.dp_is_decode = {1, 1};
  params.attention.host.q_seq_lens = {0, 1};
  params.attention.host.kv_seq_lens = {0, 1};
  params.attention.device.q_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.device.kv_seq_lens = torch::tensor({0, 1}, ints);
  params.attention.device.new_cache_slots = torch::tensor({0}, ints);
  params.attention.device.block_tables = torch::zeros({1, 1}, ints);
  params.embedding.input_embedding =
      torch::full({1, kHiddenSize}, 0.25f, options);
  const void* embedding_storage = params.embedding.input_embedding.data_ptr();
  const torch::Tensor tokens = torch::tensor({1}, ints);
  const torch::Tensor positions = torch::tensor({0}, ints);
  std::vector<KVCache> caches;
  caches.emplace_back(KVCacheTensors{torch::zeros({1, 1, 16, 64}, options),
                                     torch::zeros({1, 1, 16, 64}, options)});

  // The real decoder reaches gather_dp_tokens. Patching {1, 0} to {1, 1}
  // selects its equal-count sync gather; passing the original vector selects
  // the uneven async path instead. The probe stops before expert computation.
  EXPECT_THROW(model.forward(tokens, positions, caches, params),
               SyncGatherObserved);
  EXPECT_EQ(dp_group.sync_calls(), 1);
  EXPECT_EQ(dp_group.async_calls(), 0);
  EXPECT_EQ(dp_group.input_rows(), 1);
  EXPECT_EQ(params.parallel.dp_global_token_nums, std::vector<int32_t>({1, 0}));
  EXPECT_EQ(params.embedding.input_embedding.data_ptr(), embedding_storage);
}

}  // namespace
}  // namespace xllm
