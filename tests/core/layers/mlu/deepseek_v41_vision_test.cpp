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
#include "core/layers/mlu/deepseek_v41/deepseek_v41_vision.h"

#include <gtest/gtest.h>
#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm::layer {
namespace {

class DeepseekV41VisionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    args_.mm_hidden_size(8)
        .mm_intermediate_size(8)
        .mm_projection_dim(8)
        .mm_patch_size(2)
        .mm_downsample_ratio(2)
        .mm_num_hidden_layers(1)
        .mm_num_attention_heads(1)
        .mm_num_channels(3)
        .mm_head_dim(8)
        .mm_rope_theta(10000);
    const std::vector<std::pair<std::string, std::vector<int64_t>>> shapes{
        {"vision.patch_embed.proj.weight", {8, 3, 2, 2}},
        {"vision.patch_embed.proj.bias", {8}},
        {"vision.norm.weight", {8}},
        {"image_start", {8}},
        {"image_end", {8}},
        {"image_newline", {8}},
        {"aligner.w1.weight", {8, 32}},
        {"aligner.w1.bias", {8}},
        {"aligner.w2.weight", {8, 8}},
        {"aligner.w2.bias", {8}},
        {"vision.blocks.0.attn.wqkv.weight", {24, 8}},
        {"vision.blocks.0.attn.wqkv.bias", {24}},
        {"vision.blocks.0.attn.wo.weight", {8, 8}},
        {"vision.blocks.0.attn.wo.bias", {8}},
        {"vision.blocks.0.mlp.w1.weight", {16, 8}},
        {"vision.blocks.0.mlp.w2.weight", {8, 8}},
        {"vision.blocks.0.norm1.weight", {8}},
        {"vision.blocks.0.norm2.weight", {8}}};
    weights_.reserve(shapes.size());
    weight_names_.reserve(shapes.size());
    for (const auto& [name, shape] : shapes) {
      torch::Tensor value = torch::empty(shape);
      value.copy_((torch::arange(value.numel()).remainder(17) - 8)
                      .to(torch::kFloat32)
                      .reshape(shape) /
                  80);
      if (name.find("norm") != std::string::npos) {
        value.fill_(1);
      }
      weights_.emplace(name, value);
      weight_names_.emplace_back(name);
    }
    weights_.at("image_start").fill_(-2);
    weights_.at("image_end").fill_(2);
    weights_.at("image_newline").fill_(1);
  }

  DeepseekV41Vision create(const torch::Device& device,
                           torch::ScalarType dtype = torch::kFloat32) {
    ParallelArgs parallel(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
    ModelContext context(parallel,
                         args_,
                         QuantArgs(),
                         torch::TensorOptions().device(device).dtype(dtype));
    DeepseekV41Vision vision(context);
    // Load shards independently, including optional model. prefixes.
    for (const auto& [name, value] : weights_) {
      vision->load_state_dict(StateDict({{"model." + name, value}}));
    }
    vision->verify_loaded_weights();
    return vision;
  }

  torch::Tensor reference(const torch::Tensor& patches, int64_t h, int64_t w) {
    const auto norm = [](const torch::Tensor& x) {
      return x * torch::rsqrt(x.square().mean(-1, true) + 1e-6);
    };
    const auto linear = [&](const torch::Tensor& x, const std::string& prefix) {
      const auto bias = weights_.find(prefix + ".bias");
      return torch::nn::functional::linear(
          x,
          weights_.at(prefix + ".weight").flatten(1),
          bias == weights_.end() ? torch::Tensor() : bias->second);
    };
    torch::Tensor x = linear(patches, "vision.patch_embed.proj");
    const auto qkv = linear(norm(x), "vision.blocks.0.attn.wqkv").chunk(3, -1);
    torch::Tensor positions =
        torch::stack({torch::arange(h).unsqueeze(1).expand({h, w}),
                      torch::arange(w).unsqueeze(0).expand({h, w})},
                     -1)
            .reshape({-1, 2, 1})
            .to(torch::kFloat32);
    torch::Tensor freqs = (positions * torch::tensor({1.0f, 0.01f})).flatten(1);
    const auto rotate = [&](const torch::Tensor& value) {
      const auto parts = value.chunk(2, -1);
      return torch::cat({parts[0] * freqs.cos() - parts[1] * freqs.sin(),
                         parts[1] * freqs.cos() + parts[0] * freqs.sin()},
                        -1);
    };
    torch::Tensor q = rotate(qkv[0]);
    torch::Tensor k = rotate(qkv[1]);
    torch::Tensor attention =
        torch::softmax(q.matmul(k.t()) / std::sqrt(8.0), -1).matmul(qkv[2]);
    x = x + linear(attention, "vision.blocks.0.attn.wo");
    const auto gate_up = linear(norm(x), "vision.blocks.0.mlp.w1").chunk(2, -1);
    x = x +
        linear(torch::silu(gate_up[0]) * gate_up[1], "vision.blocks.0.mlp.w2");
    x = norm(x).reshape({h, w, 8}).permute({2, 0, 1});
    x = torch::nn::functional::pad(
        x, torch::nn::functional::PadFuncOptions({0, w % 2, 0, h % 2}));
    x = torch::nn::functional::unfold(
            x.unsqueeze(0),
            torch::nn::functional::UnfoldFuncOptions(2).stride(2))
            .squeeze(0)
            .t();
    return linear(torch::gelu(linear(x, "aligner.w1")), "aligner.w2");
  }

  ModelArgs args_;
  std::unordered_map<std::string, torch::Tensor> weights_;
  std::vector<std::string> weight_names_;
};

TEST_F(DeepseekV41VisionTest, MatchesDenseReferenceAndOddRowInterleave) {
  torch::InferenceMode inference;
  const torch::Device device("mlu:0");
  torch::DeviceGuard guard(device);
  auto vision = create(device);
  torch::Tensor pixels = (torch::arange(24 * 12)
                                  .remainder(31)
                                  .to(torch::kFloat32)
                                  .reshape({24, 12}) /
                              31 -
                          0.5);
  torch::Tensor features = reference(pixels, /*h=*/6, /*w=*/4);
  // Three LLM rows of two features: pair rows 0/1, then the unpaired row 2.
  torch::Tensor expected = torch::stack({weights_.at("image_start"),
                                         features[0],
                                         features[2],
                                         features[1],
                                         features[3],
                                         weights_.at("image_newline"),
                                         weights_.at("image_newline"),
                                         features[4],
                                         features[5],
                                         weights_.at("image_newline"),
                                         weights_.at("image_end")});
  auto rows = vision->encode(pixels, torch::tensor({1, 6, 4}, torch::kInt64));
  ASSERT_EQ(rows.size(), 1);
  EXPECT_TRUE(torch::allclose(rows[0].cpu(), expected, 1e-4, 1e-5));
}

TEST_F(DeepseekV41VisionTest, Bfloat16MatchesPythonGoldenWithFloat32Norms) {
  torch::InferenceMode inference;
  const torch::Device device("mlu:0");
  torch::DeviceGuard guard(device);
  // LOCKED PAIR with test_vision_numerics_match_native_golden: the ordered
  // weight names, generation formula, pixels and ten feature values must agree.
  int64_t index = 0;
  for (const std::string& name : weight_names_) {
    torch::Tensor& value = weights_.at(name);
    value.copy_(((torch::arange(value.numel()) * 7 + index).remainder(29) - 14)
                    .reshape(value.sizes()) /
                32);
    if (name.find("norm") != std::string::npos) {
      value.copy_(1 + (torch::arange(8) - 4) * 0.0013);
    } else if (name.find("patch_embed") != std::string::npos) {
      value.mul_(0.001);
    }
    ++index;
  }
  auto vision = create(device, torch::kBFloat16);
  // This is a checkpoint precision contract: widening a BF16 copy loses it.
  const auto parameters = vision->named_parameters();
  for (const auto& [name, value] : weights_) {
    if (name.find("norm") == std::string::npos) {
      continue;
    }
    std::string parameter_name = name;
    std::replace(parameter_name.begin(), parameter_name.end(), '.', '_');
    const torch::Tensor weight = parameters[parameter_name].cpu();
    EXPECT_EQ(weight.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::equal(weight, value)) << name;
  }
  torch::Tensor pixels = torch::arange(24 * 12)
                                 .remainder(31)
                                 .to(torch::kFloat32)
                                 .reshape({24, 12}) /
                             31 -
                         0.5;
  auto rows = vision->encode(pixels, torch::tensor({1, 6, 4}, torch::kInt64));
  ASSERT_EQ(rows.size(), 1);
  ASSERT_EQ(rows[0].dim(), 2);
  ASSERT_EQ(rows[0].size(0), 11);
  ASSERT_EQ(rows[0].size(1), 8);
  EXPECT_EQ(rows[0].scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::isfinite(rows[0]).all().item<bool>());
  // Skip image_start; rows 1 and 2 are image features, before any newline.
  const torch::Tensor actual = rows[0]
                                   .slice(/*dim=*/0, /*start=*/1, /*end=*/3)
                                   .flatten()
                                   .slice(/*dim=*/0, /*start=*/0, /*end=*/10)
                                   .cpu()
                                   .to(torch::kFloat32);
  const torch::Tensor expected = torch::tensor({-0.0947265625f,
                                                0.8984375f,
                                                0.91015625f,
                                                0.0137939453125f,
                                                0.06591796875f,
                                                0.0179443359375f,
                                                0.028076171875f,
                                                -0.8671875f,
                                                -0.08642578125f,
                                                0.88671875f});
  EXPECT_TRUE(torch::allclose(actual,
                              expected,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-3))
      << "actual:\n"
      << actual << "\nexpected:\n"
      << expected;
}

}  // namespace
}  // namespace xllm::layer
