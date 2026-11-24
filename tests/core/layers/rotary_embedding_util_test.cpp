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

#include "layers/common/rotary_embedding_util.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kMaxPositions = 64;
constexpr int64_t kRotaryDim = 64;
constexpr int64_t kHalfRotaryDim = kRotaryDim / 2;

const std::vector<int64_t> kMropeSection = {11, 11, 10};
const std::vector<int64_t> kAxisByHalfDim = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
                                             2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                                             1, 2, 0, 1, 2, 0, 1, 2, 0, 1};

torch::Tensor make_cache() {
  constexpr int64_t kCacheWidth = kRotaryDim * 2;
  return torch::arange(kMaxPositions * kCacheWidth, torch::kFloat32)
      .reshape({kMaxPositions, kCacheWidth})
      .contiguous();
}

torch::Tensor make_mrope_positions() {
  return torch::tensor({{1, 2, 3}, {11, 12, 13}, {21, 22, 23}},
                       torch::TensorOptions().dtype(torch::kLong));
}

std::pair<torch::Tensor, torch::Tensor> make_mrope_ref(
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions) {
  torch::Tensor selected = cos_sin_cache.index({positions});
  std::vector<torch::Tensor> chunks = selected.chunk(/*chunks=*/2, /*dim=*/-1);
  torch::Tensor expected_cos =
      torch::empty({positions.size(1), kRotaryDim}, cos_sin_cache.options());
  torch::Tensor expected_sin =
      torch::empty({positions.size(1), kRotaryDim}, cos_sin_cache.options());

  for (int64_t column = 0; column < kRotaryDim; ++column) {
    int64_t axis = kAxisByHalfDim[column % kHalfRotaryDim];
    expected_cos.select(/*dim=*/1, column)
        .copy_(chunks[0].select(/*dim=*/0, axis).select(/*dim=*/1, column));
    expected_sin.select(/*dim=*/1, column)
        .copy_(chunks[1].select(/*dim=*/0, axis).select(/*dim=*/1, column));
  }
  return {expected_cos, expected_sin};
}

TEST(RotaryEmbeddingUtilTest, RealSectionsProduceInterleavedMrope) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = make_mrope_positions();

  auto [cos, sin] =
      rotary::apply_mrope(cos_sin_cache, positions, kMropeSection);
  auto [expected_cos, expected_sin] = make_mrope_ref(cos_sin_cache, positions);

  EXPECT_EQ(cos.sizes(), torch::IntArrayRef({positions.size(1), kRotaryDim}));
  EXPECT_EQ(sin.sizes(), torch::IntArrayRef({positions.size(1), kRotaryDim}));
  EXPECT_TRUE(torch::equal(cos, expected_cos));
  EXPECT_TRUE(torch::equal(sin, expected_sin));
  EXPECT_EQ(cos.scalar_type(), cos_sin_cache.scalar_type());
  EXPECT_EQ(sin.scalar_type(), cos_sin_cache.scalar_type());
  EXPECT_EQ(cos.device(), cos_sin_cache.device());
  EXPECT_EQ(sin.device(), cos_sin_cache.device());
  EXPECT_TRUE(cos.is_contiguous());
  EXPECT_TRUE(sin.is_contiguous());
}

TEST(RotaryEmbeddingUtilTest, QwenYarnScalesInterleavedMropeAtMillionTokens) {
  ModelArgs args;
  args.head_dim() = 256;
  args.partial_rotary_factor() = 0.25f;
  args.rope_theta() = 10000000.0f;
  args.max_position_embeddings() = 1000000;
  args.rope_scaling_original_max_position_embeddings() = 262144;
  args.rope_scaling_factor() = 4.0f;
  args.rope_scaling_beta_fast() = 32;
  args.rope_scaling_beta_slow() = 1;
  auto cache = rotary::get_qwen_mrope_cache(args, torch::TensorOptions());
  auto positions = torch::tensor(
      {{0, 262144, 999999}, {0, 262145, 999999}, {0, 262146, 999997}},
      torch::kLong);
  auto [cos, sin] = rotary::apply_mrope(cache, positions, kMropeSection);

  // Static YaRN reference for original length 262144 and factor 4.
  EXPECT_EQ(cache.size(0), 1000000);
  EXPECT_NEAR(cos[0][0].item<float>(), 1.1386294f, 1e-6f);
  EXPECT_NEAR(sin[1][18].item<float>(), 0.0800844f, 1e-5f);
  EXPECT_NEAR(sin[2][31].item<float>(), 0.0470921f, 1e-6f);
  EXPECT_NEAR(sin[2][20].item<float>(), -0.4438114f, 1e-5f);
}

TEST(RotaryEmbeddingUtilTest, QwenYarnSharesOnlyMatchingServingWindows) {
  ModelArgs args;
  args.head_dim() = kRotaryDim;
  args.partial_rotary_factor() = 1.0f;
  args.rope_theta() = 10000.0f;
  args.max_position_embeddings() = kMaxPositions;
  args.rope_scaling_original_max_position_embeddings() = kMaxPositions;
  args.rope_scaling_factor() = 2.0f;
  args.rope_scaling_beta_fast() = 32.0f;
  args.rope_scaling_beta_slow() = 1.0f;
  args.rope_scaling_attn_factor() = 1.25f;
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);

  const auto target = rotary::get_qwen_mrope_cache(args, options);
  const auto draft = rotary::get_qwen_mrope_cache(args, options);
  EXPECT_EQ(target.data_ptr(), draft.data_ptr());
  EXPECT_EQ(target.size(0), kMaxPositions);

  // Matching frequency and magnitude still produce a longer generic cache.
  const auto scaled =
      rotary::get_deepseek_rotary_embedding(args.head_dim(),
                                            kRotaryDim,
                                            kMaxPositions,
                                            kMaxPositions,
                                            args.rope_theta(),
                                            /*interleaved=*/false,
                                            args.rope_scaling_factor(),
                                            /*extrapolation_factor=*/1.0f,
                                            args.rope_scaling_attn_factor(),
                                            args.rope_scaling_beta_fast(),
                                            args.rope_scaling_beta_slow(),
                                            /*mscale=*/1.0f,
                                            /*mscale_all_dim=*/1.0f,
                                            options);
  EXPECT_EQ(scaled.size(0), 2 * kMaxPositions);

  args.max_position_embeddings() = kMaxPositions / 2;
  const auto shorter = rotary::get_qwen_mrope_cache(args, options);
  EXPECT_EQ(shorter.size(0), kMaxPositions / 2);
}

TEST(RotaryEmbeddingUtilTest, FractionalYarnBetaSetsFrequencyRamp) {
  auto frequencies = rotary::compute_yarn_inv_freq(
      /*factor=*/4.0f,
      /*extrapolation_factor=*/1,
      /*beta_fast=*/2.0f,
      /*beta_slow=*/0.5f,
      /*rotary_dim=*/16,
      /*theta=*/10000.0f,
      /*old_context_len=*/64);
  // The correction range is [1, 3]. Index 2 blends 0.1 and 0.025 equally.
  EXPECT_NEAR(frequencies[2].item<float>(), 0.0625f, 1e-7f);
}

TEST(RotaryEmbeddingUtilTest, OneDimensionalPositionsMatchExpandedPositions) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = torch::tensor({1, 2, 3}, torch::kLong);
  torch::Tensor expanded_positions = positions.expand({3, -1});

  auto [cos, sin] =
      rotary::apply_mrope(cos_sin_cache, positions, kMropeSection);
  auto [expanded_cos, expanded_sin] =
      rotary::apply_mrope(cos_sin_cache, expanded_positions, kMropeSection);

  EXPECT_TRUE(torch::equal(cos, expanded_cos));
  EXPECT_TRUE(torch::equal(sin, expanded_sin));
}

}  // namespace
}  // namespace layer
}  // namespace xllm
