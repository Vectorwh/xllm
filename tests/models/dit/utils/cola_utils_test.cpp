/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "models/dit/utils/cola_block_causal_mask.h"
#include "models/dit/utils/cola_weight_loader.h"

namespace xllm {
namespace {

TEST(ColaCheckpointKeyTest, MultipleDotDigits) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key("blocks.12.attn.3.qkv"),
            "blocks_12.attn_3.qkv");
}

TEST(ColaCheckpointKeyTest, DotBeforeNonDigitIsPreserved) {
  EXPECT_EQ(cola_checkpoint_key_to_cpp_key("encoder.weight"), "encoder.weight");
}

TEST(ColaIgnoredKeyTest, RopeFreqsSuffixIgnored) {
  EXPECT_TRUE(is_cola_ignored_checkpoint_key("model.rope.rope.freqs"));
  EXPECT_FALSE(is_cola_ignored_checkpoint_key("model.rope.rope.freqs.extra"));
}

TEST(ColaIgnoredKeyTest, NormalWeightNotIgnored) {
  EXPECT_FALSE(is_cola_ignored_checkpoint_key("model.weight"));
}

TEST(BlockCausalMaskTest, TwoSamplesCrossIsolation) {
  const torch::Tensor mask = create_block_causal_mask(
      {8, 4}, {8, 4}, /*block_size=*/4, torch::kBFloat16, torch::kCPU);
  ASSERT_EQ(mask.sizes(), torch::IntArrayRef({1, 1, 12, 12}));
  EXPECT_EQ(mask.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(mask.slice(2, 0, 8).slice(3, 8).lt(0).all().item<bool>());
  EXPECT_TRUE(mask.slice(2, 8).slice(3, 0, 8).lt(0).all().item<bool>());
  EXPECT_TRUE(mask.slice(2, 0, 4).slice(3, 4, 8).lt(0).all().item<bool>());
  EXPECT_TRUE(mask.slice(2, 4, 8).slice(3, 0, 8).eq(0).all().item<bool>());
  EXPECT_TRUE(mask.slice(2, 8).slice(3, 8).eq(0).all().item<bool>());
}

}  // namespace
}  // namespace xllm
