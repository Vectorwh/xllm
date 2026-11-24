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

#include "runtime/dflash/dflash_context_positions.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {
namespace {

TEST(DFlashContextPositionsTest, TextPositionsPassThroughWithoutRebuild) {
  const torch::Tensor positions = torch::tensor({4, 5, 6}, torch::kInt);

  EXPECT_FALSE(dflash_detail::is_mrope_context_positions(positions, 3));
}

TEST(DFlashContextPositionsTest, CpGatherAlignsPaddedMetadataAndBootstrapRows) {
  const torch::Tensor positions = torch::tensor({4, 5, 8}, torch::kInt);
  const torch::Tensor slots = torch::tensor({31, 32, 73}, torch::kInt);
  const torch::Tensor restore_index = torch::tensor({4, 1, 3}, torch::kLong);
  const torch::Tensor embeddings =
      torch::tensor({{-10, -11}, {50, 51}, {-20, -21}, {80, 81}, {40, 41}});

  const auto [gather_positions, gather_slots] =
      dflash_detail::scatter_cp_context_metadata(
          positions, slots, restore_index, embeddings.size(0));
  EXPECT_TRUE(torch::equal(gather_positions,
                           torch::tensor({0, 5, 0, 8, 4}, torch::kInt)));
  EXPECT_TRUE(torch::equal(gather_slots,
                           torch::tensor({-1, 32, -1, 73, 31}, torch::kInt)));

  const torch::Tensor selected_tokens = torch::tensor({1, 2}, torch::kInt);
  const torch::Tensor bootstrap = dflash_detail::select_prefill_context_rows(
      embeddings, selected_tokens, restore_index);
  EXPECT_TRUE(torch::equal(bootstrap, torch::tensor({{50, 51}, {80, 81}})));
}

TEST(DFlashContextPositionsTest, MropePositionsRequireThreeAlignedAxes) {
  const torch::Tensor positions = torch::zeros({3, 4}, torch::kInt);

  EXPECT_TRUE(dflash_detail::is_mrope_context_positions(positions, 4));
}

TEST(DFlashContextPositionsTest, RebuildsMultipleSequencesWithoutCrossing) {
  const std::vector<int32_t> positions =
      dflash_detail::build_token_aligned_context_positions(
          {5, 12, 20}, {5, 4, 2}, 11);
  const std::vector<int32_t> expected = {0, 1, 2, 3, 4, 8, 9, 10, 11, 18, 19};

  EXPECT_EQ(positions, expected);
}

}  // namespace
}  // namespace xllm
