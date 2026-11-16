/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/speculative/dflash_async_input_builder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>

namespace xllm::dflash_async {
namespace {

testing::AssertionResult compare_next_draft_inputs(
    const NextDraftInputs& actual,
    const NextDraftInputs& expected) {
  constexpr std::array<const char*, 6> kFieldNames = {"accepted_counts",
                                                      "anchor_tokens",
                                                      "query_tokens",
                                                      "query_positions",
                                                      "query_slots",
                                                      "kv_lengths"};
  const std::array<torch::Tensor, 6> actual_fields = {actual.accepted_counts,
                                                      actual.anchor_tokens,
                                                      actual.query_tokens,
                                                      actual.query_positions,
                                                      actual.query_slots,
                                                      actual.kv_lengths};
  const std::array<torch::Tensor, 6> expected_fields = {
      expected.accepted_counts,
      expected.anchor_tokens,
      expected.query_tokens,
      expected.query_positions,
      expected.query_slots,
      expected.kv_lengths};
  for (size_t field_index = 0; field_index < kFieldNames.size();
       ++field_index) {
    if (actual_fields[field_index].scalar_type() !=
            expected_fields[field_index].scalar_type() ||
        actual_fields[field_index].device() !=
            expected_fields[field_index].device() ||
        actual_fields[field_index].sizes() !=
            expected_fields[field_index].sizes() ||
        !torch::equal(actual_fields[field_index],
                      expected_fields[field_index])) {
      return testing::AssertionFailure() << kFieldNames[field_index];
    }
  }
  return testing::AssertionSuccess();
}

TEST(DFlashAsyncInputBuilderTest, ReorderedSubsetsPreserveRowIdentity) {
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kQueryWidth = 15;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kMaskTokenId = 42;
  constexpr int64_t kBatchSize = 8;
  torch::Tensor base_positions =
      torch::tensor({95, 112, 113, 126, 127, 128, 255, 256}, torch::kLong);
  torch::Tensor accepted_counts =
      torch::tensor({1, 16, 2, 15, 4, 3, 8, 1}, torch::kLong);
  torch::Tensor token_offsets =
      torch::arange(kAcceptedWidth, torch::kLong).unsqueeze(0);
  torch::Tensor tokens =
      torch::arange(kBatchSize * kAcceptedWidth, torch::kLong)
          .view({kBatchSize, kAcceptedWidth});
  torch::Tensor accepted =
      torch::where(token_offsets.lt(accepted_counts.unsqueeze(1)),
                   tokens,
                   torch::full_like(tokens, -1));
  torch::Tensor block_tables =
      torch::arange(kBatchSize * 4, torch::kInt32).view({kBatchSize, 4});
  NextDraftInputs whole = prepare_next_draft(accepted,
                                             base_positions,
                                             block_tables,
                                             kMaskTokenId,
                                             kBlockSize,
                                             kQueryWidth);
  ContextKvInputs whole_context =
      prepare_context_kv(accepted, base_positions, block_tables, kBlockSize);
  EXPECT_TRUE(torch::equal(whole.accepted_counts, accepted_counts));
  EXPECT_TRUE(torch::equal(
      whole.anchor_tokens,
      torch::tensor({0, 31, 33, 62, 67, 82, 103, 112}, torch::kLong)));
  for (const torch::Tensor& rows :
       {torch::tensor({7, 1, 4}, torch::kLong),
        torch::tensor({0, 6, 3, 2, 5}, torch::kLong)}) {
    torch::Tensor local_accepted = accepted.index_select(0, rows);
    torch::Tensor local_positions = base_positions.index_select(0, rows);
    torch::Tensor local_tables = block_tables.index_select(0, rows);
    NextDraftInputs local = prepare_next_draft(local_accepted,
                                               local_positions,
                                               local_tables,
                                               kMaskTokenId,
                                               kBlockSize,
                                               kQueryWidth);
    ContextKvInputs context = prepare_context_kv(
        local_accepted, local_positions, local_tables, kBlockSize);
    EXPECT_TRUE(torch::equal(local.accepted_counts,
                             whole.accepted_counts.index_select(0, rows)));
    EXPECT_TRUE(torch::equal(local.anchor_tokens,
                             whole.anchor_tokens.index_select(0, rows)));
    EXPECT_TRUE(
        torch::equal(local.kv_lengths, whole.kv_lengths.index_select(0, rows)));
    EXPECT_TRUE(
        torch::equal(local.query_slots.view({rows.numel(), kQueryWidth}),
                     whole.query_slots.view({kBatchSize, kQueryWidth})
                         .index_select(0, rows)));
    EXPECT_TRUE(
        torch::equal(local.query_tokens.view({rows.numel(), kQueryWidth}),
                     whole.query_tokens.view({kBatchSize, kQueryWidth})
                         .index_select(0, rows)));
    EXPECT_TRUE(
        torch::equal(context.slots.view({rows.numel(), kAcceptedWidth}),
                     whole_context.slots.view({kBatchSize, kAcceptedWidth})
                         .index_select(0, rows)));
  }
}

TEST(DFlashAsyncInputBuilderTest, DraftQueryDoesNotMapUnusedContextTail) {
  const torch::Tensor accepted_tokens =
      torch::tensor({{31, -1, 99, -1}}, torch::kLong);
  const torch::Tensor base_positions = torch::tensor({126}, torch::kLong);
  const torch::Tensor block_table = torch::tensor({{8}}, torch::kInt32);
  constexpr int64_t kMaskTokenId = 999;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kQueryWidth = 1;
  const NextDraftInputs result = prepare_next_draft(accepted_tokens,
                                                    base_positions,
                                                    block_table,
                                                    kMaskTokenId,
                                                    kBlockSize,
                                                    kQueryWidth);
  const NextDraftInputs expected{torch::tensor({1}, torch::kLong),
                                 torch::tensor({31}, torch::kLong),
                                 torch::tensor({31}, torch::kLong),
                                 torch::tensor({127}, torch::kLong),
                                 torch::tensor({1151}, torch::kInt32),
                                 torch::tensor({128}, torch::kInt32)};
  EXPECT_TRUE(compare_next_draft_inputs(result, expected));
}

TEST(DFlashAsyncInputBuilderTest, AllAcceptanceLengthsAcrossPageBoundary) {
  constexpr int64_t kAcceptedWidth = 16;
  for (const int64_t query_width : {1, 15}) {
    SCOPED_TRACE(query_width);
    constexpr int64_t kBlockSize = 128;
    constexpr int64_t kMaskTokenId = 42;
    constexpr int64_t kBatchSize = 2 * kAcceptedWidth;
    const torch::Tensor accepted_counts =
        torch::arange(1, kAcceptedWidth + 1, torch::kLong).repeat_interleave(2);
    const torch::Tensor token_columns =
        torch::arange(kAcceptedWidth, torch::kLong).unsqueeze(0);
    const torch::Tensor tokens =
        torch::arange(kBatchSize * kAcceptedWidth, torch::kLong)
            .reshape({kBatchSize, kAcceptedWidth}) +
        100;
    const torch::Tensor accepted =
        torch::where(token_columns.lt(accepted_counts.unsqueeze(1)),
                     tokens,
                     torch::full_like(tokens, -1));
    const torch::Tensor base_positions =
        torch::tensor({127, 128}, torch::kLong).repeat({kAcceptedWidth});
    const torch::Tensor block_tables =
        torch::arange(kBatchSize * 3, torch::kInt32).reshape({kBatchSize, 3});
    const NextDraftInputs actual = prepare_next_draft(accepted,
                                                      base_positions,
                                                      block_tables,
                                                      kMaskTokenId,
                                                      kBlockSize,
                                                      query_width);
    const ContextKvInputs actual_context =
        prepare_context_kv(accepted, base_positions, block_tables, kBlockSize);

    std::vector<int64_t> anchors;
    std::vector<int64_t> query_tokens;
    std::vector<int64_t> query_positions;
    std::vector<int32_t> query_slots;
    std::vector<int32_t> kv_lengths;
    std::vector<int64_t> context_positions;
    std::vector<int32_t> context_slots;
    for (int64_t row = 0; row < kBatchSize; ++row) {
      const int64_t count = accepted_counts[row].item<int64_t>();
      const int64_t base = base_positions[row].item<int64_t>();
      const int64_t anchor = accepted[row][count - 1].item<int64_t>();
      anchors.push_back(anchor);
      kv_lengths.push_back(base + count + query_width);
      for (int64_t column = 0; column < query_width; ++column) {
        const int64_t position = base + count + column;
        query_tokens.push_back(column == 0 ? anchor : kMaskTokenId);
        query_positions.push_back(position);
        const int64_t page = position / kBlockSize;
        query_slots.push_back(block_tables[row][page].item<int32_t>() *
                                  kBlockSize +
                              position % kBlockSize);
      }
      for (int64_t column = 0; column < kAcceptedWidth; ++column) {
        const int64_t position = base + column;
        context_positions.push_back(position);
        if (column < count) {
          const int64_t page = position / kBlockSize;
          context_slots.push_back(block_tables[row][page].item<int32_t>() *
                                      kBlockSize +
                                  position % kBlockSize);
        } else {
          context_slots.push_back(-1);
        }
      }
    }

    const NextDraftInputs expected{accepted_counts,
                                   torch::tensor(anchors, torch::kLong),
                                   torch::tensor(query_tokens, torch::kLong),
                                   torch::tensor(query_positions, torch::kLong),
                                   torch::tensor(query_slots, torch::kInt32),
                                   torch::tensor(kv_lengths, torch::kInt32)};
    EXPECT_TRUE(compare_next_draft_inputs(actual, expected));
    EXPECT_TRUE(torch::equal(actual_context.positions,
                             torch::tensor(context_positions, torch::kLong)));
    EXPECT_TRUE(torch::equal(actual_context.slots,
                             torch::tensor(context_slots, torch::kInt32)));
  }
}
TEST(DFlashAsyncInputBuilderTest, DefaultQueryWidthUsesAcceptedColumnWidth) {
  const torch::Tensor accepted =
      torch::tensor({{10, -1, -1}, {20, 21, 22}}, torch::kLong);
  const torch::Tensor bases = torch::tensor({3, 6}, torch::kLong);
  const torch::Tensor tables =
      torch::tensor({{7, 2, 9}, {3, 8, 5}}, torch::kInt32);
  const NextDraftInputs actual = prepare_next_draft(
      accepted, bases, tables, /*mask_token_id=*/999, /*block_size=*/4);
  const NextDraftInputs expected{
      torch::tensor({1, 3}, torch::kLong),
      torch::tensor({10, 22}, torch::kLong),
      torch::tensor({10, 999, 999, 22, 999, 999}, torch::kLong),
      torch::tensor({4, 5, 6, 9, 10, 11}, torch::kLong),
      torch::tensor({8, 9, 10, 21, 22, 23}, torch::kInt32),
      torch::tensor({7, 12}, torch::kInt32)};
  EXPECT_TRUE(compare_next_draft_inputs(actual, expected));
  const ContextKvInputs context =
      prepare_context_kv(accepted, bases, tables, /*block_size=*/4);
  EXPECT_TRUE(torch::equal(context.positions,
                           torch::tensor({3, 4, 5, 6, 7, 8}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      context.slots, torch::tensor({31, -1, -1, 34, 35, 20}, torch::kInt32)));
}

TEST(DFlashAsyncInputBuilderTest,
     OnlyTheContiguousAcceptedPrefixDeterminesQuery) {
  torch::Tensor accepted = torch::tensor(
      {{31, 32, -1, 77, -1, -1, 88, -1, -1, -1, -1, -1, -1, -1, -1, 99}},
      torch::kLong);
  constexpr int64_t kMaskTokenId = 999;
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kQueryWidth = 15;
  NextDraftInputs result =
      prepare_next_draft(accepted,
                         torch::tensor({127}, torch::kLong),
                         torch::tensor({{8, 3, 9}}, torch::kInt32),
                         kMaskTokenId,
                         kBlockSize,
                         kQueryWidth);
  EXPECT_EQ(result.accepted_counts.item<int64_t>(), 2);
  EXPECT_EQ(result.anchor_tokens.item<int64_t>(), 32);
  torch::Tensor query_tokens =
      torch::full({kQueryWidth}, kMaskTokenId, torch::kLong);
  query_tokens[0] = 32;
  EXPECT_TRUE(torch::equal(result.query_tokens, query_tokens));
  EXPECT_TRUE(torch::equal(result.query_positions,
                           torch::arange(129, 144, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(result.query_slots, torch::arange(385, 400, torch::kInt32)));
  EXPECT_EQ(result.kv_lengths.item<int64_t>(), 144);
}

TEST(DFlashAsyncInputBuilderTest, RejectsNonPositiveQueryWidth) {
  torch::Tensor accepted_tokens = torch::tensor({{31, -1}}, torch::kLong);
  torch::Tensor base_positions = torch::tensor({0}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{0}}, torch::kInt32);
  constexpr int64_t kMaskTokenId = 999;
  constexpr int64_t kBlockSize = 128;
  for (const int64_t query_width : {0, -1}) {
    EXPECT_DEATH(prepare_next_draft(accepted_tokens,
                                    base_positions,
                                    block_table,
                                    kMaskTokenId,
                                    kBlockSize,
                                    query_width),
                 "query_width");
  }
}

TEST(DFlashAsyncInputBuilderTest, FixedReplayMatchesGoldenFields) {
  const torch::Tensor accepted =
      torch::tensor({{0, 17, -1, 99}, {31, 32, 33, -1}}, torch::kLong);
  const torch::Tensor bases = torch::tensor({126, 127}, torch::kLong);
  const torch::Tensor tables =
      torch::tensor({{7, 2, 9}, {3, 8, 5}}, torch::kInt32);
  const NextDraftInputs expected{
      torch::tensor({2, 3}, torch::kLong),
      torch::tensor({17, 33}, torch::kLong),
      torch::tensor({17, 999, 33, 999}, torch::kLong),
      torch::tensor({128, 129, 130, 131}, torch::kLong),
      torch::tensor({256, 257, 1026, 1027}, torch::kInt32),
      torch::tensor({130, 132}, torch::kInt32)};
  const NextDraftInputs actual =
      prepare_next_draft(accepted, bases, tables, 999, 128, 2);
  EXPECT_TRUE(compare_next_draft_inputs(actual, expected));

  constexpr std::array<const char*, 6> kFieldNames = {"accepted_counts",
                                                      "anchor_tokens",
                                                      "query_tokens",
                                                      "query_positions",
                                                      "query_slots",
                                                      "kv_lengths"};
  for (size_t field_index = 0; field_index < kFieldNames.size();
       ++field_index) {
    SCOPED_TRACE(kFieldNames[field_index]);
    NextDraftInputs changed = actual;
    const std::array<torch::Tensor*, 6> fields = {&changed.accepted_counts,
                                                  &changed.anchor_tokens,
                                                  &changed.query_tokens,
                                                  &changed.query_positions,
                                                  &changed.query_slots,
                                                  &changed.kv_lengths};
    *fields[field_index] = fields[field_index]->clone();
    fields[field_index]->reshape({-1})[0].add_(1);
    const testing::AssertionResult mismatch =
        compare_next_draft_inputs(changed, expected);
    EXPECT_FALSE(mismatch);
    EXPECT_STREQ(mismatch.message(), kFieldNames[field_index]);
  }
  NextDraftInputs wrong_slot_dtype = actual;
  wrong_slot_dtype.query_slots = wrong_slot_dtype.query_slots.to(torch::kLong);
  const auto slot_dtype_mismatch =
      compare_next_draft_inputs(wrong_slot_dtype, expected);
  EXPECT_FALSE(slot_dtype_mismatch);
  EXPECT_STREQ(slot_dtype_mismatch.message(), "query_slots");
}

TEST(DFlashAsyncInputBuilderTest, RejectsInvalidBatchGeometry) {
  const torch::Tensor accepted = torch::tensor({{0, -1}}, torch::kLong);
  const torch::Tensor bases = torch::tensor({0}, torch::kLong);
  const torch::Tensor tables = torch::tensor({{7, 2}}, torch::kInt32);
  EXPECT_DEATH(prepare_context_kv(accepted.reshape({2}), bases, tables, 128),
               "Check failed");
  EXPECT_DEATH(prepare_context_kv(accepted, bases.reshape({1, 1}), tables, 128),
               "Check failed");
  EXPECT_DEATH(prepare_context_kv(accepted, bases, tables.reshape({2}), 128),
               "Check failed");
  EXPECT_DEATH(prepare_context_kv(accepted.repeat({2, 1}), bases, tables, 128),
               "Check failed");
  EXPECT_DEATH(
      prepare_next_draft(accepted, bases, tables.repeat({2, 1}), 999, 128, 15),
      "Check failed");
}

TEST(DFlashAsyncInputBuilderTest, RejectsOutOfRangeValidContextAndQuerySlots) {
  const torch::Tensor accepted = torch::tensor({{0, -1}}, torch::kLong);
  const torch::Tensor tables = torch::tensor({{7}}, torch::kInt32);
  EXPECT_ANY_THROW(prepare_context_kv(
      accepted, torch::tensor({-1}, torch::kLong), tables, 128));
  EXPECT_ANY_THROW(prepare_context_kv(
      accepted, torch::tensor({128}, torch::kLong), tables, 128));
  EXPECT_ANY_THROW(prepare_next_draft(
      accepted, torch::tensor({127}, torch::kLong), tables, 999, 128, 15));
}

TEST(DFlashAsyncInputBuilderTest, ContextKvUsesOnlyContiguousAcceptedPrefixes) {
  torch::Tensor accepted_tokens = torch::tensor({{0, -1, 77, 88},
                                                 {31, 0, -1, 99},
                                                 {0, 42, 43, -1},
                                                 {51, 52, 0, 54},
                                                 {-1, 0, 63, 64}},
                                                torch::kLong);
  torch::Tensor base_positions = torch::tensor({0, 3, 6, 7, 8}, torch::kInt32);
  torch::Tensor block_table = torch::tensor(
      {{3, 7, 2}, {5, 2, 8}, {9, 4, 6}, {8, 3, 1}, {4, 5, 7}}, torch::kInt32);
  constexpr int64_t kBlockSize = 4;
  ContextKvInputs result = prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize);
  torch::Tensor expected_positions = torch::tensor(
      {{0, 1, 2, 3}, {3, 4, 5, 6}, {6, 7, 8, 9}, {7, 8, 9, 10}, {8, 9, 10, 11}},
      torch::kLong);
  torch::Tensor expected_slots = torch::tensor({{12, -1, -1, -1},
                                                {23, 8, -1, -1},
                                                {18, 19, 24, -1},
                                                {15, 4, 5, 6},
                                                {-1, -1, -1, -1}},
                                               torch::kInt32);
  EXPECT_TRUE(torch::equal(result.positions, expected_positions.reshape({-1})));
  EXPECT_TRUE(torch::equal(result.slots, expected_slots.reshape({-1})));
  EXPECT_EQ(result.positions.scalar_type(), torch::kLong);
  EXPECT_EQ(result.slots.scalar_type(), torch::kInt32);
  EXPECT_EQ(result.positions.device(), base_positions.device());
  EXPECT_EQ(result.slots.device(), block_table.device());
}

TEST(DFlashAsyncInputBuilderTest, ContextKvDoesNotRequireNextQueryBlocks) {
  constexpr int64_t kAcceptedWidth = 16;
  constexpr int64_t kBlockSize = 128;
  torch::Tensor accepted_tokens =
      torch::arange(kAcceptedWidth, torch::kLong).reshape({1, kAcceptedWidth});
  torch::Tensor base_positions = torch::tensor({112}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{8}}, torch::kInt32);
  ContextKvInputs result = prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize);
  EXPECT_TRUE(
      torch::equal(result.positions, torch::arange(112, 128, torch::kLong)));
  EXPECT_TRUE(
      torch::equal(result.slots, torch::arange(1136, 1152, torch::kInt32)));
}

TEST(DFlashAsyncInputBuilderTest, ContextKvDoesNotMapRejectedTailPositions) {
  torch::Tensor accepted_tokens = torch::tensor(
      {{0, -1, 42, -1}, {0, 33, -1, 77}, {-1, 0, 99, -1}}, torch::kLong);
  torch::Tensor base_positions = torch::tensor({127, 126, 128}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{8}, {3}, {9}}, torch::kInt32);
  constexpr int64_t kBlockSize = 128;
  ContextKvInputs result = prepare_context_kv(
      accepted_tokens, base_positions, block_table, kBlockSize);
  torch::Tensor expected_positions = torch::tensor(
      {{127, 128, 129, 130}, {126, 127, 128, 129}, {128, 129, 130, 131}},
      torch::kLong);
  torch::Tensor expected_slots =
      torch::tensor({{1151, -1, -1, -1}, {510, 511, -1, -1}, {-1, -1, -1, -1}},
                    torch::kInt32);
  EXPECT_TRUE(torch::equal(result.positions, expected_positions.reshape({-1})));
  EXPECT_TRUE(torch::equal(result.slots, expected_slots.reshape({-1})));
}

TEST(DFlashAsyncInputBuilderTest, ContextKvRejectsNonPositiveBlockSize) {
  torch::Tensor accepted_tokens = torch::tensor({{0, -1}}, torch::kLong);
  torch::Tensor base_positions = torch::tensor({0}, torch::kLong);
  torch::Tensor block_table = torch::tensor({{0}}, torch::kInt32);
  for (const int64_t block_size : {0, -1}) {
    EXPECT_DEATH(prepare_context_kv(
                     accepted_tokens, base_positions, block_table, block_size),
                 "block_size");
  }
}

TEST(DFlashAsyncInputBuilderTest,
     MultiTableContextAndDraftUseCircularSwaSlots) {
  constexpr int64_t kBlockSize = 4;
  const std::vector<torch::Tensor> multi_block_tables = {
      torch::tensor({{7, 3}, {7, 3}}, torch::kInt32),
      torch::tensor({{11, 17, 23}, {11, 17, 23}}, torch::kInt32)};
  const DraftCacheBlockTable table =
      prepare_draft_cache_block_table(torch::Tensor(),
                                      multi_block_tables,
                                      /*num_sequences=*/2,
                                      torch::Device(torch::kCPU));
  ASSERT_TRUE(table.is_circular);
  EXPECT_TRUE(torch::equal(table.blocks, multi_block_tables.front()));
  const torch::Tensor bases = torch::tensor({6, 7}, torch::kLong);
  const torch::Tensor tokens =
      torch::tensor({{10, 11, -1, -1}, {20, 21, 22, -1}}, torch::kLong);
  const ContextKvInputs context =
      prepare_context_kv(tokens, bases, table, kBlockSize);
  EXPECT_TRUE(
      torch::equal(context.positions,
                   torch::tensor({6, 7, 8, 9, 7, 8, 9, 10}, torch::kLong)));
  EXPECT_TRUE(torch::equal(
      context.slots,
      torch::tensor({14, 15, -1, -1, 15, 28, 29, -1}, torch::kInt32)));
  const NextDraftInputs next = prepare_next_draft(tokens,
                                                  bases,
                                                  table,
                                                  /*mask_token_id=*/999,
                                                  kBlockSize,
                                                  /*query_width=*/3);
  const NextDraftInputs expected{
      torch::tensor({2, 3}, torch::kLong),
      torch::tensor({11, 22}, torch::kLong),
      torch::tensor({11, 999, 999, 22, 999, 999}, torch::kLong),
      torch::tensor({8, 9, 10, 10, 11, 12}, torch::kLong),
      torch::tensor({28, 29, 30, 30, 31, 12}, torch::kInt32),
      torch::tensor({11, 13}, torch::kInt32)};
  EXPECT_TRUE(compare_next_draft_inputs(next, expected));
}

TEST(DFlashAsyncInputBuilderTest, MultiTableIgnoresPaddedBatchRows) {
  std::vector<torch::Tensor> multi_block_tables = {
      torch::tensor({{7, 3}, {-1, -1}}, torch::kInt32),
      torch::tensor({{11}, {-1}}, torch::kInt32)};
  DraftCacheBlockTable table = prepare_draft_cache_block_table(
      torch::Tensor(), multi_block_tables, 1, torch::Device(torch::kCPU));
  EXPECT_EQ(table.blocks.size(0), 1);
  torch::Tensor tokens = torch::tensor({{0, -1, 42}}, torch::kLong);
  torch::Tensor bases = torch::tensor({7}, torch::kLong);
  ContextKvInputs context = prepare_context_kv(tokens, bases, table, 4);
  EXPECT_TRUE(torch::equal(context.slots, torch::tensor({15, -1, -1})));
  EXPECT_EQ(multi_block_tables.front().size(0), 2);
}

TEST(DFlashAsyncInputBuilderTest, SingleTableKeepsLinearAddressing) {
  torch::Tensor blocks = torch::tensor({{7, 3, 11}}, torch::kInt32);
  DraftCacheBlockTable table = prepare_draft_cache_block_table(
      blocks, {}, 1, torch::Device(torch::kCPU));
  EXPECT_FALSE(table.is_circular);
  EXPECT_EQ(table.blocks.data_ptr(), blocks.data_ptr());
  ContextKvInputs context =
      prepare_context_kv(torch::tensor({{42}}, torch::kLong),
                         torch::tensor({8}, torch::kLong),
                         table,
                         4);
  EXPECT_EQ(context.slots[0].item<int32_t>(), 44);
}

}  // namespace
}  // namespace xllm::dflash_async
