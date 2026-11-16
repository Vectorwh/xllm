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

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/framework/request/sequence.h"
#include "framework/block/block_manager_test_access.h"
#include "framework/block/linear_state_block_manager.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

// Build a distinct prefix hash keyed by a single byte so tests can reference
// checkpoints by an easy-to-read tag.
XXH3Key make_hash(uint8_t tag) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> bytes{};
  bytes.fill(tag);
  return XXH3Key(bytes.data());
}

int32_t insert_checkpoint(LinearStateBlockManager& cache, const XXH3Key& hash) {
  std::vector<Block> allocated = cache.allocate(1);
  if (allocated.empty()) {
    return -1;
  }
  const int32_t slot = allocated.front().id();
  allocated.front().set_hash_value(hash.data);
  cache.cache(allocated);
  return slot;
}

BlockManager::Options linear_options(uint32_t num_slots, int32_t block_size) {
  BlockManager::Options options;
  options.num_blocks(num_slots).block_size(block_size);
  return options;
}

Sequence make_prefetch_sequence(size_t num_tokens) {
  static RequestSamplingParam sampling_param;
  static StoppingChecker stopping_checker(/*max_generated_tokens=*/16,
                                          /*max_context_len=*/0,
                                          /*eos_token=*/-1,
                                          /*ignore_eos=*/false,
                                          /*stop_tokens=*/{},
                                          /*stop_sequences=*/{});
  SequenceParams seq_params;
  seq_params.seq_capacity = num_tokens + 16;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  IncrementalDecoder decoder("", num_tokens, false, false);
  return Sequence(/*index=*/0,
                  std::vector<int32_t>(num_tokens, 7),
                  torch::Tensor(),
                  MMData(),
                  std::move(decoder),
                  seq_params);
}

std::vector<int32_t> placed_ids(const Sequence& sequence,
                                size_t target_blocks) {
  const Slice<Block> blocks =
      sequence.host_kv_state().blocks(BlockType::LINEAR);
  std::vector<int32_t> ids;
  ids.reserve(target_blocks);
  for (size_t index = 0; index < target_blocks; ++index) {
    if (index < blocks.size() && blocks[index].is_valid()) {
      ids.push_back(blocks[index].id());
    } else {
      ids.push_back(-1);
    }
  }
  return ids;
}

}  // namespace

TEST(LinearStatePrefixCacheTest, PaddingSlotReservedAndLiveSlotsExhaust) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));

  std::vector<Block> first_blocks = cache.allocate(1);
  std::vector<Block> second_blocks = cache.allocate(1);
  ASSERT_EQ(first_blocks.size(), 1u);
  ASSERT_EQ(second_blocks.size(), 1u);
  Block first = std::move(first_blocks.front());
  Block second = std::move(second_blocks.front());
  ASSERT_TRUE(first.is_valid());
  ASSERT_TRUE(second.is_valid());
  EXPECT_GE(first.id(), 1);
  EXPECT_GE(second.id(), 1);
  EXPECT_NE(first.id(), second.id());

  // No free slot and no checkpoint to reclaim -> invalid handle.
  EXPECT_TRUE(cache.allocate(1).empty());
}

TEST(LinearStatePrefixCacheTest, DroppingLiveHandleReleasesSlot) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  int32_t reused_id = -1;
  {
    std::vector<Block> live_blocks = cache.allocate(1);
    ASSERT_EQ(live_blocks.size(), 1u);
    Block live = std::move(live_blocks.front());
    ASSERT_TRUE(live.is_valid());
    reused_id = live.id();
    // Only one usable slot; it is now live.
    EXPECT_TRUE(cache.allocate(1).empty());
  }
  // Handle dropped: the slot is free again.
  std::vector<Block> again = cache.allocate(1);
  ASSERT_EQ(again.size(), 1u);
  EXPECT_EQ(again.front().id(), reused_id);
}

TEST(LinearStatePrefixCacheTest, InsertMatchAndDedup) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/4, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);

  const int32_t slot = insert_checkpoint(cache, h1);
  EXPECT_GE(slot, 1);
  EXPECT_TRUE(block_test::contains(&cache, h1));
  Block pinned = block_test::match(&cache, h1);
  ASSERT_TRUE(pinned.is_valid());
  EXPECT_EQ(pinned.id(), slot);
  EXPECT_TRUE(pinned.is_shared());

  // Submit a different physical block with the same hash to the cache itself.
  const size_t free_before = cache.num_free_blocks();
  std::vector<Block> duplicate = cache.allocate(1);
  ASSERT_EQ(duplicate.size(), 1u);
  EXPECT_NE(duplicate.front().id(), slot);
  duplicate.front().set_hash_value(h1.data);
  cache.cache(duplicate);
  duplicate.clear();
  EXPECT_EQ(cache.num_blocks_in_prefix_cache(), 1u);
  EXPECT_EQ(cache.num_free_blocks(), free_before);
  EXPECT_EQ(block_test::match(&cache, h1).id(), slot);

  // Unknown hashes miss.
  EXPECT_FALSE(block_test::contains(&cache, make_hash(2)));
  EXPECT_FALSE(block_test::match(&cache, make_hash(2)).is_valid());
}

TEST(LinearStatePrefixCacheTest, InsertEvictsLeastRecentlyUsed) {
  // 3 physical slots -> 2 usable checkpoint slots.
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  insert_checkpoint(cache, h1);
  insert_checkpoint(cache, h2);
  ASSERT_TRUE(block_test::contains(&cache, h1));
  ASSERT_TRUE(block_test::contains(&cache, h2));

  // Cache full: a third checkpoint evicts the oldest (h1).
  insert_checkpoint(cache, h3);
  EXPECT_FALSE(block_test::contains(&cache, h1));
  EXPECT_TRUE(block_test::contains(&cache, h2));
  EXPECT_TRUE(block_test::contains(&cache, h3));
}

TEST(LinearStatePrefixCacheTest, MatchRefreshesLruOrder) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);
  const XXH3Key h3 = make_hash(3);

  insert_checkpoint(cache, h1);
  insert_checkpoint(cache, h2);
  // Touch h1 so h2 becomes the least-recently-used entry.
  {
    Block matched = block_test::match(&cache, h1);
    ASSERT_TRUE(matched.is_valid());
  }
  // With the hit released, survival must come from recency, not a pin.
  insert_checkpoint(cache, h3);
  EXPECT_TRUE(block_test::contains(&cache, h1));
  EXPECT_FALSE(block_test::contains(&cache, h2));
  EXPECT_TRUE(block_test::contains(&cache, h3));
}

TEST(LinearStatePrefixCacheTest, AcquireLiveSlotReclaimsCheckpoint) {
  // 2 physical slots -> 1 usable slot.
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);

  const int32_t checkpoint_slot = insert_checkpoint(cache, h1);
  ASSERT_GE(checkpoint_slot, 1);
  ASSERT_TRUE(block_test::contains(&cache, h1));

  // The only usable slot holds a checkpoint; acquiring a live slot must reclaim
  // it.
  std::vector<Block> live_blocks = cache.allocate(1);
  ASSERT_EQ(live_blocks.size(), 1u);
  Block live = std::move(live_blocks.front());
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), checkpoint_slot);
  EXPECT_FALSE(block_test::contains(&cache, h1));
}

TEST(LinearStatePrefixCacheTest, AcquireLiveSlotSkipsPinnedCheckpoint) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/3, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t pinned_slot = insert_checkpoint(cache, h1);
  ASSERT_GE(pinned_slot, 1);
  Block restore_pin = block_test::match(&cache, h1);
  ASSERT_TRUE(restore_pin.is_valid());
  EXPECT_EQ(restore_pin.id(), pinned_slot);

  const int32_t reclaimable_slot = insert_checkpoint(cache, h2);
  ASSERT_GE(reclaimable_slot, 1);

  std::vector<Block> live_blocks = cache.allocate(1);
  ASSERT_EQ(live_blocks.size(), 1u);
  Block live = std::move(live_blocks.front());
  ASSERT_TRUE(live.is_valid());
  EXPECT_EQ(live.id(), reclaimable_slot);
  EXPECT_TRUE(block_test::contains(&cache, h1));
  EXPECT_FALSE(block_test::contains(&cache, h2));
}

TEST(LinearStatePrefixCacheTest, InsertCannotReclaimLiveSlots) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  std::vector<Block> live_blocks = cache.allocate(1);
  ASSERT_EQ(live_blocks.size(), 1u);
  Block live = std::move(live_blocks.front());
  ASSERT_TRUE(live.is_valid());

  // A published checkpoint cannot steal a live sequence slot.
  EXPECT_EQ(insert_checkpoint(cache, make_hash(1)), -1);
  EXPECT_FALSE(block_test::contains(&cache, make_hash(1)));
}

TEST(LinearStatePrefixCacheTest, ReleasedPinAllowsCheckpointReclaim) {
  LinearStateBlockManager cache(
      linear_options(/*num_slots=*/2, /*block_size=*/16));
  const XXH3Key h1 = make_hash(1);
  const XXH3Key h2 = make_hash(2);

  const int32_t slot = insert_checkpoint(cache, h1);
  ASSERT_GE(slot, 1);
  {
    Block restore_pin = block_test::match(&cache, h1);
    ASSERT_TRUE(restore_pin.is_valid());
    EXPECT_EQ(insert_checkpoint(cache, h2), -1);
  }

  EXPECT_EQ(insert_checkpoint(cache, h2), slot);
  EXPECT_FALSE(block_test::contains(&cache, h1));
  EXPECT_TRUE(block_test::contains(&cache, h2));
}

// batch 8, V 2, stride 4, two groups. Valid slots sit at the end of each
// stride and batch n reuses batch n-2. A short tail that cannot fill one
// stride stays invalid.
TEST(LinearStatePrefixCacheTest, PrefetchCyclesTwoGroupsPastTheFirstPair) {
  BlockManager::Options options =
      linear_options(/*num_slots=*/8, /*block_size=*/128);
  options.enable_prefix_cache(true)
      .prefetch_batch_size(8)
      .prefetch_valid_blocks_per_batch(2);
  LinearStateBlockManager cache(options);
  Sequence sequence = make_prefetch_sequence(/*num_tokens=*/20 * 128);

  ASSERT_TRUE(cache.allocate_for_prefetch(&sequence, /*num_tokens=*/20 * 128));
  const std::vector<int32_t> ids = placed_ids(sequence, /*target_blocks=*/20);
  ASSERT_EQ(ids.size(), 20u);

  std::vector<int32_t> group0;
  std::vector<int32_t> group1;
  group0.reserve(2);
  group1.reserve(2);
  for (size_t index = 0; index < ids.size(); ++index) {
    const bool slot_end = (index + 1) % 4 == 0;
    if (!slot_end) {
      EXPECT_EQ(ids[index], -1) << "index " << index;
      continue;
    }
    EXPECT_GE(ids[index], 1) << "index " << index;
    const size_t batch = index / 8;
    if (batch % 2 == 0) {
      group0.push_back(ids[index]);
    } else {
      group1.push_back(ids[index]);
    }
  }
  ASSERT_EQ(group0.size(), 3u);
  ASSERT_EQ(group1.size(), 2u);
  EXPECT_EQ(group0[0], ids[3]);
  EXPECT_EQ(group0[1], ids[7]);
  EXPECT_EQ(group0[2], ids[19]);
  EXPECT_EQ(group0[2], group0[0]);
  EXPECT_EQ(group1[0], ids[11]);
  EXPECT_EQ(group1[1], ids[15]);
  EXPECT_NE(group0[0], group0[1]);
  EXPECT_NE(group1[0], group1[1]);
  EXPECT_NE(group0[0], group1[0]);
  EXPECT_NE(group0[1], group1[0]);
}

// A full host prefix cache is still reusable. Two groups must be allocated by
// evicting published checkpoints instead of collapsing onto one page.
TEST(LinearStatePrefixCacheTest, PrefetchReclaimsCachedSlotsForBothGroups) {
  BlockManager::Options options =
      linear_options(/*num_slots=*/5, /*block_size=*/128);
  options.enable_prefix_cache(true)
      .prefetch_batch_size(8)
      .prefetch_valid_blocks_per_batch(2);
  LinearStateBlockManager cache(options);
  // Padding occupies slot 0. Publish the other four so the free list is empty.
  for (uint8_t tag = 1; tag <= 4; ++tag) {
    ASSERT_GE(insert_checkpoint(cache, make_hash(tag)), 1);
  }
  EXPECT_EQ(cache.num_free_blocks(), 0u);
  EXPECT_GE(cache.num_available_blocks(), 4u);

  Sequence sequence = make_prefetch_sequence(/*num_tokens=*/16 * 128);
  ASSERT_TRUE(cache.allocate_for_prefetch(&sequence, /*num_tokens=*/16 * 128));
  const std::vector<int32_t> ids = placed_ids(sequence, /*target_blocks=*/16);
  ASSERT_GE(ids[3], 1);
  ASSERT_GE(ids[7], 1);
  ASSERT_GE(ids[11], 1);
  ASSERT_GE(ids[15], 1);
  EXPECT_NE(ids[3], ids[7]);
  EXPECT_NE(ids[3], ids[11]);
  EXPECT_NE(ids[7], ids[11]);
  EXPECT_NE(ids[3], ids[15]);
  EXPECT_NE(ids[7], ids[15]);
  EXPECT_NE(ids[11], ids[15]);
}

// 15 KV blocks end inside a stride. Checkpoints stay on the stride grid, so
// the last partial stride is left empty. With stride 4 that is 512 tokens.
TEST(LinearStatePrefixCacheTest, IncompleteStrideStaysEmpty) {
  BlockManager::Options options =
      linear_options(/*num_slots=*/8, /*block_size=*/128);
  options.enable_prefix_cache(true)
      .prefetch_batch_size(8)
      .prefetch_valid_blocks_per_batch(2);
  LinearStateBlockManager cache(options);
  Sequence sequence = make_prefetch_sequence(/*num_tokens=*/15 * 128);

  ASSERT_TRUE(cache.allocate_for_prefetch(&sequence, /*num_tokens=*/15 * 128));
  const std::vector<int32_t> ids = placed_ids(sequence, /*target_blocks=*/15);
  for (size_t index = 0; index < ids.size(); ++index) {
    const bool checkpoint = index == 3 || index == 7 || index == 11;
    if (!checkpoint) {
      EXPECT_EQ(ids[index], -1) << "index " << index;
    }
  }
  ASSERT_GE(ids[3], 1);
  ASSERT_GE(ids[7], 1);
  ASSERT_GE(ids[11], 1);
  EXPECT_NE(ids[3], ids[7]);
  EXPECT_NE(ids[3], ids[11]);
  EXPECT_NE(ids[7], ids[11]);
}

}  // namespace xllm
