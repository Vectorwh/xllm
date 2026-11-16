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

#include <cstring>
#include <numeric>
#include <vector>

#include "framework/block/block_manager_impl.h"
#include "util/hash_util.h"

namespace xllm {
namespace {

BlockManager::Options cache_options(
    uint32_t physical_blocks,
    int32_t block_size,
    BlockHasherType hasher_type = BlockHasherType::TEXT) {
  BlockManager::Options options;
  options.num_blocks(physical_blocks)
      .block_size(block_size)
      .enable_prefix_cache(true)
      .hasher_type(hasher_type);
  return options;
}

}  // namespace

TEST(PrefixCacheProbeTest, ProbeDoesNotPinOrPromoteBlocks) {
  BlockManagerImpl cache(cache_options(3, 16));
  const std::vector<int32_t> first(16, 7);
  const std::vector<int32_t> second(16, 8);
  auto first_blocks = cache.allocate(1);
  auto second_blocks = cache.allocate(1);
  cache.cache(first, first_blocks);
  cache.cache(second, second_blocks);
  std::vector<XXH3Key> hashes;
  extend_prefix_hashes(BlockHasherType::TEXT, MMData(), first, 16, 1, hashes);
  const uint32_t references = first_blocks.front().ref_count();
  EXPECT_EQ(cache.probe_prefix_hits(hashes), (std::vector<uint8_t>{1}));
  EXPECT_EQ(first_blocks.front().ref_count(), references);
  first_blocks.clear();
  second_blocks.clear();
  ASSERT_EQ(cache.num_evictable_blocks(), 2u);
  auto pressure = cache.allocate(1);
  ASSERT_EQ(pressure.size(), 1u);
  EXPECT_TRUE(cache.probe_prefix_hits(hashes).empty());
  EXPECT_EQ(cache.allocate_shared(second).size(), 1u);
}

TEST(PrefixCacheMtpTest, DifferentNextTokenCannotReuseFirstBlock) {
  BlockManagerImpl cache(cache_options(8, 16, BlockHasherType::MTP_TEXT));
  std::vector<int32_t> tokens(33, 7);
  auto blocks = cache.allocate(/*num_blocks=*/2);
  cache.cache(tokens, blocks);
  ASSERT_EQ(cache.num_blocks_in_prefix_cache(), 2u);
  ASSERT_EQ(cache.allocate_shared(tokens).size(), 2u);

  tokens[16] = 8;
  EXPECT_TRUE(cache.allocate_shared(tokens).empty());
  for (const Block& block : blocks) {
    EXPECT_EQ(block.ref_count(), 2u);
  }
  blocks.clear();
  cache.reset_prefix_cache();
  EXPECT_EQ(cache.num_blocks_in_prefix_cache(), 0u);
}

TEST(PrefixCacheMtpTest, RequiresNextTokenEvenWithPrecomputedHashes) {
  BlockManagerImpl cache(cache_options(8, 16, BlockHasherType::MTP_TEXT));
  const std::vector<int32_t> tokens(33, 7);
  std::vector<XXH3Key> hashes;
  extend_prefix_hashes(BlockHasherType::MTP_TEXT,
                       MMData(),
                       tokens,
                       /*block_size=*/16,
                       /*boundary_blocks=*/2,
                       hashes);
  auto blocks = cache.allocate(/*num_blocks=*/2);
  const std::vector<size_t> lengths = {0, 1, 15, 16, 17, 31, 32, 33};
  const std::vector<size_t> ready_blocks = {0, 0, 0, 0, 1, 1, 1, 2};
  for (size_t i = 0; i < lengths.size(); ++i) {
    SCOPED_TRACE(lengths[i]);
    const Slice<int32_t> available(tokens, lengths[i]);
    cache.cache(available, blocks, /*existed=*/0, MMData(), hashes);
    EXPECT_EQ(cache.num_blocks_in_prefix_cache(), ready_blocks[i]);
    EXPECT_EQ(cache.allocate_shared(available, {}, MMData(), hashes).size(),
              ready_blocks[i]);
  }
}

TEST(PrefixCacheMtpTest, ForkedRequestsNeverMixIncompatibleBlocks) {
  BlockManagerImpl cache(cache_options(16, 16, BlockHasherType::MTP_TEXT));
  std::vector<int32_t> tokens_a(33, 7);
  tokens_a[16] = 11;
  auto blocks_a = cache.allocate(/*num_blocks=*/2);
  cache.cache(tokens_a, blocks_a);
  ASSERT_EQ(cache.num_blocks_in_prefix_cache(), 2u);

  std::vector<int32_t> tokens_b = tokens_a;
  tokens_b[16] = 12;
  auto blocks_b = cache.allocate(/*num_blocks=*/2);
  cache.cache(tokens_b, blocks_b);
  ASSERT_EQ(cache.num_blocks_in_prefix_cache(), 4u);
  const auto matched = cache.allocate_shared(tokens_b);
  ASSERT_EQ(matched.size(), 2u);
  EXPECT_EQ(matched[0].id(), blocks_b[0].id());
  EXPECT_EQ(matched[1].id(), blocks_b[1].id());

  tokens_b[32] = 13;
  const auto forked = cache.allocate_shared(tokens_b);
  ASSERT_EQ(forked.size(), 1u);
  EXPECT_EQ(forked[0].id(), blocks_b[0].id());
}

TEST(PrefixCacheMtpTest, IncludesLookaheadAndMatchesPrecomputedChain) {
  BlockManagerImpl cache(cache_options(8, 16, BlockHasherType::MTP_TEXT));
  const std::vector<int32_t> tokens(33, 7);
  std::vector<XXH3Key> hashes;
  extend_prefix_hashes(BlockHasherType::MTP_TEXT,
                       MMData(),
                       tokens,
                       /*block_size=*/16,
                       /*boundary_blocks=*/2,
                       hashes);
  ASSERT_EQ(hashes.size(), 2u);
  XXH3Key expected;
  xxh3_128bits_hash(/*pre_hash_value=*/nullptr,
                    Slice<int32_t>(tokens, /*size=*/17),
                    expected.data);
  EXPECT_EQ(hashes[0], expected);
  auto blocks = cache.allocate(/*num_blocks=*/2);
  cache.cache(tokens, blocks);
  ASSERT_EQ(cache.num_blocks_in_prefix_cache(), 2u);
  const auto matched = cache.allocate_shared(tokens, {}, MMData(), hashes);
  ASSERT_EQ(matched.size(), 2u);
  EXPECT_EQ(matched[0].id(), blocks[0].id());
  EXPECT_EQ(matched[1].id(), blocks[1].id());
}

TEST(PrefixCacheMtpTest, ResumedMultimodalMatchKeepsOverlappingContent) {
  MMData mm_data;
  auto& item = mm_data.add(MMType::IMAGE);
  item.mutable_state().mutable_token_pos() = {16, 1};
  std::memset(item.mutable_state().mutable_schedule_data().key.data,
              1,
              XXH3_128BITS_HASH_VALUE_LEN);
  BlockManagerImpl cache(cache_options(8, 16, BlockHasherType::MTP_MM));
  const std::vector<int32_t> tokens(33, 7);
  auto blocks = cache.allocate(/*num_blocks=*/2);
  cache.cache(tokens, blocks, /*existed=*/0, mm_data);
  ASSERT_EQ(cache.num_blocks_in_prefix_cache(), 2u);
  const auto first =
      cache.allocate_shared(Slice<int32_t>(tokens, /*size=*/17), {}, mm_data);
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(cache.allocate_shared(tokens, first, mm_data).size(), 2u);
  item.mutable_state().mutable_schedule_data().key.data[0] = 2;
  EXPECT_TRUE(cache.allocate_shared(tokens, {}, mm_data).empty());
}

// Build the chained block hashes for `tokens`, matching the chain produced by
// xxh3_128bits_hash() inside PrefixCache and Sequence::update_block_hashes().
namespace {

std::vector<XXH3Key> build_chained_hashes(const std::vector<int32_t>& tokens,
                                          uint32_t block_size) {
  const size_t n_blocks = tokens.size() / block_size;
  std::vector<XXH3Key> hashes;
  hashes.reserve(n_blocks);
  const Slice<int32_t> slice(tokens);
  for (size_t b = 0; b < n_blocks; ++b) {
    XXH3Key key;
    const uint8_t* pre = (b == 0) ? nullptr : hashes.back().data;
    xxh3_128bits_hash(
        pre, slice.slice(b * block_size, (b + 1) * block_size), key.data);
    hashes.emplace_back(key);
  }
  return hashes;
}

std::vector<int32_t> make_tokens(size_t count) {
  std::vector<int32_t> tokens(count);
  std::iota(tokens.begin(), tokens.end(), 1);
  return tokens;
}

}  // namespace

TEST(PrefixCacheTest, InsertOperation) {
  BlockManagerImpl cache(cache_options(5, 4));
  const std::vector<int32_t> first{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<int32_t> second{9, 10, 11, 12, 13, 14, 15, 16, 17};
  EXPECT_TRUE(cache.allocate_shared(first).empty());
  {
    auto blocks = cache.allocate(2);
    ASSERT_EQ(blocks.size(), 2u);
    cache.cache(first, blocks);
  }
  {
    auto blocks = cache.allocate(2);
    ASSERT_EQ(blocks.size(), 2u);
    cache.cache(second, blocks);
  }
  {
    auto matched = cache.allocate_shared(first);
    ASSERT_EQ(matched.size(), 2u);
    // Reinserting a hit must not add another resident for the same chain.
    cache.cache(first, matched);
    EXPECT_EQ(cache.num_blocks_in_prefix_cache(), 4u);
  }
  // A touched prefix survives; the colder chain loses its tail first.
  auto first_pressure = cache.allocate(1);
  ASSERT_EQ(first_pressure.size(), 1u);
  EXPECT_EQ(cache.allocate_shared(second).size(), 1u);
  EXPECT_EQ(cache.allocate_shared(first).size(), 2u);
  auto second_pressure = cache.allocate(1);
  ASSERT_EQ(second_pressure.size(), 1u);
  EXPECT_TRUE(cache.allocate_shared(second).empty());
}

TEST(PrefixCacheTest, EvictOperation) {
  BlockManagerImpl cache(cache_options(3, 4));
  const std::vector<int32_t> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto blocks = cache.allocate(2);
  ASSERT_EQ(blocks.size(), 2u);
  cache.cache(tokens, blocks);
  EXPECT_TRUE(cache.allocate(1).empty());
  blocks.clear();
  ASSERT_EQ(cache.num_evictable_blocks(), 2u);
  auto pressure = cache.allocate(2);
  ASSERT_EQ(pressure.size(), 2u);
  EXPECT_EQ(cache.num_blocks_in_prefix_cache(), 0u);
  EXPECT_TRUE(cache.allocate_shared(tokens).empty());
  pressure.clear();
  EXPECT_EQ(cache.num_free_blocks(), cache.num_total_blocks());
}

TEST(HashUtilTest, XXHash3) {
  const std::vector<int32_t> parent{1, 2, 3, 4};
  const std::vector<int32_t> child{5, 6, 7, 8};
  XXH3Key parent_hash;
  XXH3Key chained_hash;
  XXH3Key unchained_hash;
  xxh3_128bits_hash(nullptr, parent, parent_hash.data);
  xxh3_128bits_hash(parent_hash.data, child, chained_hash.data);
  xxh3_128bits_hash(nullptr, child, unchained_hash.data);
  EXPECT_FALSE(chained_hash == unchained_hash);
}

TEST(PrefixCacheTest, PrecomputedHashMatchesComputedMatch) {
  const uint32_t block_size = 4;
  const uint32_t n_blocks = 4;
  BlockManager::Options options;
  options.num_blocks(n_blocks + 1)
      .block_size(block_size)
      .enable_prefix_cache(true);
  BlockManagerImpl prefix_cache(options);

  std::vector<int32_t> token_ids = make_tokens(n_blocks * block_size);
  const Slice<int32_t> tokens(token_ids);
  const std::vector<XXH3Key> hashes =
      build_chained_hashes(token_ids, block_size);
  const Slice<XXH3Key> hash_slice(hashes);

  // Release all external aliases so eviction exercises cache-only residents.
  {
    std::vector<Block> blocks = prefix_cache.allocate(n_blocks);
    prefix_cache.cache(tokens, blocks);
  }

  // Full hits must agree without leaving pins behind for the partial hit.
  {
    auto full_computed = prefix_cache.allocate_shared(tokens);
    auto full_precomputed =
        prefix_cache.allocate_shared(tokens, {}, MMData(), hash_slice);
    ASSERT_EQ(full_computed.size(), n_blocks);
    ASSERT_EQ(full_precomputed.size(), full_computed.size());
    for (size_t i = 0; i < full_computed.size(); ++i) {
      EXPECT_EQ(full_computed[i].id(), full_precomputed[i].id());
    }
  }

  // Both paths must stop at the same first miss.
  auto pressure = prefix_cache.allocate(2);
  ASSERT_EQ(pressure.size(), 2u);
  auto partial_computed = prefix_cache.allocate_shared(tokens);
  auto partial_precomputed =
      prefix_cache.allocate_shared(tokens, {}, MMData(), hash_slice);
  EXPECT_LT(partial_computed.size(), n_blocks);
  EXPECT_GT(partial_computed.size(), 0u);
  ASSERT_EQ(partial_precomputed.size(), partial_computed.size());
  for (size_t i = 0; i < partial_computed.size(); ++i) {
    EXPECT_EQ(partial_computed[i].id(), partial_precomputed[i].id());
  }
}

TEST(PrefixCacheTest, PrecomputedHashInsertMatchesComputedInsert) {
  const uint32_t block_size = 4;
  const uint32_t n_blocks = 3;
  BlockManager::Options options;
  options.num_blocks(n_blocks + 1)
      .block_size(block_size)
      .enable_prefix_cache(true);

  std::vector<int32_t> token_ids = make_tokens(n_blocks * block_size);
  const Slice<int32_t> tokens(token_ids);
  const std::vector<XXH3Key> hashes =
      build_chained_hashes(token_ids, block_size);
  const Slice<XXH3Key> hash_slice(hashes);

  // Cache A: insert via the on-the-fly compute path.
  BlockManagerImpl cache_compute(options);
  std::vector<Block> blocks_a = cache_compute.allocate(n_blocks);
  cache_compute.cache(tokens, blocks_a);

  // Cache B: insert via the precomputed path.
  BlockManagerImpl cache_precomputed(options);
  std::vector<Block> blocks_b = cache_precomputed.allocate(n_blocks);
  cache_precomputed.cache(
      tokens, blocks_b, /*existed=*/0, MMData(), hash_slice);

  EXPECT_EQ(cache_compute.num_blocks_in_prefix_cache(), n_blocks);
  EXPECT_EQ(cache_precomputed.num_blocks_in_prefix_cache(), n_blocks);

  for (size_t i = 0; i < n_blocks; ++i) {
    // Both insert paths must stamp the same hash onto each block ...
    EXPECT_EQ(std::memcmp(blocks_a[i].get_immutable_hash_value(),
                          blocks_b[i].get_immutable_hash_value(),
                          XXH3_128BITS_HASH_VALUE_LEN),
              0);
    // ... and it must equal the independently computed chain.
    EXPECT_EQ(std::memcmp(blocks_b[i].get_immutable_hash_value(),
                          hashes[i].data,
                          XXH3_128BITS_HASH_VALUE_LEN),
              0);
  }

  // A full match against the precomputed-built cache hits every block.
  auto matched =
      cache_precomputed.allocate_shared(tokens, {}, MMData(), hash_slice);
  EXPECT_EQ(matched.size(), n_blocks);
}

}  // namespace xllm
