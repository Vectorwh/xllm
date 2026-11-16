/* Copyright 2025-2026 The xLLM Authors.

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

#include "embedding_block_manager.h"

#include <gtest/gtest.h>

namespace xllm {

TEST(EmbeddingBlockManagerTest, AllocateAndFreeRoundTrip) {
  // id 0 is reserved for padding (matching BlockManagerImpl), so a pool sized
  // for 4 physical slots exposes 3 usable blocks.
  EmbeddingBlockManager manager(4, "single");

  EXPECT_EQ(manager.num_total_blocks(), 3);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0);
  EXPECT_EQ(manager.num_free_blocks(), 3);
  EXPECT_EQ(manager.num_used_blocks(), 0);
  EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 0.0);

  auto blocks = manager.allocate(2);
  ASSERT_EQ(blocks.size(), 2);
  for (const auto& block : blocks) {
    EXPECT_GE(block.id(), 1) << "padding slot 0 must never be allocated";
  }
  EXPECT_EQ(manager.num_free_blocks(), 1);
  EXPECT_EQ(manager.num_used_blocks(), 2);
  EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 2.0 / 3.0);

  blocks.clear();
  EXPECT_TRUE(blocks.empty());
  EXPECT_EQ(manager.num_free_blocks(), 3);
  EXPECT_EQ(manager.num_used_blocks(), 0);
  EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 0.0);
}

TEST(EmbeddingBlockManagerTest, AllocateReturnsEmptyWhenExhausted) {
  // id 0 is reserved internally, so 3 physical slots expose 2 usable blocks.
  EmbeddingBlockManager manager(3, "single");
  auto blocks = manager.allocate(2);
  ASSERT_EQ(blocks.size(), 2);
  EXPECT_TRUE(manager.allocate(1).empty());
}

TEST(EmbeddingBlockManagerTest, ConstructorRejectsZeroBlocks) {
  EXPECT_DEATH({ EmbeddingBlockManager manager(0, "single"); }, "No blocks");
}

}  // namespace xllm
