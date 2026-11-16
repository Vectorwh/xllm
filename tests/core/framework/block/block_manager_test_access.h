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

#pragma once

#include <utility>
#include <vector>

#include "framework/block/block_manager_pool.h"
#include "framework/block/composite_block_manager.h"
#include "framework/block/linear_state_block_manager.h"

namespace xllm::block_test {

inline BlockManager* leaf_of(const CompositeBlockManager& composite,
                             BlockType type) {
  const auto& leaves = composite.leaf_entries();
  auto it = leaves.find(type);
  return it == leaves.end() ? nullptr : it->second.leaf.get();
}

// Pool subclasses already own the protected manager topology. Keep this
// inspection in the test subclass rather than granting a production friend.
class TestBlockManagerPool final : public BlockManagerPool {
 public:
  using BlockManagerPool::BlockManagerPool;

  BlockManager* leaf_of(int32_t dp_rank, BlockType type) const {
    const auto* composite = static_cast<const CompositeBlockManager*>(
        block_managers_.at(static_cast<size_t>(dp_rank)).get());
    return block_test::leaf_of(*composite, type);
  }
};

inline BlockManager* leaf_of(BlockManagerPool& pool,
                             int32_t dp_rank,
                             BlockType type) {
  return dynamic_cast<TestBlockManagerPool&>(pool).leaf_of(dp_rank, type);
}

inline LinearStateBlockManager* linear_leaf(BlockManagerPool& pool,
                                            int32_t dp_rank) {
  if (!pool.options().enable_linear_state()) {
    return nullptr;
  }
  return static_cast<LinearStateBlockManager*>(
      leaf_of(pool, dp_rank, BlockType::LINEAR));
}

inline bool contains(const LinearStateBlockManager* leaf,
                     const XXH3Key& prefix_hash) {
  const auto hits =
      leaf->probe_prefix_hits(Slice<XXH3Key>(&prefix_hash, /*size=*/1));
  return !hits.empty() && hits.front() != 0;
}

// Exercise the same refcount and LRU path as serving. A completed LINEAR
// checkpoint needs one full stride plus the excluded live-write token.
inline Block match(LinearStateBlockManager* leaf, const XXH3Key& prefix_hash) {
  const std::vector<int32_t> tokens(static_cast<size_t>(leaf->block_size()) + 1,
                                    0);
  auto blocks = leaf->allocate_shared(
      tokens, {}, MMData(), Slice<XXH3Key>(&prefix_hash, /*size=*/1));
  return blocks.empty() ? Block() : std::move(blocks.front());
}

}  // namespace xllm::block_test
