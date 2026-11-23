/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/parallel_state/linear_attention_parallel.h"
#include "framework/parallel_state/parallel_args.h"
#include "platform/platform.h"
#include "util/utils.h"

namespace xllm {

CacheTensorLayoutContext make_cache_tensor_layout_context(
    const ParallelArgs& parallel_args,
    const ModelArgs& model_args,
    int64_t block_token_capacity,
    const LinearAttentionPolicy& linear_policy);

void apply_cache_shape_layout(const KVCacheShape& shape,
                              CacheTensorLayoutContext* context);

// Grouped caches use each tensor's physical token-axis length. Ordinary caches
// retain the configured capacity for producer-capacity validation.
int64_t effective_block_token_capacity(const CacheTensorLayoutContext& context,
                                       const KVCacheTensor& cache_tensor);

// Adds an explicit logical descriptor to a cache tensor produced by KVCache.
// Returns false with an actionable error when the tensor cannot be represented
// without guessing an unsupported layout.
bool describe_cache_tensor(const CacheTensorLayoutContext& context,
                           KVCacheTensor* cache_tensor,
                           std::string* error);

// Physical index pages owned by one logical cache block. The NPU keeps the
// whole DSA indexer cache on every rank, so a logical block B maps to index
// pages [B * split, (B + 1) * split) (see platform.h). Every other backend
// allocates a single index page per block. This is the allocation predicate of
// KVCacheShape::init_index_cache_shape() and the transfer geometry of
// describe_cache_tensor(), so both consume this one definition. Defined inline
// here on purpose: kv_cache_estimation.cpp lives in :kv_cache_estimation, which
// :kv_cache already depends on, so a .cpp definition in this library would make
// the dependency a cycle. Both callees below are header-only.
inline int64_t indexer_pages_per_block() {
  if (!Platform::requires_dsa_indexer_cache_replication()) {
    return 1;
  }
  return std::max<int32_t>(util::kv_split_size_effective(), 1);
}

// Physical tensor rows one logical cache block spans for `role`, the row-count
// companion of the logical descriptor describe_cache_tensor() writes for the
// same role: SSM packs its checkpoint stride into one block, the NPU indexer
// cache covers every kv_split page, and a replicated pool (a standalone
// drafter) covers the whole logical block. Every other role stores one row per
// logical block.
int64_t physical_rows_per_resource(KVCacheTensorRole role,
                                   int64_t replicated_block_pages,
                                   int64_t ssm_checkpoint_stride);

}  // namespace xllm
