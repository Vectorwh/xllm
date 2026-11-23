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

#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"

namespace xllm {

// Model/cache construction supplies capacities; Host copy groups only consume
// these logical resource counts and the actual persistent tensor views.
struct HostCachePlan {
  std::map<BlockType, int64_t> block_counts;
  int64_t block_size = 0;
  int64_t window_size = 0;
  int64_t window_block_size = 0;
  bool kv_tp_replicated = false;
};

HostCachePlan build_host_cache_plan(const std::vector<KVCache>& caches,
                                    const KVCacheShape& shape,
                                    const KVCacheCreateOptions& options,
                                    double host_blocks_factor,
                                    InstanceRole instance_role,
                                    bool enable_mla,
                                    const HostCachePlan* target_plan = nullptr);

}  // namespace xllm
