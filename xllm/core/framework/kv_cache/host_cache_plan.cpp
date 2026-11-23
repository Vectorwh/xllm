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

#include "framework/kv_cache/host_cache_plan.h"

#include <glog/logging.h>

#include <limits>

#include "framework/config/scheduler_config.h"
#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "platform/platform.h"
#include "util/utils.h"

namespace xllm {

HostCachePlan build_host_cache_plan(const std::vector<KVCache>& caches,
                                    const KVCacheShape& shape,
                                    const KVCacheCreateOptions& options,
                                    double host_blocks_factor,
                                    InstanceRole instance_role,
                                    bool enable_mla,
                                    const HostCachePlan* target_plan) {
  HostCachePlan plan;
  plan.block_size = options.block_size();
  plan.window_size = options.window_size();
  plan.window_block_size = options.block_size();
  plan.kv_tp_replicated = enable_mla;
  std::map<BlockType, int64_t> device_counts;
  for (const KVCache& cache : caches) {
    for (const BlockType type :
         {BlockType::KV, BlockType::SWA, BlockType::LINEAR}) {
      const BlockTypeTensorMap tensors =
          cache.get_block_type_tensors(type, /*checkpoint_row=*/0);
      for (const auto& [role, tensor] : tensors) {
        const torch::Tensor view =
            shape.has_grouped_cache_layout()
                ? tensor
                : fold_tensor_rows_for_transfer(
                      tensor, role, shape.replicated_block_pages());
        CHECK_GT(view.dim(), 1);
        CHECK_GT(view.size(0), 0);
        const auto [it, inserted] = device_counts.emplace(type, view.size(0));
        CHECK(inserted || it->second == view.size(0))
            << "Persistent planes of a resource must share logical block IDs.";
      }
    }
  }

  for (const auto& [type, count] : device_counts) {
    plan.block_counts.emplace(
        type, scale_host_block_count(count, host_blocks_factor));
  }

  if (shape.has_grouped_cache_layout()) {
    const bool npu_dsv4 =
        !util::is_deepseek_v41_family_model_type(options.model_type()) &&
        Platform::is_npu();
    plan.block_size = dsv4_kv_block_size(npu_dsv4);
    plan.window_block_size = dsv4_swa_block_size();
    // Owned DSV KV was TP-sharded before logical resource unification.
    plan.kv_tp_replicated = false;
    const auto kv = device_counts.find(BlockType::KV);
    if (kv != device_counts.end() && device_counts.count(BlockType::SWA) != 0) {
      plan.block_counts.at(BlockType::SWA) =
          dsv4_host_swa_block_count(kv->second,
                                    host_blocks_factor,
                                    options.window_size(),
                                    plan.window_block_size,
                                    plan.block_size);
    }
  }

  // A draft that owns SWA follows the target's SWA pool. A target without SWA
  // still publishes KV block ids, and the draft SWA pool uses that count.
  if (target_plan != nullptr && device_counts.count(BlockType::SWA) != 0) {
    const auto target_swa = target_plan->block_counts.find(BlockType::SWA);
    if (target_swa != target_plan->block_counts.end()) {
      plan.block_counts.insert_or_assign(BlockType::SWA, target_swa->second);
    } else {
      const auto target_kv = target_plan->block_counts.find(BlockType::KV);
      CHECK(target_kv != target_plan->block_counts.end())
          << "Draft SWA requires target SWA or target KV block count.";
      plan.block_counts.insert_or_assign(BlockType::SWA, target_kv->second);
    }
  }

  const auto linear = device_counts.find(BlockType::LINEAR);
  if (linear != device_counts.end()) {
    if (instance_role == InstanceRole::DECODE) {
      CHECK_LE(linear->second, std::numeric_limits<int64_t>::max() / 2);
      plan.block_counts.at(BlockType::LINEAR) = linear->second * 2;
    } else {
      // Counted KV blocks size the prefill checkpoint pool. A pool that only
      // published a LINEAR plane keeps that counted row count.
      const auto kv_count = device_counts.find(BlockType::KV);
      if (kv_count != device_counts.end() || shape.has_key_cache_shape()) {
        const int64_t kv_blocks = kv_count != device_counts.end()
                                      ? kv_count->second
                                      : shape.key_cache_shape()[0];
        const int64_t host_kv_blocks =
            scale_host_block_count(kv_blocks, host_blocks_factor);
        plan.block_counts
            .at(BlockType::LINEAR) = prefill_host_linear_block_count(
            host_kv_blocks,
            options.block_size(),
            SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill());
      }
    }
  }
  if (target_plan != nullptr && plan.block_counts.count(BlockType::KV) != 0 &&
      target_plan->block_counts.count(BlockType::KV) != 0) {
    CHECK_EQ(plan.block_size, target_plan->block_size)
        << "Target/draft KV must share the block size.";
    CHECK_GE(plan.block_counts.at(BlockType::KV),
             target_plan->block_counts.at(BlockType::KV))
        << "Draft KV must cover every target logical block ID.";
  }

  return plan;
}

}  // namespace xllm
