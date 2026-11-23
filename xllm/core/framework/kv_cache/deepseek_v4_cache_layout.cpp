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

#include "framework/kv_cache/deepseek_v4_cache_layout.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "framework/kv_cache/deepseek_v4_cache_geometry.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "platform/platform.h"

namespace xllm {

Dsv4LayerCacheLayout build_dsv4_layer_cache_layout(
    const Dsv4CacheLayoutOptions& options) {
  CHECK_GT(options.head_dim, 0);
  Dsv4LayerCacheLayout layout;
  layout.tensors.reserve(8);
  auto add_tensor = [&](BlockType type,
                        KVCacheTensorRole role,
                        std::vector<int64_t> shape,
                        torch::ScalarType dtype) {
    layout.tensors.emplace_back(
        CacheTensorAllocationSpec{type, role, std::move(shape), dtype});
  };
  auto add_kv = [&](BlockType type,
                    KVCacheTensorRole role,
                    int64_t rows,
                    int64_t width,
                    torch::ScalarType dtype) {
    std::vector<int64_t> shape = Platform::is_mlu()
                                     ? std::vector<int64_t>{1, rows, width}
                                     : std::vector<int64_t>{rows, 1, width};
    add_tensor(type, role, std::move(shape), dtype);
  };
  auto add_state = [&](KVCacheTensorRole packed_role,
                       KVCacheTensorRole kv_role,
                       KVCacheTensorRole score_role,
                       int64_t rows,
                       int64_t width) {
    if (Platform::is_mlu()) {
      add_tensor(
          BlockType::SWA, packed_role, {rows, 2 * width}, torch::kFloat32);
      return;
    }
    add_tensor(BlockType::SWA, kv_role, {rows, width}, torch::kFloat32);
    add_tensor(BlockType::SWA, score_role, {rows, width}, torch::kFloat32);
  };

  const int32_t ratio = options.compress_ratio;
  const bool has_kv_cache =
      options.is_v41 ? options.is_kv_source_layer && (ratio == 1 || ratio == 2)
                     : ratio == 4 || ratio == 128;
  // NPU DSV4 KV is 2048 tokens. Every other KV span, and every SWA span, is
  // 128. Physical dim is the KV span divided by the compress ratio.
  const bool npu_dsv4 = !options.is_v41 && Platform::is_npu();
  const int64_t kv_block_size = dsv4_kv_block_size(npu_dsv4);
  const int64_t swa_block_size = dsv4_swa_block_size();
  layout.block_size = kv_block_size;
  if (has_kv_cache) {
    const int64_t rows = dsv4_physical_dim(kv_block_size, ratio);
    add_kv(BlockType::KV,
           KVCacheTensorRole::KEY,
           rows,
           options.head_dim,
           options.dtype);
    if (options.is_v41 || ratio == 4) {
      const DeepSeekV4CachePolicy policy = get_dsv4_cache_policy(options.dtype);
      add_kv(BlockType::KV,
             KVCacheTensorRole::INDEX,
             rows,
             std::max<int64_t>(options.index_head_dim, 1),
             policy.index_dtype);
      if (policy.has_indexer_cache_scale) {
        add_tensor(BlockType::KV,
                   KVCacheTensorRole::INDEX_SCALE,
                   {rows, 1},
                   policy.scale_dtype);
      }
    }
  }
  add_kv(BlockType::SWA,
         KVCacheTensorRole::SWA,
         swa_block_size,
         options.head_dim,
         options.dtype);
  if (!has_kv_cache || (options.is_v41 && ratio == 1)) {
    return layout;
  }
  const int64_t state_rows =
      options.is_v41 ? kDsv41CompressStateRows : swa_block_size;
  const int64_t state_width =
      ratio == 4 ? 2 * options.head_dim : options.head_dim;
  add_state(KVCacheTensorRole::COMPRESS_STATE,
            KVCacheTensorRole::KV_STATE,
            KVCacheTensorRole::SCORE_STATE,
            state_rows,
            state_width);
  if (!options.is_v41 && ratio == 4) {
    add_state(KVCacheTensorRole::COMPRESS_INDEX_STATE,
              KVCacheTensorRole::INDEX_KV_STATE,
              KVCacheTensorRole::INDEX_SCORE_STATE,
              state_rows,
              2 * std::max<int64_t>(options.index_head_dim, 1));
  }
  return layout;
}

}  // namespace xllm
