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
#include <vector>

#include "framework/kv_cache/cache_tensor_allocation_spec.h"

namespace xllm {

struct Dsv4CacheLayoutOptions {
  int64_t head_dim = 0;
  int64_t index_head_dim = 0;
  int32_t compress_ratio = 1;
  torch::ScalarType dtype = torch::kBFloat16;
  bool is_v41 = false;
  bool is_kv_source_layer = false;
};

struct Dsv4LayerCacheLayout {
  int64_t block_size = 0;
  std::vector<CacheTensorAllocationSpec> tensors;
};

// Model-owned declaration shared by capacity planning and actual allocation.
// Only owned tensors appear here; V4.1 reuse layers declare their own window
// but no additional KV or compressor state.
Dsv4LayerCacheLayout build_dsv4_layer_cache_layout(
    const Dsv4CacheLayoutOptions& options);

}  // namespace xllm
