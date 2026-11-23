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

#include <torch/types.h>

#include <cstdint>
#include <vector>

#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache_tensor_role.h"

namespace xllm {

// One owned execution tensor bound to a scheduling resource. block_shape
// excludes the pool capacity dimension; its physical rows need not equal the
// resource's original-token span. Aliases are views, not additional owners.
struct CacheTensorAllocationSpec {
  BlockType block_type = BlockType::KV;
  KVCacheTensorRole role = KVCacheTensorRole::INVALID;
  std::vector<int64_t> block_shape;
  torch::ScalarType dtype = torch::kBFloat16;
};

}  // namespace xllm
