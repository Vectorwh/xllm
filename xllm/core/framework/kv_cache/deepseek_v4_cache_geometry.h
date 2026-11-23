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

namespace xllm {

constexpr int32_t kDsv4C4CompressRatio = 4;
constexpr int32_t kDsv4C128CompressRatio = 128;
// Fixed token spans. SWA is 128 on every DSV4 and DSV4.1 device. KV is 2048
// only for DSV4 on NPU; DSV4 on MLU and DSV4.1 on NPU or MLU use 128.
// Physical dim is the KV span divided by the compress ratio.
constexpr int64_t kDsv4NpuBlockSize = 2048;
constexpr int64_t kDsv4FixedBlockSize = 128;
constexpr int64_t kDsv4SwaBlockSize = 128;

// Tokens in one logical KV block. npu_dsv4 is true only for DSV4 on NPU.
[[nodiscard]] inline constexpr int64_t dsv4_kv_block_size(bool npu_dsv4) {
  return npu_dsv4 ? kDsv4NpuBlockSize : kDsv4FixedBlockSize;
}

// Tokens in one SWA block. Every DSV4 and DSV4.1 device uses 128.
[[nodiscard]] inline constexpr int64_t dsv4_swa_block_size() {
  return kDsv4SwaBlockSize;
}

// Physical tensor dim of one logical block: block_size / compress_ratio.
[[nodiscard]] inline constexpr int64_t dsv4_physical_dim(
    int64_t block_size,
    int32_t compress_ratio) {
  return compress_ratio > 0 ? block_size / compress_ratio : 0;
}

}  // namespace xllm
