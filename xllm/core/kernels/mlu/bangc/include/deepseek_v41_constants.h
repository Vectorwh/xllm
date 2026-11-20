/* Copyright 2026 The xLLM Authors.

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

namespace xllm::kernel::mlu::bangc {

// MLU V4.1 SWA, compressed KV, and index caches share this page width.
// Candidate-selection blocks have an independent width from ModelArgs.
inline constexpr int32_t kDeepseekV41MluKVBlockSize = 128;

inline constexpr bool supports_dsa_qb_width(int64_t width) {
  return width == 4096 || width == 8192 || width == 16384 || width == 32768;
}

}  // namespace xllm::kernel::mlu::bangc
