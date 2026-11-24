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
namespace xllm::kernel::mlu::bangc {
inline constexpr int32_t kFfnHidden = 5120;
inline constexpr int32_t kFfnIntermediate = 2304;
inline constexpr int32_t kFfnMaxExperts = 512;
// At most ceil(rows / 96) + experts units, dealt to eight cluster tables.
inline constexpr int32_t kMaxLaunchRows = (8 * 512 - kFfnMaxExperts) * 96;
void fused_moe_ffn_launch(void* queue,
                          void* output,
                          void* workspace,
                          const int8_t* input,
                          const float* input_scale,
                          const int8_t* weight13,
                          const int8_t* weight2,
                          const void* scale13,
                          const void* scale2,
                          const float* smooth,
                          const int32_t* counts,
                          int32_t rows,
                          int32_t experts,
                          float limit,
                          bool scale16,
                          int64_t skip_rows);
}  // namespace xllm::kernel::mlu::bangc
