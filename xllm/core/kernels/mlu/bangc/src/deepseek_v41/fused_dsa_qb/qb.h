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

#include "cnrt.h"

namespace xllm::kernel::mlu::bangc::dsa_qb {

cnrtRet_t launch_quantize_smooth(cnrtQueue_t queue,
                                 const void* qr,
                                 const void* smooth,
                                 void* q_int8,
                                 int32_t m,
                                 const void* weight_scale,
                                 void* ts_bf16,
                                 void* ws_bf16,
                                 int32_t n);

cnrtRet_t launch_small_int8(cnrtQueue_t queue,
                            const void* qr_bf16,
                            const void* smooth,
                            const void* weight,
                            const void* weight_scale,
                            const void* positions,
                            const void* rope_cache,
                            int32_t rope_rows,
                            void* out_bf16,
                            int32_t m,
                            int32_t n);

cnrtRet_t launch_large_int8(cnrtQueue_t queue,
                            const void* q_int8,
                            const void* weight,
                            const void* token_scale,
                            const void* weight_scale,
                            const void* positions,
                            const void* rope_cache,
                            int32_t rope_rows,
                            void* out_bf16,
                            int32_t m,
                            int32_t n);

cnrtRet_t launch_bf16(cnrtQueue_t queue,
                      const void* qr_bf16,
                      const void* weight_bf16,
                      const void* positions,
                      const void* rope_cache,
                      int32_t rope_rows,
                      void* out_bf16,
                      int32_t m,
                      int32_t n);

}  // namespace xllm::kernel::mlu::bangc::dsa_qb
