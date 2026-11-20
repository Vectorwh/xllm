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

#include <mlu.h>

#include "kernels/mlu/bangc/src/deepseek_v41/fused_dsa_output_projection/output_projection.h"

namespace xllm::kernel::mlu::bangc::output_projection {
__mlu_global__ void rope_quant_wo_a_kernel(const bfloat16_t* x,
                                           const int32_t* pos,
                                           const float* rope,
                                           int32_t last_pos,
                                           const float* smooth_a,
                                           int8_t* q_a,
                                           float* scale_a,
                                           int32_t m,
                                           int32_t g,
                                           uint32_t g_shift);
__mlu_global__ void grouped_w8a8_gemm_kernel(const int8_t* q_a,
                                             const float* scale_a,
                                             const int8_t* weight,
                                             const float* scale_w,
                                             bfloat16_t* y_out,
                                             int32_t m,
                                             int32_t g);
__mlu_global__ void quant_wo_b_kernel(const bfloat16_t* y_in,
                                      const float* smooth_b,
                                      int8_t* out_q,
                                      float* out_scale,
                                      int32_t m,
                                      int32_t cols);
__mlu_global__ void fused_decode_tp8_kernel(const bfloat16_t* x,
                                            const int32_t* pos,
                                            const float* rope,
                                            int32_t last_pos,
                                            const int8_t* weight,
                                            const float* scale_w,
                                            const float* smooth_a,
                                            const float* smooth_b,
                                            int8_t* out_q,
                                            float* out_scale,
                                            int32_t m);
__mlu_global__ void fused_decode_rope_gemm_kernel(const bfloat16_t* x,
                                                  const int32_t* pos,
                                                  const float* rope,
                                                  int32_t last_pos,
                                                  const int8_t* weight,
                                                  const float* scale_w,
                                                  const float* smooth_a,
                                                  bfloat16_t* y_out,
                                                  int32_t m,
                                                  int32_t g,
                                                  uint32_t cpg_shift);
}  // namespace xllm::kernel::mlu::bangc::output_projection
