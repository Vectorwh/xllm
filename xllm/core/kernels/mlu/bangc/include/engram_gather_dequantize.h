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

namespace xllm::kernel::mlu::bangc {

// Gather owned FP8 table rows and apply exp2(scale_code - 127) in FP32.
// bytes [R,D], scales [R,S]: uint8; hashes [T,C]: int64;
// fp8_values [256]: canonical E4M3FN float32 LUT (code0 is positive zero). R =
// table_end - table_start, D % S = 0. The BF16 result is [T,C,D]. Foreign
// hashes produce positive zero, including when an owned table value or scale
// would decode to NaN.
// Optional row_indices maps each output token to a source token. It must be
// contiguous 1D int64 on the same device; callers guarantee values in [0,T).
// Repeated indices are supported, and output is [row_indices.numel(),C,D].
torch::Tensor engram_gather_dequantize(const torch::Tensor& bytes,
                                       const torch::Tensor& scales,
                                       const torch::Tensor& hashes,
                                       const torch::Tensor& fp8_values,
                                       int64_t table_start,
                                       int64_t table_end,
                                       const torch::Tensor& row_indices = {});

}  // namespace xllm::kernel::mlu::bangc
