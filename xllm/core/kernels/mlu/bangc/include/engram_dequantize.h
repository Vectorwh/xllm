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

// Decode contiguous FP8 payload and E8M0 scales, masking unowned hash rows.
// bytes [T*C,D] uint8, scales [T*C,S] uint8, hashes [T,C] int64,
// all_owned requires the caller to prove every hash lies in the owned range.
// fp8_values must be the canonical E4M3FN byte-to-float table, [256] float32.
// Output -> [T,C,D] bfloat16. D must be divisible by S.
torch::Tensor engram_dequantize(const torch::Tensor& bytes,
                                const torch::Tensor& scales,
                                const torch::Tensor& hashes,
                                const torch::Tensor& fp8_values,
                                int64_t table_start,
                                int64_t table_end,
                                bool all_owned = false);

// Decode directly from [physical_rows,D+S] packed payload, without splitting
// device tensors. packed.stride(1) must be 1; row padding is supported.
// For head_shard_count=1 rows are [T,C,D+S]. For R>1 they are the raw
// rank-major collective layout [R,T,ceil(C/R),D+S]; padded heads are ignored.
// Output is [T,C,D] BF16. The LUT and ownership contracts match the API above.
// Optional row_indices maps each output token to a source token. It must be
// contiguous 1D int64 on the same device; callers guarantee values in [0,T).
// Repeated indices are supported, and output is [row_indices.numel(),C,D].
// The packed source layout always uses the original T, including head shards.
torch::Tensor engram_dequantize_packed(const torch::Tensor& packed,
                                       const torch::Tensor& hashes,
                                       const torch::Tensor& fp8_values,
                                       int64_t dim,
                                       int64_t scale_dim,
                                       int64_t table_start,
                                       int64_t table_end,
                                       int64_t head_shard_count = 1,
                                       bool all_owned = false,
                                       const torch::Tensor& row_indices = {});

}  // namespace xllm::kernel::mlu::bangc
