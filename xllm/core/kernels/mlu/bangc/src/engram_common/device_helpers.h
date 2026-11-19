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

#include <bang.h>
#include <stdint.h>

namespace xllm::kernel::mlu::bangc::engram_detail {
namespace {

// Decode a contiguous tile using NRAM-resident FP8 codes, scales and LUT.
// scale_codes begins at the group containing start; scratch/output buffers
// must hold count rounded up to 32 elements. Keep scaling in FP32 until the
// caller rounds to BF16, including E8M0's subnormal and infinite factors.
__mlu_func__ inline void decode_fp8_tile(float* values,
                                         uint32_t* indices,
                                         const uint8_t* codes,
                                         const uint8_t* scale_codes,
                                         const int32_t* lut,
                                         int64_t start,
                                         int32_t count,
                                         int32_t group_size) {
  const int32_t aligned = (count + 31) / 32 * 32;
  __bang_uint82int32(reinterpret_cast<int32_t*>(indices), codes, aligned, 0, 0);
  __bang_lut(reinterpret_cast<int32_t*>(values), indices, lut, aligned, 256);
  const int32_t first_group = start / group_size;
  int32_t column = 0;
  while (column < count) {
    const int64_t global_column = start + column;
    const int32_t group = global_column / group_size;
    const int32_t remaining = group_size - global_column % group_size;
    const int32_t length =
        remaining < count - column ? remaining : count - column;
    const uint32_t code = scale_codes[group - first_group];
    union ScaleBits {
      uint32_t bits;
      float value;
    } factor;
    // exp2(code - 127), including subnormal 2^-127 and exp2(128).
    factor.bits = code == 0     ? 0x00400000U
                  : code == 255 ? 0x7f800000U
                                : code << 23;
    __bang_mul_scalar(values + column, values + column, factor.value, length);
    column += length;
  }
}

template <bool Mapped>
__mlu_func__ inline void load_row_hashes(int64_t* row_hashes,
                                         const int64_t* hashes,
                                         int64_t first,
                                         int32_t count,
                                         int64_t columns,
                                         const int64_t* row_indices) {
  if constexpr (Mapped) {
    // Preserve the contiguous head rows of each mapped token. Hash prefetch
    // remains asynchronous and keeps the original two-tile lookahead.
    int32_t copied = 0;
    while (copied < count) {
      const int64_t output_row = first + copied;
      const int64_t column = output_row % columns;
      const int32_t length = static_cast<int32_t>(
          columns - column < count - copied ? columns - column
                                            : count - copied);
      const int64_t source =
          row_indices[output_row / columns] * columns + column;
      __memcpy_async(row_hashes + copied,
                     hashes + source,
                     length * sizeof(int64_t),
                     GDRAM2NRAM);
      copied += length;
    }
  } else {
    __memcpy_async(
        row_hashes, hashes + first, count * sizeof(int64_t), GDRAM2NRAM);
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu::bangc::engram_detail
