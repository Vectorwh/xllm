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

#include "kernels/mlu/bangc/include/engram_dequantize.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/engram_dequantize/launcher.h"

namespace xllm::kernel::mlu::bangc {

torch::Tensor engram_dequantize(const torch::Tensor& bytes,
                                const torch::Tensor& scales,
                                const torch::Tensor& hashes,
                                const torch::Tensor& fp8_values,
                                int64_t table_start,
                                int64_t table_end,
                                bool all_owned) {
  CHECK(bytes.defined() && scales.defined() && hashes.defined() &&
        fp8_values.defined());
  CHECK(bytes.device().is_privateuseone());
  for (const auto& tensor : {bytes, scales, hashes, fp8_values}) {
    CHECK(tensor.device() == bytes.device());
    CHECK(tensor.is_contiguous());
  }
  CHECK_EQ(bytes.scalar_type(), torch::kUInt8);
  CHECK_EQ(scales.scalar_type(), torch::kUInt8);
  CHECK_EQ(hashes.scalar_type(), torch::kInt64);
  CHECK_EQ(fp8_values.scalar_type(), torch::kFloat32);
  CHECK_EQ(fp8_values.numel(), 256);
  CHECK_EQ(bytes.dim(), 2);
  CHECK_EQ(scales.dim(), 2);
  CHECK_EQ(hashes.dim(), 2);
  CHECK_EQ(bytes.size(0), hashes.numel());
  CHECK_EQ(scales.size(0), bytes.size(0));
  CHECK_GT(bytes.size(1), 0);
  CHECK_GT(scales.size(1), 0);
  CHECK_EQ(bytes.size(1) % scales.size(1), 0);
  CHECK_LE(bytes.size(1), std::numeric_limits<int32_t>::max());
  CHECK_GE(table_start, 0);
  CHECK_GE(table_end, table_start);
  const torch::DeviceGuard guard(bytes.device());
  auto output = torch::empty({hashes.size(0), hashes.size(1), bytes.size(1)},
                             bytes.options().dtype(torch::kBFloat16));
  engram_dequantize_launch(torch_mlu::getCurMLUStream(),
                           bytes.data_ptr(),
                           scales.data_ptr(),
                           hashes.data_ptr(),
                           fp8_values.data_ptr(),
                           output.data_ptr(),
                           bytes.size(0),
                           static_cast<int32_t>(bytes.size(1)),
                           static_cast<int32_t>(scales.size(1)),
                           table_start,
                           table_end,
                           all_owned);
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
  return output;
}

torch::Tensor engram_dequantize_packed(const torch::Tensor& packed,
                                       const torch::Tensor& hashes,
                                       const torch::Tensor& fp8_values,
                                       int64_t dim,
                                       int64_t scale_dim,
                                       int64_t table_start,
                                       int64_t table_end,
                                       int64_t head_shard_count,
                                       bool all_owned,
                                       const torch::Tensor& row_indices) {
  CHECK(packed.defined() && hashes.defined() && fp8_values.defined());
  CHECK(packed.device().is_privateuseone());
  for (const auto& tensor : {packed, hashes, fp8_values}) {
    CHECK(tensor.device() == packed.device());
  }
  CHECK_EQ(packed.scalar_type(), torch::kUInt8);
  CHECK_EQ(hashes.scalar_type(), torch::kInt64);
  CHECK_EQ(fp8_values.scalar_type(), torch::kFloat32);
  CHECK_EQ(packed.dim(), 2);
  CHECK_EQ(hashes.dim(), 2);
  CHECK(hashes.is_contiguous() && fp8_values.is_contiguous());
  CHECK_EQ(fp8_values.numel(), 256);
  CHECK_GT(dim, 0);
  CHECK_GT(scale_dim, 0);
  CHECK_EQ(dim % scale_dim, 0);
  CHECK_LE(dim, std::numeric_limits<int32_t>::max());
  CHECK_EQ(packed.size(1), dim + scale_dim);
  CHECK_EQ(packed.stride(1), 1);
  CHECK_GE(packed.stride(0), packed.size(1));
  CHECK_GT(head_shard_count, 0);
  CHECK_GE(table_start, 0);
  CHECK_GE(table_end, table_start);
  if (row_indices.defined()) {
    CHECK_EQ(row_indices.dim(), 1);
    CHECK_EQ(row_indices.scalar_type(), torch::kInt64);
    CHECK(row_indices.is_contiguous());
    CHECK(row_indices.device() == packed.device());
  }
  const int64_t tokens = hashes.size(0);
  const int64_t output_tokens =
      row_indices.defined() ? row_indices.numel() : tokens;
  const int64_t columns = hashes.size(1);
  const int64_t shard_columns =
      columns == 0 ? 0 : (columns - 1) / head_shard_count + 1;
  CHECK_LE(head_shard_count,
           std::numeric_limits<int64_t>::max() /
               std::max<int64_t>(shard_columns, 1));
  const int64_t physical_columns = head_shard_count * shard_columns;
  CHECK_LE(tokens,
           std::numeric_limits<int64_t>::max() /
               std::max<int64_t>(physical_columns, 1));
  CHECK_EQ(packed.size(0), tokens * physical_columns);
  const torch::DeviceGuard guard(packed.device());
  CHECK_LE(output_tokens,
           std::numeric_limits<int64_t>::max() / std::max<int64_t>(columns, 1));
  auto output = torch::empty({output_tokens, columns, dim},
                             packed.options().dtype(torch::kBFloat16));
  engram_dequantize_packed_launch(
      torch_mlu::getCurMLUStream(),
      packed.data_ptr(),
      hashes.data_ptr(),
      fp8_values.data_ptr(),
      output.data_ptr(),
      output_tokens * columns,
      static_cast<int32_t>(dim),
      static_cast<int32_t>(scale_dim),
      packed.stride(0),
      tokens,
      columns,
      head_shard_count,
      table_start,
      table_end,
      all_owned,
      row_indices.defined() ? row_indices.data_ptr() : nullptr);
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
  return output;
}

}  // namespace xllm::kernel::mlu::bangc
