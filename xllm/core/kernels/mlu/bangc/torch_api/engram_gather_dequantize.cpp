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

#include "kernels/mlu/bangc/include/engram_gather_dequantize.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/engram_gather_dequantize/launcher.h"

namespace xllm::kernel::mlu::bangc {

torch::Tensor engram_gather_dequantize(const torch::Tensor& bytes,
                                       const torch::Tensor& scales,
                                       const torch::Tensor& hashes,
                                       const torch::Tensor& fp8_values,
                                       int64_t table_start,
                                       int64_t table_end,
                                       const torch::Tensor& row_indices) {
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
  CHECK_GE(table_start, 0);
  CHECK_GE(table_end, table_start);
  CHECK_EQ(bytes.size(0), table_end - table_start);
  CHECK_EQ(scales.size(0), bytes.size(0));
  CHECK_GT(bytes.size(1), 0);
  CHECK_GT(scales.size(1), 0);
  CHECK_EQ(bytes.size(1) % scales.size(1), 0);
  CHECK_LE(bytes.size(1), std::numeric_limits<int32_t>::max());
  if (row_indices.defined()) {
    CHECK_EQ(row_indices.dim(), 1);
    CHECK_EQ(row_indices.scalar_type(), torch::kInt64);
    CHECK(row_indices.is_contiguous());
    CHECK(row_indices.device() == bytes.device());
  }
  const int64_t output_tokens =
      row_indices.defined() ? row_indices.numel() : hashes.size(0);
  CHECK_LE(output_tokens,
           std::numeric_limits<int64_t>::max() /
               std::max<int64_t>(hashes.size(1), 1));
  const torch::DeviceGuard guard(bytes.device());
  auto output = torch::empty({output_tokens, hashes.size(1), bytes.size(1)},
                             bytes.options().dtype(torch::kBFloat16));
  engram_gather_dequantize_launch(
      torch_mlu::getCurMLUStream(),
      bytes.data_ptr(),
      scales.data_ptr(),
      hashes.data_ptr(),
      fp8_values.data_ptr(),
      output.data_ptr(),
      output_tokens * hashes.size(1),
      static_cast<int32_t>(bytes.size(1)),
      static_cast<int32_t>(scales.size(1)),
      table_start,
      table_end,
      hashes.size(1),
      row_indices.defined() ? row_indices.data_ptr() : nullptr);
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
  return output;
}

}  // namespace xllm::kernel::mlu::bangc
