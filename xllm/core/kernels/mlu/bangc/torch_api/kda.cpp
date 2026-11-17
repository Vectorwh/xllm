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

#include "kernels/mlu/bangc/include/kda.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>

#include "kernels/mlu/bangc/src/kda/launch.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu::bangc {
namespace {
int32_t to_kernel_dtype(torch::ScalarType dtype) {
  if (dtype == torch::kFloat32) {
    return kKdaFloat32;
  }
  return kKdaBFloat16;
}

}  // namespace

void kda_state_writeback(const torch::Tensor& source,
                         torch::Tensor& cache,
                         const torch::Tensor& indices) {
  CHECK(source.defined());
  CHECK(cache.defined());
  CHECK(source.device().is_privateuseone());
  CHECK(source.device() == cache.device());
  CHECK(source.dim() >= 2);
  CHECK(cache.dim() == source.dim());
  CHECK(source.sizes().slice(1) == cache.sizes().slice(1));
  CHECK(indices.dim() == 1);
  CHECK(bangc::detail::check_tensor(indices, torch::kInt32, source.device()));
  CHECK(source.size(0) == indices.numel());
  CHECK(source.scalar_type() == torch::kFloat32 ||
        source.scalar_type() == torch::kBFloat16);
  CHECK(cache.scalar_type() == torch::kFloat32 ||
        cache.scalar_type() == torch::kBFloat16);
  if (indices.numel() == 0) {
    return;
  }

  torch::DeviceGuard guard(source.device());
  const int64_t width = source.numel() / source.size(0);
  if (width == 0) {
    return;
  }
  const torch::Tensor rows = source.view({source.size(0), width});
  const torch::Tensor cache_rows = cache.view({cache.size(0), width});
  CHECK_EQ(rows.stride(1), 1);
  CHECK_EQ(cache_rows.stride(1), 1);
  // Rows must not overlap: the kernel addresses each row as
  // base + row * stride(0), so a shorter row stride would make adjacent source
  // or slot rows alias and silently share writes.
  CHECK_GE(rows.stride(0), width) << "kda_state_writeback source rows overlap";
  CHECK_GE(cache_rows.stride(0), width)
      << "kda_state_writeback cache rows overlap";
  // Slot 0 is the padding sentinel; the kernel also drops slots outside
  // [1, cache.size(0)) so a stale or foreign id cannot write past the cache.
  CHECK_EQ(cnrtSuccess,
           launch_kda_state_writeback(rows.const_data_ptr(),
                                      cache_rows.mutable_data_ptr(),
                                      indices.const_data_ptr<int32_t>(),
                                      indices.numel(),
                                      width,
                                      rows.stride(0),
                                      cache_rows.stride(0),
                                      cache.size(0),
                                      to_kernel_dtype(source.scalar_type()),
                                      to_kernel_dtype(cache.scalar_type()),
                                      torch_mlu::getCurMLUStream()));
}

}  // namespace xllm::kernel::mlu::bangc
