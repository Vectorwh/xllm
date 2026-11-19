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

#include "kernels/mlu/bangc/include/engram_gate.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <cmath>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/engram_gate/launcher.h"

namespace xllm::kernel::mlu::bangc {

torch::Tensor engram_gate(const torch::Tensor& hidden,
                          const torch::Tensor& kv,
                          const torch::Tensor& q_weight,
                          const torch::Tensor& k_weight,
                          double eps) {
  CHECK(hidden.defined());
  CHECK(hidden.device().is_privateuseone());
  CHECK_EQ(hidden.dim(), 3);
  CHECK_GT(hidden.size(1), 0);
  CHECK_GT(hidden.size(2), 0);
  CHECK_LE(hidden.size(1), std::numeric_limits<int32_t>::max() - 1);
  CHECK_LE(hidden.size(2), std::numeric_limits<int32_t>::max());
  CHECK(std::isfinite(eps) && eps > 0.0);
  CHECK(std::isfinite(static_cast<float>(eps)) &&
        static_cast<float>(eps) > 0.0f);
  for (const auto& tensor : {hidden, kv, q_weight, k_weight}) {
    CHECK(tensor.defined());
    CHECK(tensor.device() == hidden.device());
    CHECK_EQ(tensor.scalar_type(), torch::kBFloat16);
    CHECK(tensor.is_contiguous());
  }
  const int64_t tokens = hidden.size(0);
  const int64_t heads = hidden.size(1);
  const int64_t width = hidden.size(2);
  CHECK_EQ(kv.dim(), 2);
  CHECK_EQ(kv.size(0), tokens);
  CHECK_EQ(kv.size(1), (heads + 1) * width);
  CHECK(q_weight.sizes() == torch::IntArrayRef({heads, width}));
  CHECK(k_weight.sizes() == q_weight.sizes());
  const torch::DeviceGuard guard(hidden.device());
  auto output = torch::empty_like(hidden);
  if (tokens == 0) {
    return output;
  }
  CHECK_EQ(engram_gate_launch(torch_mlu::getCurMLUStream(),
                              hidden.data_ptr(),
                              kv.data_ptr(),
                              q_weight.data_ptr(),
                              k_weight.data_ptr(),
                              output.data_ptr(),
                              tokens,
                              static_cast<int32_t>(heads),
                              static_cast<int32_t>(width),
                              static_cast<float>(eps)),
           0);
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
  return output;
}

}  // namespace xllm::kernel::mlu::bangc
