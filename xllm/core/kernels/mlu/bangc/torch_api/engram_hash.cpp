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

#include "kernels/mlu/bangc/include/engram_hash.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <limits>

#include "framework/core/MLUStream.h"
#include "framework/core/device.h"
#include "kernels/mlu/bangc/src/engram_hash/launcher.h"

namespace xllm::kernel::mlu::bangc {

torch::Tensor engram_hash(const torch::Tensor& tokens,
                          const torch::Tensor& starts,
                          const torch::Tensor& history,
                          const torch::Tensor& token_map,
                          const torch::Tensor& multipliers,
                          const torch::Tensor& primes,
                          const torch::Tensor& offsets,
                          int64_t pad_token) {
  CHECK(tokens.device().is_privateuseone());
  CHECK(tokens.scalar_type() == torch::kInt32 ||
        tokens.scalar_type() == torch::kInt64);
  CHECK_EQ(tokens.dim(), 1);
  CHECK(tokens.is_contiguous());
  for (const auto& tensor :
       {starts, history, token_map, multipliers, primes, offsets}) {
    CHECK(tensor.device() == tokens.device());
    CHECK(tensor.is_contiguous());
  }
  CHECK_EQ(starts.scalar_type(), torch::kInt32);
  for (const auto& tensor :
       {history, token_map, multipliers, primes, offsets}) {
    CHECK_EQ(tensor.scalar_type(), torch::kInt64);
  }
  CHECK_EQ(starts.dim(), 1);
  CHECK_EQ(history.dim(), 2);
  CHECK_EQ(token_map.dim(), 1);
  CHECK_EQ(multipliers.dim(), 2);
  CHECK_EQ(primes.dim(), 2);
  CHECK_EQ(offsets.sizes(), primes.sizes());
  CHECK_GE(starts.numel(), 2);
  CHECK_EQ(history.size(0), starts.numel() - 1);
  CHECK_EQ(history.size(1), multipliers.size(1) - 1);
  CHECK_GT(history.size(1), 0);
  CHECK_GT(multipliers.size(0), 0);
  CHECK_EQ(primes.size(0), multipliers.size(0));
  CHECK_GT(primes.size(1), 0);
  CHECK_EQ(primes.size(1) % history.size(1), 0);
  CHECK_GE(pad_token, 0);
  CHECK_LT(pad_token, token_map.numel());
  for (const int64_t size : {tokens.numel(),
                             starts.numel(),
                             multipliers.size(0),
                             multipliers.size(1),
                             primes.size(1)}) {
    CHECK_LE(size, std::numeric_limits<int32_t>::max());
  }
  const torch::DeviceGuard guard(tokens.device());
  auto output =
      torch::empty({multipliers.size(0), tokens.numel(), primes.size(1)},
                   tokens.options().dtype(torch::kInt64));
  if (tokens.numel() == 0) {
    return output;
  }
  const auto* properties =
      torch_mlu::getDeviceProperties(tokens.device().index());
  const int32_t cores =
      properties->cluster_count * properties->core_num_per_cluster;
  CHECK_GT(cores, 0);
  engram_hash_launch(torch_mlu::getCurMLUStream(),
                     tokens.data_ptr(),
                     tokens.scalar_type() == torch::kInt32,
                     starts.data_ptr<int32_t>(),
                     history.data_ptr<int64_t>(),
                     token_map.data_ptr<int64_t>(),
                     multipliers.data_ptr<int64_t>(),
                     primes.data_ptr<int64_t>(),
                     offsets.data_ptr<int64_t>(),
                     output.data_ptr<int64_t>(),
                     tokens.numel(),
                     static_cast<int32_t>(history.size(0)),
                     static_cast<int32_t>(multipliers.size(0)),
                     static_cast<int32_t>(multipliers.size(1)),
                     static_cast<int32_t>(primes.size(1)),
                     token_map.numel(),
                     pad_token,
                     cores);
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
  return output;
}

}  // namespace xllm::kernel::mlu::bangc
