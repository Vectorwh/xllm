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

#include <torch/library.h>

#include <algorithm>
#include <optional>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor key_pool(const torch::Tensor& hidden_states,
                       const torch::Tensor& key_weight,
                       const torch::Tensor& gate_weight,
                       const torch::Tensor& ape,
                       torch::Tensor& state_cache,
                       const torch::Tensor& block_table,
                       const torch::Tensor& start_positions,
                       const torch::Tensor& norm_weight,
                       const torch::Tensor& norm_bias,
                       const torch::Tensor& cu_seqlens,
                       int64_t pool_size,
                       double norm_epsilon) {
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_GT(pool_size, 0);
  CHECK_EQ(state_cache.scalar_type(), torch::kFloat32);
  CHECK_EQ(norm_weight.scalar_type(), torch::kFloat32);
  CHECK_EQ(norm_bias.scalar_type(), torch::kFloat32);
  const int64_t output_rows =
      std::min(hidden_states.size(0),
               hidden_states.size(0) / pool_size + block_table.size(0));
  torch::Tensor output =
      torch::empty({output_rows, key_weight.size(0)}, hidden_states.options());
  const std::optional<torch::Tensor> no_tensor;
  const int64_t rotary_mode = 0;
  const int64_t state_stride = state_cache.stride(0);
  EXEC_NPU_CMD(aclnnKeyPool,
               hidden_states,
               key_weight,
               gate_weight,
               ape,
               state_cache,
               block_table,
               start_positions,
               norm_weight,
               norm_bias,
               no_tensor,
               no_tensor,
               cu_seqlens,
               no_tensor,
               pool_size,
               norm_epsilon,
               rotary_mode,
               state_stride,
               output);
  return output;
}

}  // namespace xllm::kernel::npu
