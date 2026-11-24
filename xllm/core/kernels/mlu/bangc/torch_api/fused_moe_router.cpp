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

#include "kernels/mlu/bangc/include/fused_moe_router.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <cmath>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/fused_moe_router/launcher.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu::bangc {
namespace {
// [experts, 5120] -> [K block][LT16][experts / 16][block_size], the WRAM order
// of one K block of all experts.
torch::Tensor pack_weight(const torch::Tensor& weight, int64_t block_size) {
  CHECK(weight.defined());
  CHECK(weight.device().is_privateuseone());
  CHECK_EQ(weight.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weight.dim(), 2);
  CHECK_EQ(weight.size(1), kRouterHidden);
  CHECK_GT(weight.size(0), 0);
  CHECK_LE(weight.size(0), 384);
  CHECK_EQ(weight.size(0) % 64, 0);
  CHECK(weight.is_contiguous());
  const torch::DeviceGuard guard(weight.device());
  const int64_t experts = weight.size(0);
  const int64_t blocks = kRouterHidden / block_size;
  return weight.view({experts / 64, 16, 4, blocks, block_size})
      .permute({3, 1, 0, 2, 4})
      .contiguous()
      .view({blocks, 16, experts / 16, block_size});
}
}  // namespace

torch::Tensor pack_moe_router_weight(const torch::Tensor& weight) {
  return pack_weight(weight, kRouterBlock);
}

torch::Tensor try_pack_moe_router_weight(const torch::Tensor& weight,
                                         int64_t experts,
                                         int64_t topk) {
  const bool supported =
      (experts == 384 && topk == 6) || (experts == 128 && topk == 3);
  if (!supported || !detail::check_tensor(weight, torch::kBFloat16) ||
      weight.sizes() != torch::IntArrayRef({experts, kRouterHidden}) ||
      !detail::supports_mtp592_union8(weight.device())) {
    return {};
  }
  return pack_moe_router_weight(weight);
}

torch::Tensor pack_moe_router_resident_weight(const torch::Tensor& weight) {
  CHECK(weight.defined());
  if (weight.size(0) != kRouterResidentExperts) {
    return {};
  }
  return pack_weight(weight, kRouterResidentBlock);
}

torch::Tensor try_pack_moe_router_resident_weight(const torch::Tensor& weight,
                                                  int64_t experts,
                                                  int64_t topk) {
  if (experts != kRouterResidentExperts || topk != 3 ||
      !detail::check_tensor(weight, torch::kBFloat16) ||
      weight.sizes() != torch::IntArrayRef({experts, kRouterHidden}) ||
      !detail::supports_mtp592_union8(weight.device())) {
    return {};
  }
  return pack_moe_router_resident_weight(weight);
}

std::tuple<torch::Tensor, torch::Tensor> fused_moe_router(
    const torch::Tensor& hidden_states,
    const torch::Tensor& packed_weight,
    const torch::Tensor& bias,
    int64_t topk,
    double route_scale,
    const std::optional<torch::Tensor>& input_ids,
    const std::optional<torch::Tensor>& bias_vl,
    int64_t image_token_id) {
  CHECK(hidden_states.defined() && packed_weight.defined() && bias.defined());
  CHECK(hidden_states.device().is_privateuseone());
  CHECK_EQ(hidden_states.scalar_type(), torch::kBFloat16);
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(hidden_states.size(1), kRouterHidden);
  // Device partition arithmetic uses signed 32-bit row counts with headroom.
  CHECK_LE(hidden_states.size(0), std::numeric_limits<int32_t>::max() / 32);
  CHECK_EQ(packed_weight.scalar_type(), torch::kBFloat16);
  CHECK_EQ(packed_weight.dim(), 4);
  const int64_t block_size = packed_weight.size(3);
  const bool resident = block_size == kRouterResidentBlock;
  CHECK(resident || block_size == kRouterBlock);
  CHECK_EQ(packed_weight.size(0), kRouterHidden / block_size);
  CHECK_EQ(packed_weight.size(1), 16);
  const int64_t experts = packed_weight.size(2) * 16;
  CHECK(!resident || experts == kRouterResidentExperts);
  CHECK_GT(experts, 0);
  CHECK_LE(experts, 384);
  CHECK_EQ(experts % 64, 0);
  CHECK_GE(topk, 2);
  CHECK_LE(topk, 8);
  CHECK_EQ(bias.scalar_type(), torch::kFloat32);
  CHECK_EQ(bias.dim(), 1);
  CHECK_EQ(bias.size(0), experts);
  CHECK(hidden_states.is_contiguous() && packed_weight.is_contiguous() &&
        bias.is_contiguous());
  CHECK(packed_weight.device() == hidden_states.device());
  CHECK(bias.device() == hidden_states.device());
  CHECK(std::isfinite(route_scale));
  CHECK_LE(std::abs(route_scale), std::numeric_limits<float>::max());
  const torch::DeviceGuard guard(hidden_states.device());
  const int64_t rows = hidden_states.size(0);
  RouterBias router_bias{bias.data_ptr<float>(),
                         nullptr,
                         nullptr,
                         image_token_id,
                         /*int64_ids=*/false};
  torch::Tensor token_ids;
  if (input_ids.has_value()) {
    CHECK(bias_vl.has_value() && image_token_id >= 0);
    CHECK(bias_vl->device() == bias.device() && bias_vl->is_contiguous() &&
          bias_vl->sizes() == bias.sizes() &&
          bias_vl->scalar_type() == torch::kFloat32);
    CHECK(input_ids->device() == hidden_states.device() &&
          input_ids->numel() == rows);
    CHECK(input_ids->scalar_type() == torch::kInt32 ||
          input_ids->scalar_type() == torch::kInt64);
    token_ids = input_ids->contiguous();
    router_bias.visual = bias_vl->data_ptr<float>();
    router_bias.input_ids = token_ids.data_ptr();
    router_bias.int64_ids = token_ids.scalar_type() == torch::kInt64;
  }
  torch::Tensor weights = torch::empty(
      {rows, topk}, hidden_states.options().dtype(torch::kFloat32));
  torch::Tensor ids =
      torch::empty({rows, topk}, hidden_states.options().dtype(torch::kInt32));
  if (rows == 0) {
    return {weights, ids};
  }
  if (resident) {
    fused_moe_router_resident_launch(torch_mlu::getCurMLUStream(),
                                     hidden_states.data_ptr(),
                                     packed_weight.data_ptr(),
                                     router_bias,
                                     weights.data_ptr<float>(),
                                     ids.data_ptr<int32_t>(),
                                     static_cast<int32_t>(rows),
                                     static_cast<int32_t>(topk),
                                     static_cast<float>(route_scale));
    CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
    return {weights, ids};
  }
  const int32_t partitions = router_k_partitions(static_cast<int32_t>(rows));
  torch::Tensor workspace;
  if (partitions > 1) {
    workspace =
        torch::empty({partitions * rows * experts + 32}, weights.options());
  }
  fused_moe_router_launch(
      torch_mlu::getCurMLUStream(),
      hidden_states.data_ptr(),
      packed_weight.data_ptr(),
      router_bias,
      weights.data_ptr<float>(),
      ids.data_ptr<int32_t>(),
      static_cast<int32_t>(rows),
      static_cast<int32_t>(experts),
      static_cast<int32_t>(topk),
      static_cast<float>(route_scale),
      partitions,
      workspace.defined() ? workspace.data_ptr<float>() : nullptr);
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
  return {weights, ids};
}
}  // namespace xllm::kernel::mlu::bangc
