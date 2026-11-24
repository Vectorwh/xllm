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

#include <optional>
#include <tuple>

namespace xllm::kernel::mlu::bangc {
// Batches up to this size use the workspace paths of the standard packing.
inline constexpr int64_t kMoeRouterWorkspaceRows = 950;

// Prepare BF16 [experts, 5120] once at weight load, outside graph capture.
torch::Tensor pack_moe_router_weight(const torch::Tensor& weight);
// Returns an undefined tensor for layouts outside the model fast path.
torch::Tensor try_pack_moe_router_weight(const torch::Tensor& weight,
                                         int64_t experts,
                                         int64_t topk);
// Second packing [8, 16, experts/16, 640] for gates whose whole weight fits
// cluster SRAM (128 experts). Serves batches above kMoeRouterWorkspaceRows.
torch::Tensor pack_moe_router_resident_weight(const torch::Tensor& weight);
// Returns an undefined tensor for layouts outside the resident fast path.
torch::Tensor try_pack_moe_router_resident_weight(const torch::Tensor& weight,
                                                  int64_t experts,
                                                  int64_t topk);

// BF16 [tokens, 5120], packed BF16 weight from either packing, FP32 bias.
// input_ids select bias_vl for image tokens. Omitted IDs use the text bias.
// Returns FP32 weights and INT32 expert IDs [tokens, topk]. Runs on the
// current stream; workspace and outputs are graph-capture compatible.
std::tuple<torch::Tensor, torch::Tensor> fused_moe_router(
    const torch::Tensor& hidden_states,
    const torch::Tensor& packed_weight,
    const torch::Tensor& bias,
    int64_t topk,
    double route_scale,
    const std::optional<torch::Tensor>& input_ids = std::nullopt,
    const std::optional<torch::Tensor>& bias_vl = std::nullopt,
    int64_t image_token_id = -1);
}  // namespace xllm::kernel::mlu::bangc
