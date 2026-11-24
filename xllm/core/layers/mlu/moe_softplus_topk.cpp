/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/mlu/moe_softplus_topk.h"

#include <glog/logging.h>

#include <tuple>

#include "kernels/mlu/bangc/include/fused_moe_router.h"
#include "kernels/mlu/mlu_ops_api.h"
namespace xllm {
namespace layer {

MOESoftPlusTopKImpl::MOESoftPlusTopKImpl(int64_t n_routed_experts,
                                         int64_t n_activated_experts,
                                         float route_scale,
                                         int64_t vocab_size,
                                         bool use_hash,
                                         const torch::TensorOptions& options,
                                         bool use_visual_bias,
                                         int64_t image_token_id)
    : n_routed_experts_(n_routed_experts),
      topk_(n_activated_experts),
      route_scale_(route_scale),
      vocab_size_(vocab_size),
      use_hash_(use_hash),
      use_visual_bias_(use_visual_bias),
      image_token_id_(image_token_id) {
  if (use_visual_bias_) {
    CHECK(!use_hash_) << "Visual router bias is incompatible with hash routing";
    CHECK_GE(image_token_id_, 0)
        << "Visual router bias requires a valid image token ID";
  }
  packed_gate_weight_ =
      register_buffer("packed_gate_weight", torch::empty({0}, options));
  packed_resident_gate_weight_ = register_buffer("packed_resident_gate_weight",
                                                 torch::empty({0}, options));
  if (use_hash_) {
    tid2eid_ = register_parameter(
        "tid2eid",
        torch::empty({vocab_size_, topk_}, options.dtype(torch::kInt32)),
        /*requires_grad=*/false);
    tid2eid_opt_ = tid2eid_;
  } else {
    bias_ = register_parameter(
        "bias",
        torch::empty({n_routed_experts_}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    bias_opt_ = bias_;
    if (use_visual_bias_) {
      bias_vl_ = register_parameter(
          "bias_vl",
          torch::empty({n_routed_experts_}, options.dtype(torch::kFloat32)),
          /*requires_grad=*/false);
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor> MOESoftPlusTopKImpl::forward(
    const torch::Tensor& scores,
    const std::optional<torch::Tensor>& input_ids) {
  std::optional<torch::Tensor> bias = bias_opt_;
  if (use_visual_bias_ && input_ids.has_value()) {
    CHECK(bias_vl_is_loaded_) << "Missing visual router bias_vl";
    CHECK_EQ(input_ids->numel(), scores.size(0));
    bias = torch::where(
        input_ids->reshape({-1, 1}).eq(image_token_id_), bias_vl_, bias_);
  }
  auto [weights, indices] = xllm::kernel::mlu::moe_softplus_topk(
      scores, topk_, input_ids, tid2eid_opt_, bias, route_scale_);
  return {weights, indices};
}

void MOESoftPlusTopKImpl::prepare_fused_router(
    const torch::Tensor& gate_weight) {
  packed_gate_weight_.set_data(
      torch::empty({0}, packed_gate_weight_.options()));
  packed_resident_gate_weight_.set_data(
      torch::empty({0}, packed_resident_gate_weight_.options()));
  if (!use_hash_) {
    torch::Tensor packed = kernel::mlu::bangc::try_pack_moe_router_weight(
        gate_weight, n_routed_experts_, topk_);
    if (packed.defined()) {
      packed_gate_weight_.set_data(packed);
    }
    torch::Tensor resident =
        kernel::mlu::bangc::try_pack_moe_router_resident_weight(
            gate_weight, n_routed_experts_, topk_);
    if (resident.defined()) {
      packed_resident_gate_weight_.set_data(resident);
    }
  }
}

std::optional<std::tuple<torch::Tensor, torch::Tensor>>
MOESoftPlusTopKImpl::forward_fused_router(
    const torch::Tensor& hidden_states,
    const std::optional<torch::Tensor>& input_ids) {
  if (packed_gate_weight_.numel() == 0) {
    return std::nullopt;
  }
  CHECK_GE(hidden_states.dim(), 2);
  CHECK_EQ(hidden_states.size(-1), 5120);
  const int64_t rows = hidden_states.numel() / 5120;
  const bool resident =
      packed_resident_gate_weight_.numel() > 0 &&
      rows > xllm::kernel::mlu::bangc::kMoeRouterWorkspaceRows;
  CHECK(bias_is_loaded_) << "Fused router requires the loaded correction bias";
  CHECK(!use_visual_bias_ || !input_ids.has_value() || bias_vl_is_loaded_)
      << "Missing visual router bias_vl";
  return xllm::kernel::mlu::bangc::fused_moe_router(
      hidden_states.reshape({-1, 5120}).contiguous(),
      resident ? packed_resident_gate_weight_ : packed_gate_weight_,
      bias_,
      topk_,
      route_scale_,
      use_visual_bias_ ? input_ids : std::nullopt,
      use_visual_bias_ ? std::make_optional(bias_vl_) : std::nullopt,
      image_token_id_);
}

void MOESoftPlusTopKImpl::load_state_dict(const StateDict& state_dict) {
  if (use_hash_) {
    LOAD_WEIGHT(tid2eid);
    tid2eid_opt_ = tid2eid_;
  } else {
    LOAD_WEIGHT(bias);
    bias_opt_ = bias_;
    if (use_visual_bias_) {
      LOAD_WEIGHT(bias_vl);
    }
  }
}

void MOESoftPlusTopKImpl::verify_loaded_weights() const {
  CHECK(use_hash_ || bias_is_loaded_) << "sqrtsoftplus routing requires bias";
  CHECK(!use_visual_bias_ || bias_vl_is_loaded_)
      << "Missing visual router bias_vl";
}

}  // namespace layer
}  // namespace xllm
