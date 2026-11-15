/* Copyright 2026 The xLLM Authors.

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

// Registers torch.ops.xllm_ops for the MUSA dispatcher key. The device
// kernels live in this directory or in shared CUDA sources compiled into the
// MUSA library; this file only binds those kernel::musa entry points.

#include "core/kernels/musa/musa_ops_library.h"

#include <glog/logging.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "core/kernels/musa/musa_ops_api.h"

namespace xllm {
namespace {

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps) {
  auto output = torch::empty_like(input);
  auto rms_weight = weight;
  kernel::musa::rms_norm(output, input, rms_weight, eps);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    double eps) {
  auto rms_weight = weight;
  kernel::musa::fused_add_rms_norm(input, residual, rms_weight, eps);
  return std::make_tuple(input, residual);
}

torch::Tensor silu_and_mul(const torch::Tensor& input) {
  auto sizes = input.sizes().vec();
  CHECK(!sizes.empty() && sizes.back() % 2 == 0)
      << "silu_and_mul: last dim must be even, got " << input.sizes();
  sizes.back() /= 2;
  auto output = torch::empty(sizes, input.options());
  kernel::musa::active(output, input, "silu", /*is_gated=*/true);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> moe_fused_topk(
    const torch::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    const std::string& scoring_func) {
  CHECK_EQ(gating_output.dim(), 2) << "MoE gating output must be a matrix";
  CHECK_GT(topk, 0) << "MoE topk must be positive";
  CHECK_LE(topk, gating_output.size(1)) << "MoE topk exceeds the expert count";
  auto logits = gating_output;
  return kernel::musa::moe_fused_topk(
      logits, topk, renormalize, std::nullopt, scoring_func);
}

torch::Tensor fused_qk_norm_rope(torch::Tensor& qkv,
                                 int64_t num_heads_q,
                                 int64_t num_heads_k,
                                 int64_t num_heads_v,
                                 int64_t head_dim,
                                 double eps,
                                 const torch::Tensor& q_weight,
                                 const torch::Tensor& k_weight,
                                 const torch::Tensor& cos_sin_cache,
                                 bool interleaved,
                                 const torch::Tensor& position_ids) {
  kernel::musa::fused_qk_norm_rope(qkv,
                                   num_heads_q,
                                   num_heads_k,
                                   num_heads_v,
                                   head_dim,
                                   eps,
                                   q_weight,
                                   k_weight,
                                   cos_sin_cache,
                                   interleaved,
                                   position_ids);
  return qkv;
}

torch::Tensor reshape_paged_cache_op(const torch::Tensor& slot_mapping,
                                     const torch::Tensor& keys,
                                     const torch::Tensor& values,
                                     torch::Tensor& key_cache,
                                     torch::Tensor& value_cache) {
  kernel::musa::reshape_paged_cache(
      slot_mapping, keys, values, key_cache, value_cache);
  return key_cache;
}

}  // namespace

void ensure_xllm_ops_registered() {}

}  // namespace xllm

TORCH_LIBRARY(xllm_ops, m) {
  m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def(
      "fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor "
      "weight, float eps) -> (Tensor, Tensor)");
  m.def("silu_and_mul(Tensor input) -> Tensor");
  m.def(
      "moe_fused_topk(Tensor gating_output, int topk, bool renormalize, str "
      "scoring_func) -> (Tensor, Tensor)");
  m.def(
      "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, "
      "int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor "
      "k_weight, Tensor cos_sin_cache, bool interleaved, Tensor position_ids) "
      "-> Tensor");
  m.def(
      "reshape_paged_cache(Tensor slot_mapping, Tensor keys, Tensor values, "
      "Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ops, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(xllm::rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(xllm::fused_add_rms_norm));
  m.impl("silu_and_mul", TORCH_FN(xllm::silu_and_mul));
  m.impl("moe_fused_topk", TORCH_FN(xllm::moe_fused_topk));
  m.impl("fused_qk_norm_rope", TORCH_FN(xllm::fused_qk_norm_rope));
  m.impl("reshape_paged_cache", TORCH_FN(xllm::reshape_paged_cache_op));
}
