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

#include "kernels/mlu/bangc/include/causal_conv1d_prefill.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cstdint>
#include <optional>

#include "kernels/mlu/bangc/src/causal_conv1d_prefill/launch.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu {
namespace {

bool is_supported_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
         dtype == torch::kBFloat16;
}

int32_t to_kernel_dtype(torch::ScalarType dtype) {
  if (dtype == torch::kFloat32) {
    return bangc::kPrefillFloat32;
  }
  if (dtype == torch::kFloat16) {
    return bangc::kPrefillFloat16;
  }
  return bangc::kPrefillBFloat16;
}

bool has_positive_strides(const torch::Tensor& tensor) {
  for (const int64_t stride : tensor.strides()) {
    if (stride <= 0) {
      return false;
    }
  }
  return true;
}

// The distinct-read prefill path requires an explicit final-slot table plus a
// per-request validity mask; the plain prefill path additionally allows an
// independent initial-slot table. Anything else stays on the Triton kernel.
// num_requests always equals cache_indices->numel(): the caller derives it from
// that tensor and already rejected an absent table.
bool matches_state_contract(const std::optional<torch::Tensor>& cache_indices,
                            const std::optional<torch::Tensor>& initial_state,
                            const std::optional<torch::Tensor>& read_indices,
                            const torch::Device& device,
                            int64_t num_requests) {
  if (!cache_indices.has_value() ||
      !bangc::detail::check_tensor(*cache_indices, torch::kInt32, device) ||
      cache_indices->dim() != 1) {
    return false;
  }
  if (!initial_state.has_value() ||
      !bangc::detail::check_tensor(*initial_state, torch::kBool, device) ||
      initial_state->dim() != 1 || initial_state->numel() != num_requests) {
    return false;
  }
  if (read_indices.has_value() && read_indices->defined()) {
    if (!bangc::detail::check_tensor(*read_indices, torch::kInt32, device) ||
        read_indices->dim() != 1 || read_indices->numel() != num_requests) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool bangc_causal_conv1d_prefill(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& batch,
    const torch::Tensor& token_block_offset,
    int32_t nt,
    const std::optional<torch::Tensor>& bias_opt,
    const std::optional<torch::Tensor>& cache_indices_opt,
    const std::optional<torch::Tensor>& has_initial_state_opt,
    const std::optional<torch::Tensor>& initial_state_idx_opt,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    const std::optional<torch::Tensor>& read_cache_indices_opt,
    bool inplace_final_state,
    torch::Tensor& out) {
  if (!x.defined() || !x.device().is_privateuseone()) {
    return false;
  }
  if (x.dim() != 2 || weight.dim() != 2 || conv_states.dim() != 3) {
    return false;
  }
  if (!is_supported_dtype(x.scalar_type()) ||
      weight.scalar_type() != x.scalar_type() ||
      conv_states.scalar_type() != x.scalar_type() ||
      out.scalar_type() != x.scalar_type()) {
    return false;
  }
  if (weight.device() != x.device() || conv_states.device() != x.device() ||
      out.device() != x.device() || out.sizes() != x.sizes()) {
    return false;
  }
  if (weight.size(0) != x.size(0) || conv_states.size(1) != x.size(0)) {
    return false;
  }
  if (weight.size(1) < 2 || weight.size(1) > 5 ||
      conv_states.size(2) != weight.size(1) - 1) {
    return false;
  }
  if (!has_positive_strides(x) || !has_positive_strides(weight) ||
      !has_positive_strides(conv_states) || !has_positive_strides(out)) {
    return false;
  }
  if (!bangc::detail::check_tensor(
          query_start_loc, torch::kInt32, x.device()) ||
      query_start_loc.dim() != 1 ||
      !bangc::detail::check_tensor(batch, torch::kInt32, x.device()) ||
      batch.dim() != 1 ||
      !bangc::detail::check_tensor(
          token_block_offset, torch::kInt32, x.device()) ||
      token_block_offset.dim() != 1) {
    return false;
  }
  // The chunk tables are allocated to a fixed capacity (padded with -1 beyond
  // the scheduled blocks); nt carries the real number of scheduled blocks, so
  // only require that it fits within the tables rather than covering them.
  if (nt <= 0 || batch.numel() != token_block_offset.numel() ||
      nt > batch.numel()) {
    return false;
  }
  // Checkpoint indices and speculative accepted-token counts belong to the
  // update/decode contract, not to this prefill kernel.
  if (initial_state_idx_opt.has_value() ||
      num_accepted_tokens_opt.has_value() || !inplace_final_state) {
    return false;
  }
  const int64_t num_requests =
      cache_indices_opt.has_value() ? cache_indices_opt->numel() : 0;
  if (num_requests <= 0 || query_start_loc.numel() != num_requests + 1) {
    return false;
  }
  if (!matches_state_contract(cache_indices_opt,
                              has_initial_state_opt,
                              read_cache_indices_opt,
                              x.device(),
                              num_requests)) {
    return false;
  }
  if (bias_opt.has_value()) {
    if (!bias_opt->defined() || bias_opt->device() != x.device() ||
        bias_opt->scalar_type() != x.scalar_type() || bias_opt->dim() != 1 ||
        bias_opt->numel() != x.size(0) || bias_opt->stride(0) <= 0) {
      return false;
    }
  }

  const torch::Tensor& cache_indices = *cache_indices_opt;
  const bool use_read_indices =
      read_cache_indices_opt.has_value() && read_cache_indices_opt->defined();

  torch::DeviceGuard guard(x.device());
  bangc::PrefillParams p{};
  p.x = x.data_ptr();
  p.weight = weight.data_ptr();
  p.state = conv_states.data_ptr();
  p.output = out.data_ptr();
  p.bias = bias_opt.has_value() ? bias_opt->data_ptr() : nullptr;
  p.query = query_start_loc.data_ptr<int32_t>();
  p.batch = batch.data_ptr<int32_t>();
  p.chunks = token_block_offset.data_ptr<int32_t>();
  p.indices = cache_indices.data_ptr<int32_t>();
  p.read_indices =
      use_read_indices ? read_cache_indices_opt->data_ptr<int32_t>() : nullptr;
  p.initial = has_initial_state_opt->data_ptr<bool>();
  p.dim = x.size(0);
  p.nt = nt;
  p.x_channel_stride = x.stride(0);
  p.x_token_stride = x.stride(1);
  p.weight_channel_stride = weight.stride(0);
  p.weight_width_stride = weight.stride(1);
  p.state_slot_stride = conv_states.stride(0);
  p.state_channel_stride = conv_states.stride(1);
  p.state_history_stride = conv_states.stride(2);
  p.output_channel_stride = out.stride(0);
  p.output_token_stride = out.stride(1);
  p.index_stride = cache_indices.stride(0);
  p.read_index_stride =
      use_read_indices ? read_cache_indices_opt->stride(0) : 0;
  p.bias_stride = bias_opt.has_value() ? bias_opt->stride(0) : 0;
  p.width = static_cast<int32_t>(weight.size(1));
  const int32_t dtype = to_kernel_dtype(x.scalar_type());
  const cnrtRet_t result =
      bangc::launch_prefill(p, dtype, torch_mlu::getCurMLUStream());
  CHECK_EQ(result, cnrtSuccess) << "Causal Conv1d prefill launch error";
  return true;
}

}  // namespace xllm::kernel::mlu
