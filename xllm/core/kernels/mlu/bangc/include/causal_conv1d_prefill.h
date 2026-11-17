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

namespace xllm::kernel::mlu {

// Dispatches the causal Conv1d prefill convolution to the BangC kernel.
// Returns true when the launch was issued. Returns false for inputs outside the
// supported contract, in which case no device work is enqueued and the output
// is left untouched so the caller can select the Triton path beforehand.
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
    torch::Tensor& out);

}  // namespace xllm::kernel::mlu
