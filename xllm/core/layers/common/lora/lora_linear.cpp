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

#include "layers/common/lora/lora_linear.h"

#include <glog/logging.h>

#include <cstddef>
#include <utility>

#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor compute_lora_delta(const torch::Tensor& input,
                                 const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 float scaling,
                                 ProcessGroup* tp_group) {
  CHECK_EQ(input.size(-1), a.size(1));
  CHECK(input.device() == a.device());
  CHECK(input.scalar_type() == a.scalar_type());
  torch::Tensor intermediate =
      torch::matmul(input, a.transpose(/*dim0=*/0, /*dim1=*/1));
  // Row-parallel projections sum partials in each adapter's rank space.
  if (tp_group != nullptr && tp_group->world_size() > 1) {
    intermediate = parallel_state::reduce(intermediate, tp_group);
  }
  torch::Tensor delta =
      torch::matmul(intermediate, b.transpose(/*dim0=*/0, /*dim1=*/1));
  if (scaling != 1.0f) {
    delta.mul_(scaling);
  }
  return delta;
}

torch::Tensor add_lora_delta(const torch::Tensor& input,
                             torch::Tensor output,
                             const torch::Tensor& a,
                             const torch::Tensor& b,
                             float scaling,
                             ProcessGroup* tp_group) {
  torch::Tensor delta = compute_lora_delta(input, a, b, scaling, tp_group);
  CHECK(delta.sizes() == output.sizes())
      << "LoRA must not broadcast the base output";
  output.add_(delta.to(output.scalar_type()));
  return output;
}

}  // namespace

LoRABaseColumnParallelLinearImpl::LoRABaseColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : base_(in_features,
            out_features,
            bias,
            gather_output,
            quant_args,
            process_group,
            options,
            linear_extra_args),
      supports_local_lora_(!gather_output) {
  register_module("base", base_);
}

void LoRABaseColumnParallelLinearImpl::load_state_dict(uint64_t adapter_id,
                                                       torch::Tensor A,
                                                       torch::Tensor B,
                                                       float scaling,
                                                       int64_t output_offset) {
  CHECK(supports_local_lora_)
      << "Column LoRA weights require local base output";
  CHECK_GT(adapter_id, 0);
  CHECK_LT(adapter_id, adapter_weights_.max_size());
  CHECK_GE(output_offset, 0);
  CHECK_LE(output_offset + B.size(0), base_->weight().size(0))
      << "LoRA B exceeds the local output of the base projection";
  const std::size_t slot = static_cast<std::size_t>(adapter_id);
  if (slot >= adapter_weights_.size()) {
    adapter_weights_.resize(slot + 1);
  }
  for (const AdapterWeights& existing : adapter_weights_[slot]) {
    CHECK_NE(existing.output_offset, output_offset)
        << "LoRA weights already installed for adapter " << adapter_id;
  }
  adapter_weights_[slot].push_back(
      AdapterWeights{std::move(A), std::move(B), scaling, output_offset});
}

torch::Tensor LoRABaseColumnParallelLinearImpl::forward(
    torch::Tensor input,
    std::span<const uint64_t> adapter_ids) {
  torch::Tensor output = base_->forward(input);
  for (const uint64_t adapter_id : adapter_ids) {
    if (adapter_id >= adapter_weights_.size()) {
      continue;
    }
    for (const AdapterWeights& weights :
         adapter_weights_[static_cast<std::size_t>(adapter_id)]) {
      if (weights.output_offset == 0 && weights.B.size(0) == output.size(-1)) {
        output = add_lora_delta(input,
                                output,
                                weights.A,
                                weights.B,
                                weights.scaling,
                                /*tp_group=*/nullptr);
      } else {
        const torch::Tensor delta = compute_lora_delta(
            input, weights.A, weights.B, weights.scaling, /*tp_group=*/nullptr);
        output.narrow(/*dim=*/-1, weights.output_offset, weights.B.size(0))
            .add_(delta.to(output.scalar_type()));
      }
    }
  }
  return output;
}

LoRABaseRowParallelLinearImpl::LoRABaseRowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool enable_result_reduction,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : base_(in_features,
            out_features,
            bias,
            input_is_parallelized,
            enable_result_reduction,
            quant_args,
            process_group,
            options,
            linear_extra_args),
      // A single-rank group needs neither parallel input nor reduction for
      // the delta to match the base output.
      supports_local_lora_((input_is_parallelized && enable_result_reduction) ||
                           process_group == nullptr ||
                           process_group->world_size() <= 1) {
  register_module("base", base_);
}

void LoRABaseRowParallelLinearImpl::load_state_dict(uint64_t adapter_id,
                                                    torch::Tensor A,
                                                    torch::Tensor B,
                                                    float scaling) {
  CHECK(supports_local_lora_)
      << "Row LoRA weights require local input and reduced base output";
  CHECK_GT(adapter_id, 0);
  CHECK_LT(adapter_id, adapter_weights_.max_size());
  const std::size_t slot = static_cast<std::size_t>(adapter_id);
  if (slot >= adapter_weights_.size()) {
    adapter_weights_.resize(slot + 1);
  }
  CHECK(!adapter_weights_[slot].has_value())
      << "LoRA weights already installed for adapter " << adapter_id;
  adapter_weights_[slot] = AdapterWeights{std::move(A), std::move(B), scaling};
}

torch::Tensor LoRABaseRowParallelLinearImpl::forward(
    torch::Tensor input,
    std::span<const uint64_t> adapter_ids) {
  torch::Tensor output = base_->forward(input);
  ProcessGroup* tp_group = base_->process_group();
  for (const uint64_t adapter_id : adapter_ids) {
    if (adapter_id >= adapter_weights_.size()) {
      continue;
    }
    const std::optional<AdapterWeights>& weights =
        adapter_weights_[static_cast<std::size_t>(adapter_id)];
    if (weights.has_value()) {
      output = add_lora_delta(
          input, output, weights->A, weights->B, weights->scaling, tp_group);
    }
  }
  return output;
}

LoRABaseAddMatmulWeightTransposedImpl::LoRABaseAddMatmulWeightTransposedImpl(
    int64_t in_features,
    int64_t out_features,
    bool with_bias,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args)
    : base_(in_features, out_features, with_bias, options, quant_args),
      in_features_(in_features),
      out_features_(out_features),
      weight_dtype_(options.dtype().toScalarType()) {
  register_module("base", base_);
}

void LoRABaseAddMatmulWeightTransposedImpl::load_state_dict(uint64_t adapter_id,
                                                            torch::Tensor A,
                                                            torch::Tensor B,
                                                            float scaling) {
  CHECK_GT(adapter_id, 0);
  CHECK_LT(adapter_id, adapter_weights_.max_size());
  CHECK_EQ(A.size(1), in_features_);
  CHECK_EQ(B.size(0), out_features_);
  const std::size_t slot = static_cast<std::size_t>(adapter_id);
  if (slot >= adapter_weights_.size()) {
    adapter_weights_.resize(slot + 1);
  }
  CHECK(!adapter_weights_[slot].has_value())
      << "LoRA weights already installed for adapter " << adapter_id;
  adapter_weights_[slot] = AdapterWeights{std::move(A), std::move(B), scaling};
}

torch::Tensor LoRABaseAddMatmulWeightTransposedImpl::forward(
    const torch::Tensor& input,
    std::span<const uint64_t> adapter_ids) {
  torch::Tensor output = base_->forward(input);
  for (const uint64_t adapter_id : adapter_ids) {
    if (adapter_id >= adapter_weights_.size()) {
      continue;
    }
    const std::optional<AdapterWeights>& weights =
        adapter_weights_[static_cast<std::size_t>(adapter_id)];
    if (weights.has_value()) {
      output = add_lora_delta(input,
                              output,
                              weights->A,
                              weights->B,
                              weights->scaling,
                              /*tp_group=*/nullptr);
    }
  }
  return output;
}

}  // namespace layer
}  // namespace xllm
