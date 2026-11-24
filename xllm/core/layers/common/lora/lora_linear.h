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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <vector>

#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/add_matmul.h"
#include "layers/common/linear.h"

namespace xllm {
namespace layer {

class LoRABaseColumnParallelLinearImpl : public torch::nn::Module {
 public:
  LoRABaseColumnParallelLinearImpl() = default;

  LoRABaseColumnParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      bool gather_output,
      const QuantArgs& quant_args,
      ProcessGroup* process_group,
      const torch::TensorOptions& options,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  // Takes ownership of weights already prepared for this TP rank at startup.
  // output_offset places B's rows inside the local base output: fused
  // projections (e.g. QKV) install one independent (A, B) pair per lane.
  void load_state_dict(uint64_t adapter_id,
                       torch::Tensor A,
                       torch::Tensor B,
                       float scaling,
                       int64_t output_offset = 0);

  torch::Tensor forward(torch::Tensor input,
                        std::span<const uint64_t> adapter_ids = {});

  void load_state_dict(const StateDict& state_dict) {
    base_->load_state_dict(state_dict);
  }
  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes) {
    base_->load_state_dict(state_dict, prefixes);
  }
  bool is_weight_loaded() const { return base_->is_weight_loaded(); }
  torch::Tensor weight() const { return base_->weight(); }
  ProcessGroup* process_group() const { return base_->process_group(); }

  void pretty_print(std::ostream& stream) const override {
    stream << name()
           << " (LoRA-wrapped)  base_weight=" << base_->weight().sizes();
  }

 protected:
  ColumnParallelLinear base_{nullptr};

 private:
  struct AdapterWeights {
    torch::Tensor A;
    torch::Tensor B;
    float scaling;
    int64_t output_offset;
  };

  bool supports_local_lora_ = false;
  // Per adapter, one entry per installed output segment (one for plain
  // projections, one per lane for fused ones).
  std::vector<std::vector<AdapterWeights>> adapter_weights_;
};
TORCH_MODULE(LoRABaseColumnParallelLinear);

class LoRABaseRowParallelLinearImpl : public torch::nn::Module {
 public:
  LoRABaseRowParallelLinearImpl() = default;

  LoRABaseRowParallelLinearImpl(
      int64_t in_features,
      int64_t out_features,
      bool bias,
      bool input_is_parallelized,
      bool enable_result_reduction,
      const QuantArgs& quant_args,
      ProcessGroup* process_group,
      const torch::TensorOptions& options,
      const LinearExtraArgs& linear_extra_args = LinearExtraArgs());

  // Takes ownership of weights already prepared for this TP rank at startup.
  void load_state_dict(uint64_t adapter_id,
                       torch::Tensor A,
                       torch::Tensor B,
                       float scaling);

  torch::Tensor forward(torch::Tensor input,
                        std::span<const uint64_t> adapter_ids = {});

  void load_state_dict(const StateDict& state_dict) {
    base_->load_state_dict(state_dict);
  }
  bool is_weight_loaded() const { return base_->is_weight_loaded(); }
  torch::Tensor weight() const { return base_->weight(); }
  ProcessGroup* process_group() const { return base_->process_group(); }

  void pretty_print(std::ostream& stream) const override {
    stream << name()
           << " (LoRA-wrapped)  base_weight=" << base_->weight().sizes();
  }

 protected:
  RowParallelLinear base_{nullptr};

 private:
  struct AdapterWeights {
    torch::Tensor A;
    torch::Tensor B;
    float scaling;
  };

  bool supports_local_lora_ = false;
  std::vector<std::optional<AdapterWeights>> adapter_weights_;
};
TORCH_MODULE(LoRABaseRowParallelLinear);

class LoRABaseAddMatmulWeightTransposedImpl : public torch::nn::Module {
 public:
  LoRABaseAddMatmulWeightTransposedImpl() = default;

  LoRABaseAddMatmulWeightTransposedImpl(int64_t in_features,
                                        int64_t out_features,
                                        bool with_bias,
                                        const torch::TensorOptions& options,
                                        const QuantArgs& quant_args);

  // Takes ownership of weights already prepared on the device at startup.
  void load_state_dict(uint64_t adapter_id,
                       torch::Tensor A,
                       torch::Tensor B,
                       float scaling);

  torch::Tensor forward(const torch::Tensor& input,
                        std::span<const uint64_t> adapter_ids = {});

  void load_state_dict(const StateDict& state_dict) {
    base_->load_state_dict(state_dict);
  }
  void verify_loaded_weights(const std::string& prefix) const {
    base_->verify_loaded_weights(prefix);
  }
  int64_t in_features() const { return in_features_; }
  int64_t out_features() const { return out_features_; }
  torch::ScalarType weight_dtype() const { return weight_dtype_; }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " (LoRA-wrapped)  base_weight=[" << out_features_
           << ", " << in_features_ << "]";
  }

 protected:
  AddMatmulWeightTransposed base_{nullptr};

 private:
  struct AdapterWeights {
    torch::Tensor A;
    torch::Tensor B;
    float scaling;
  };

  int64_t in_features_ = 0;
  int64_t out_features_ = 0;
  torch::ScalarType weight_dtype_ = torch::kFloat32;
  std::vector<std::optional<AdapterWeights>> adapter_weights_;
};
TORCH_MODULE(LoRABaseAddMatmulWeightTransposed);

}  // namespace layer
}  // namespace xllm
