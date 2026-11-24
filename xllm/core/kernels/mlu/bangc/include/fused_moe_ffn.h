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

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

namespace xllm::kernel::mlu::bangc {

struct MoeFfnWeights {
  torch::Tensor gate_up;
  torch::Tensor down;
  torch::Tensor gate_up_scale;
  torch::Tensor down_scale;
  torch::Tensor smooth;
};

struct MoeFfnPackOptions {
  int64_t weight_bits;
  int64_t group_size;
  torch::ScalarType output_dtype;
  bool is_gated;
  std::string_view activation;
  std::optional<float> swiglu_limit;
};

class PackedMoeFfn final {
 public:
  // Repack the registered weights in place once the whole expert shard is
  // loaded. Ineligible shards retain their grouped GEMM layout.
  static std::optional<PackedMoeFfn> try_pack(MoeFfnWeights weights,
                                              const MoeFfnPackOptions& options);

  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& input_scale,
                        const torch::Tensor& counts) const;

 private:
  explicit PackedMoeFfn(MoeFfnWeights weights) : weights_(std::move(weights)) {}
  MoeFfnWeights weights_;
};

}  // namespace xllm::kernel::mlu::bangc
