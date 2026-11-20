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

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model_loader.h"

namespace xllm::test {

class TensorModelLoader final : public ModelLoader {
 public:
  explicit TensorModelLoader(StateDict weights) {
    shards_.reserve(1);
    shards_.emplace_back(std::make_unique<StateDict>(std::move(weights)));
  }

  std::unique_ptr<Tokenizer> tokenizer() const override { return nullptr; }
  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return shards_;
  }
  std::string model_weights_path() const override { return ""; }

 private:
  std::vector<std::unique_ptr<StateDict>> shards_;
};

// Generate the same CPU FP32 weights on every rank before converting to the
// model dtype. Both graph tests share the checkpoint layout and model geometry.
inline StateDict make_deepseek_v41_dspark_weights(
    const ModelArgs& args,
    const torch::TensorOptions& options) {
  const int64_t hidden_size = args.hidden_size();
  const int64_t q_rank = args.q_lora_rank();
  const int64_t heads = args.n_heads();
  const int64_t head_dim = args.head_dim();
  const int64_t groups = args.o_groups();
  const int64_t o_rank = args.o_lora_rank();
  const int64_t wo_a_width = heads * head_dim / groups;
  const int64_t wo_b_width = groups * o_rank;
  const int64_t intermediate_size = args.moe_intermediate_size();
  const int64_t hc_width = 2 * args.hc_mult() + args.hc_mult() * args.hc_mult();
  const auto random = [&](std::vector<int64_t> shape, float stddev = 0.05f) {
    return (torch::randn(shape, torch::kFloat32) * stddev).to(options);
  };
  // Unit-scale residuals and fan-in-scaled expert projections keep BF16
  // rounding across eager and padded batches from dominating the routing.
  const float expert_input_std =
      1.0f / std::sqrt(static_cast<float>(hidden_size));
  std::unordered_map<std::string, torch::Tensor> weights{
      {"mtp.0.embed.weight",
       random({args.vocab_size(), hidden_size}, /*stddev=*/1.0f)},
      {"mtp.0.head.weight", random({args.vocab_size(), hidden_size})},
      {"mtp.0.main_proj.weight",
       random({hidden_size, hidden_size * args.dspark_num_layers()})},
      {"mtp.0.main_norm.weight", torch::ones({hidden_size}, options)},
      {"mtp.0.norm.weight", torch::ones({hidden_size}, options)},
      {"mtp.0.markov_head.embed.weight",
       random({args.vocab_size(), args.markov_rank()})},
      {"mtp.0.markov_head.head.weight",
       random({args.vocab_size(), args.markov_rank()})},
      {"mtp.0.attn_norm.weight", torch::ones({hidden_size}, options)},
      {"mtp.0.ffn_norm.weight", torch::ones({hidden_size}, options)},
      {"mtp.0.attn.wq_a.weight", random({q_rank, hidden_size})},
      {"mtp.0.attn.wkv.weight", random({head_dim, hidden_size})},
      {"mtp.0.attn.q_norm.weight", torch::ones({q_rank}, options)},
      {"mtp.0.attn.kv_norm.weight", torch::ones({head_dim}, options)},
      {"mtp.0.attn.wq_b.weight", random({heads * head_dim, q_rank})},
      {"mtp.0.attn.wo_a.weight_scale",
       torch::full({wo_b_width, 1}, 1.0 / 256, options)},
      {"mtp.0.attn.wo_b.weight_scale",
       torch::full({hidden_size, 1}, 1.0 / 256, options)},
      {"mtp.0.attn.wo_a.smooth", torch::ones({wo_a_width}, options)},
      {"mtp.0.attn.wo_b.smooth", torch::ones({wo_b_width}, options)},
      {"mtp.0.attn.wo_a.weight",
       random({wo_b_width, wo_a_width}).mul(256).round().to(torch::kInt8)},
      {"mtp.0.attn.wo_b.weight",
       random({hidden_size, wo_b_width}).mul(256).round().to(torch::kInt8)},
      {"mtp.0.attn.attn_sink", torch::zeros({heads}, options)},
      {"mtp.0.ffn.gate.weight", random({args.n_routed_experts(), hidden_size})},
      {"mtp.0.ffn.gate.bias",
       torch::zeros({args.n_routed_experts()},
                    options.dtype(torch::kFloat32))}};
  for (const std::string& name : {"hc_attn_", "hc_ffn_"}) {
    const std::string prefix = "mtp.0." + name;
    weights.emplace(
        prefix + "fn",
        random({hc_width, args.hc_mult() * hidden_size}).to(torch::kFloat32));
    weights.emplace(prefix + "base", random({hc_width}).to(torch::kFloat32));
    weights.emplace(prefix + "scale",
                    torch::full({3}, 0.1, options.dtype(torch::kFloat32)));
  }
  for (int32_t expert = 0; expert < args.n_routed_experts(); ++expert) {
    const std::string prefix =
        "mtp.0.ffn.experts." + std::to_string(expert) + ".";
    weights.emplace(prefix + "w1.weight",
                    random({intermediate_size, hidden_size}, expert_input_std));
    weights.emplace(prefix + "w3.weight",
                    random({intermediate_size, hidden_size}, expert_input_std));
    weights.emplace(prefix + "w2.weight",
                    random({hidden_size, intermediate_size}));
  }
  if (args.n_shared_experts() > 0) {
    const int64_t shared_size = intermediate_size * args.n_shared_experts();
    weights.emplace("mtp.0.ffn.shared_experts.w1.weight",
                    random({shared_size, hidden_size}, expert_input_std));
    weights.emplace("mtp.0.ffn.shared_experts.w3.weight",
                    random({shared_size, hidden_size}, expert_input_std));
    weights.emplace("mtp.0.ffn.shared_experts.w2.weight",
                    random({hidden_size, shared_size}));
  }
  const auto first_layer_weights = weights;
  for (int32_t layer = 1; layer < args.n_layers(); ++layer) {
    const std::string prefix = "mtp." + std::to_string(layer) + ".";
    for (const auto& [name, tensor] : first_layer_weights) {
      weights.emplace(prefix + name.substr(6), tensor.clone());
    }
  }

  return StateDict(std::move(weights));
}

}  // namespace xllm::test
