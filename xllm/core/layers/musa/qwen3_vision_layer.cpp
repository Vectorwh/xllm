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

#include "layers/musa/qwen3_vision_layer.h"

#include <memory>
#include <tuple>

#include "layers/musa/qwen3_vision_attention.h"
#include "layers/qwen2_5_vision_layer.h"

namespace xllm {
namespace layer {

Qwen3_VisionLayerImpl::Qwen3_VisionLayerImpl(const ModelContext& context) {
  const ModelArgs& args = context.get_model_args();
  const QuantArgs& quant_config = context.get_quant_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const torch::TensorOptions& options = context.get_tensor_options();
  const int64_t dim = args.mm_hidden_size();
  attention_ =
      register_module("self_attn",
                      Qwen2VisionAttention(
                          std::make_shared<Qwen3VisionAttentionImpl>(context)));
  norm1_ = register_module("norm1", RMSNorm(dim, args.rms_norm_eps(), options));
  norm2_ = register_module("norm2", RMSNorm(dim, args.rms_norm_eps(), options));
  const bool is_gated = apply_qwen3_vision_style(norm1_, norm2_);

  mlp_ = register_module("mlp",
                         DenseMLP(dim,
                                  args.mm_intermediate_size(),
                                  /*is_gated=*/is_gated,
                                  /*has_bias=*/true,
                                  args.mm_hidden_act(),
                                  /*enable_result_reduction=*/true,
                                  quant_config,
                                  parallel_args.tp_group_,
                                  options));
}

void Qwen3_VisionLayerImpl::load_state_dict(const StateDict& state_dict) {
  load_qwen3_vision_state_dict(state_dict, attention_, mlp_, norm1_, norm2_);
}

torch::Tensor Qwen3_VisionLayerImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec,
    int32_t node_id) {
  (void)node_id;
  torch::Tensor norm_output1 = std::get<0>(norm1_(hidden_states));
  torch::Tensor output =
      hidden_states +
      attention_(
          norm_output1, m_cos_pos, m_sin_pos, cu_seq_len, cu_seq_len_vec);
  torch::Tensor norm_output2 = std::get<0>(norm2_(output));
  output = output + mlp_(norm_output2);
  return output;
}

}  // namespace layer
}  // namespace xllm
