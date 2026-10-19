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

#include <ATen/ops/scaled_dot_product_attention.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/activation.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "models/llm/llm_model_base.h"
#include "models/vlm/qwen3_vl_base.h"

namespace xllm {
namespace layer {

namespace joyimage_qwen3_detail {

inline StateDict weight_only_state_dict(const StateDict& state_dict) {
  std::unordered_map<std::string, torch::Tensor> weights;
  torch::Tensor weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    weights.emplace("weight", weight);
  }
  return StateDict(std::move(weights));
}

inline torch::Tensor unsharded_row_matmul(
    const torch::Tensor& input,
    const RowParallelLinear& linear,
    ProcessGroup* process_group,
    const std::optional<torch::Tensor>& bias = std::nullopt) {
  torch::Tensor gathered_input = input;
  torch::Tensor gathered_weight = linear->weight();
  if (process_group->world_size() > 1) {
    gathered_input =
        xllm::parallel_state::gather(input, process_group, /*dim=*/-1);
    gathered_weight =
        xllm::parallel_state::gather(gathered_weight, process_group, /*dim=*/1);
  }
  xllm::kernel::MatmulParams params;
  params.a = gathered_input;
  params.b = gathered_weight;
  params.bias = bias;
  return xllm::kernel::matmul(params);
}

inline torch::Tensor rotate_half(const torch::Tensor& input) {
  const int64_t half_dim = input.size(-1) / 2;
  return torch::cat({-input.slice(-1, half_dim, input.size(-1)),
                     input.slice(-1, 0, half_dim)},
                    -1);
}

}  // namespace joyimage_qwen3_detail

class JoyImageQwen3RMSNormImpl final : public torch::nn::Module {
 public:
  JoyImageQwen3RMSNormImpl(int64_t dim,
                           double eps,
                           const torch::TensorOptions& options)
      : norm_dim_(dim), eps_(eps) {
    weight_ = register_parameter(
        "weight", torch::empty({dim}, options), /*requires_grad=*/false);
  }

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      torch::Tensor input,
      std::optional<torch::Tensor> residual = std::nullopt) {
    const std::vector<int64_t> original_shape = input.sizes().vec();
    input = input.reshape({-1, norm_dim_});
    std::optional<torch::Tensor> residual_out = std::nullopt;
    if (residual.has_value()) {
      residual_out = input + residual.value().reshape({-1, norm_dim_});
      input = residual_out.value();
    }
    const torch::ScalarType input_dtype = input.scalar_type();
    const torch::Tensor input_fp32 = input.to(torch::kFloat32);
    const torch::Tensor variance = input_fp32.pow(2).mean(-1, true);
    torch::Tensor output =
        weight_ * (input_fp32 * torch::rsqrt(variance + eps_)).to(input_dtype);
    output = output.view(original_shape);
    if (residual_out.has_value()) {
      residual_out = residual_out.value().view(original_shape);
    }
    return std::make_tuple(output, residual_out);
  }

  void load_state_dict(const StateDict& state_dict) {
    torch::Tensor weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes());
      weight_.data().copy_(weight);
      weight_is_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
  }

 private:
  torch::Tensor weight_;
  bool weight_is_loaded_ = false;
  int64_t norm_dim_;
  double eps_;
};
TORCH_MODULE(JoyImageQwen3RMSNorm);

class JoyImageQwen3MLPImpl final : public torch::nn::Module {
 public:
  JoyImageQwen3MLPImpl(int64_t hidden_size,
                       int64_t intermediate_size,
                       bool is_gated,
                       bool has_bias,
                       const std::string& hidden_act,
                       const QuantArgs& quant_args,
                       ProcessGroup* process_group,
                       const torch::TensorOptions& options)
      : is_gated_(is_gated),
        process_group_(process_group),
        gate_up_bias_is_loaded_(is_gated ? 2 : 1, 0) {
    const int64_t output_size =
        is_gated_ ? intermediate_size * 2 : intermediate_size;
    gate_up_proj_ =
        register_module("gate_up_proj",
                        ColumnParallelLinear(hidden_size,
                                             output_size,
                                             has_bias,
                                             /*gather_output=*/false,
                                             quant_args,
                                             process_group_,
                                             options));
    down_proj_ =
        register_module("down_proj",
                        RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          /*bias=*/false,
                                          /*input_is_parallelized=*/true,
                                          /*enable_result_reduction=*/true,
                                          quant_args,
                                          process_group_,
                                          options));
    activation_ = register_module("activation",
                                  Activation(hidden_act, /*is_gated=*/false));
    if (has_bias) {
      down_bias_ = register_parameter("down_bias",
                                      torch::empty({hidden_size}, options),
                                      /*requires_grad=*/false);
    }
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    torch::Tensor projected = gate_up_proj_->forward(hidden_states);
    torch::Tensor activated;
    if (is_gated_) {
      const std::vector<torch::Tensor> chunks = projected.chunk(2, -1);
      activated = torch::silu(chunks[0]) * chunks[1];
    } else {
      activation_->forward(projected, activated);
    }
    const std::optional<torch::Tensor> bias =
        down_bias_.defined() ? std::optional<torch::Tensor>(down_bias_)
                             : std::nullopt;
    return joyimage_qwen3_detail::unsharded_row_matmul(
        activated, down_proj_, process_group_, bias);
  }

  void load_state_dict(const StateDict& state_dict) {
    load_state_dict(state_dict, {"gate_proj.", "up_proj."}, "down_proj.");
  }

  void load_state_dict(const StateDict& state_dict,
                       const std::vector<std::string>& gate_up_names,
                       const std::string& down_name) {
    if (is_gated_) {
      gate_up_proj_->load_state_dict(state_dict, gate_up_names);
    } else {
      CHECK_EQ(gate_up_names.size(), 1);
      gate_up_proj_->load_state_dict(
          state_dict.get_dict_with_prefix(gate_up_names[0]));
    }
    const StateDict down_dict = state_dict.get_dict_with_prefix(down_name);
    down_proj_->load_state_dict(
        joyimage_qwen3_detail::weight_only_state_dict(down_dict));
    if (down_bias_.defined()) {
      for (size_t index = 0; index < gate_up_names.size(); ++index) {
        if (state_dict.get_tensor(gate_up_names[index] + "bias").defined()) {
          gate_up_bias_is_loaded_[index] = 1;
        }
      }
      torch::Tensor bias = down_dict.get_tensor("bias");
      if (bias.defined()) {
        down_bias_.data().copy_(bias);
        down_bias_is_loaded_ = true;
      }
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(gate_up_proj_->is_weight_loaded())
        << "gate/up projection weight is not loaded for " << prefix;
    CHECK(down_proj_->is_weight_loaded())
        << "down projection weight is not loaded for " << prefix;
    if (down_bias_.defined()) {
      CHECK(std::all_of(gate_up_bias_is_loaded_.begin(),
                        gate_up_bias_is_loaded_.end(),
                        [](char loaded) { return loaded != 0; }))
          << "gate/up projection bias is not loaded for " << prefix;
      CHECK(down_bias_is_loaded_)
          << "down projection bias is not loaded for " << prefix;
    }
  }

 private:
  bool is_gated_;
  ProcessGroup* process_group_;
  std::vector<char> gate_up_bias_is_loaded_;
  bool down_bias_is_loaded_ = false;
  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};
  Activation activation_{nullptr};
  torch::Tensor down_bias_;
};
TORCH_MODULE(JoyImageQwen3MLP);

class JoyImageQwen3AttentionImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3AttentionImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const QuantArgs& quant_args = context.get_quant_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    process_group_ = parallel_args.tp_group_;
    const torch::TensorOptions options = context.get_tensor_options();
    const int64_t tensor_parallel_size = process_group_->world_size();
    const int64_t total_num_heads = args.n_heads();
    const int64_t total_num_kv_heads =
        args.n_kv_heads().value_or(total_num_heads);
    CHECK_EQ(total_num_heads % tensor_parallel_size, 0);
    num_heads_ = total_num_heads / tensor_parallel_size;
    int64_t num_kv_head_replicas = 1;
    if (total_num_kv_heads >= tensor_parallel_size) {
      CHECK_EQ(total_num_kv_heads % tensor_parallel_size, 0);
      num_kv_heads_ = total_num_kv_heads / tensor_parallel_size;
    } else {
      CHECK_EQ(tensor_parallel_size % total_num_kv_heads, 0);
      num_kv_heads_ = 1;
      num_kv_head_replicas = tensor_parallel_size / total_num_kv_heads;
    }
    head_dim_ = args.head_dim();
    q_size_ = num_heads_ * head_dim_;
    kv_size_ = num_kv_heads_ * head_dim_;
    scaling_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    CHECK(!args.attention_bias())
        << "JoyImage Qwen3-VL reference attention expects bias-free QKV";

    qkv_proj_ = register_module("qkv_proj",
                                QKVParallelLinear(args.hidden_size(),
                                                  num_heads_,
                                                  num_kv_heads_,
                                                  head_dim_,
                                                  num_kv_head_replicas,
                                                  /*bias=*/false,
                                                  /*gather_output=*/false,
                                                  parallel_args,
                                                  options,
                                                  quant_args));
    o_proj_ =
        register_module("o_proj",
                        RowParallelLinear(total_num_heads * head_dim_,
                                          args.hidden_size(),
                                          /*bias=*/false,
                                          /*input_is_parallelized=*/true,
                                          /*enable_result_reduction=*/true,
                                          quant_args,
                                          process_group_,
                                          options));
    q_norm_ = register_module(
        "q_norm",
        JoyImageQwen3RMSNorm(head_dim_, args.rms_norm_eps(), options));
    k_norm_ = register_module(
        "k_norm",
        JoyImageQwen3RMSNorm(head_dim_, args.rms_norm_eps(), options));
  }

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attention_metadata,
                        KVCache& kv_cache) {
    (void)positions;
    (void)kv_cache;
    CHECK(attention_metadata.prefill_without_cache);
    CHECK_EQ(attention_metadata.q_seq_lens_vec.size(), 1);

    torch::Tensor qkv_weight = qkv_proj_->weight();
    int32_t split_dim = 0;
    if (process_group_->world_size() > 1) {
      qkv_weight = process_group_->allgather_base_sync(qkv_weight);
      split_dim = 1;
    }
    const std::vector<torch::Tensor> weights =
        qkv_weight.split({q_size_, kv_size_, kv_size_}, split_dim);
    std::vector<torch::Tensor> projections;
    projections.reserve(3);
    for (const torch::Tensor& rank_weights : weights) {
      torch::Tensor weight = rank_weights;
      if (process_group_->world_size() > 1) {
        weight = rank_weights.reshape({-1, hidden_states.size(-1)});
      }
      xllm::kernel::MatmulParams params;
      params.a = hidden_states;
      params.b = weight;
      torch::Tensor projection = xllm::kernel::matmul(params);
      if (process_group_->world_size() > 1) {
        projection = xllm::parallel_state::scatter(
            projection, process_group_, /*dim=*/-1);
      }
      projections.emplace_back(projection);
    }

    const int64_t token_count = hidden_states.size(0);
    torch::Tensor query = std::get<0>(q_norm_->forward(projections[0]));
    torch::Tensor key = std::get<0>(k_norm_->forward(projections[1]));
    torch::Tensor value = projections[2];
    CHECK(attention_metadata.mrope_cos.defined());
    CHECK(attention_metadata.mrope_sin.defined());
    torch::Tensor cosine = attention_metadata.mrope_cos;
    torch::Tensor sine = attention_metadata.mrope_sin;
    if (cosine.dim() == 3 && cosine.size(0) == 1) {
      cosine = cosine.squeeze(0);
      sine = sine.squeeze(0);
    }
    cosine = cosine.unsqueeze(1);
    sine = sine.unsqueeze(1);
    query = query.view({token_count, num_heads_, head_dim_});
    key = key.view({token_count, num_kv_heads_, head_dim_});
    query = query * cosine + joyimage_qwen3_detail::rotate_half(query) * sine;
    key = key * cosine + joyimage_qwen3_detail::rotate_half(key) * sine;

    const torch::Tensor local_query = query.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor local_key = key.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor local_value =
        value.view({token_count, num_kv_heads_, head_dim_})
            .permute({1, 0, 2})
            .unsqueeze(0);
    torch::Tensor output = torch::scaled_dot_product_attention(
        local_query,
        local_key,
        local_value,
        /*attn_mask=*/std::nullopt,
        /*dropout_p=*/0.0,
        /*is_causal=*/attention_metadata.is_causal,
        /*scale=*/scaling_,
        /*enable_gqa=*/true);
    output = output.squeeze(0).permute({1, 0, 2}).contiguous().reshape(
        {token_count, q_size_});
    return joyimage_qwen3_detail::unsharded_row_matmul(
        output, o_proj_, process_group_);
  }

  void load_state_dict(const StateDict& state_dict) {
    qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
    o_proj_->load_state_dict(joyimage_qwen3_detail::weight_only_state_dict(
        state_dict.get_dict_with_prefix("o_proj.")));
    q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
    k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(qkv_proj_->is_weight_loaded())
        << "Q/K/V projection weights are not loaded for " << prefix;
    CHECK(o_proj_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "o_proj.weight";
    q_norm_->verify_loaded_weights(prefix + "q_norm.");
    k_norm_->verify_loaded_weights(prefix + "k_norm.");
  }

 private:
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  double scaling_ = 0.0;
  ProcessGroup* process_group_ = nullptr;
  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  JoyImageQwen3RMSNorm q_norm_{nullptr};
  JoyImageQwen3RMSNorm k_norm_{nullptr};
};
TORCH_MODULE(JoyImageQwen3Attention);

class JoyImageQwen3DecoderLayerImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3DecoderLayerImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    attention_ = register_module("self_attn", JoyImageQwen3Attention(context));
    input_norm_ =
        register_module("input_layernorm",
                        JoyImageQwen3RMSNorm(args.hidden_size(),
                                             args.rms_norm_eps(),
                                             context.get_tensor_options()));
    post_norm_ =
        register_module("post_attention_layernorm",
                        JoyImageQwen3RMSNorm(args.hidden_size(),
                                             args.rms_norm_eps(),
                                             context.get_tensor_options()));
    mlp_ =
        register_module("mlp",
                        JoyImageQwen3MLP(args.hidden_size(),
                                         args.intermediate_size(),
                                         /*is_gated=*/true,
                                         /*has_bias=*/false,
                                         args.hidden_act(),
                                         context.get_quant_args(),
                                         context.get_parallel_args().tp_group_,
                                         context.get_tensor_options()));
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const AttentionMetadata& attention_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) {
    (void)input_params;
    if (!residual.has_value()) {
      residual = hidden_states;
      hidden_states = std::get<0>(input_norm_->forward(hidden_states));
    } else {
      std::tie(hidden_states, residual) =
          input_norm_->forward(hidden_states, residual);
    }
    hidden_states = attention_->forward(
        positions, hidden_states, attention_metadata, kv_cache);
    std::tie(hidden_states, residual) =
        post_norm_->forward(hidden_states, residual);
    return mlp_->forward(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
    input_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("input_layernorm."));
    post_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("post_attention_layernorm."));
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attention_->verify_loaded_weights(prefix + "self_attn.");
    input_norm_->verify_loaded_weights(prefix + "input_layernorm.");
    post_norm_->verify_loaded_weights(prefix + "post_attention_layernorm.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
  }

 private:
  JoyImageQwen3Attention attention_{nullptr};
  JoyImageQwen3MLP mlp_{nullptr};
  JoyImageQwen3RMSNorm input_norm_{nullptr};
  JoyImageQwen3RMSNorm post_norm_{nullptr};
};
TORCH_MODULE(JoyImageQwen3DecoderLayer);

class JoyImageQwen3VisionAttentionImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3VisionAttentionImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const int64_t hidden_size = args.mm_hidden_size();
    const int64_t total_heads = args.mm_num_attention_heads();
    process_group_ = context.get_parallel_args().tp_group_;
    const int64_t tensor_parallel_size = process_group_->world_size();
    CHECK_EQ(total_heads % tensor_parallel_size, 0);
    num_heads_ = total_heads / tensor_parallel_size;
    head_dim_ = args.mm_head_dim();
    scaling_ = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    qkv_proj_ = register_module("qkv_proj",
                                QKVParallelLinear(hidden_size,
                                                  num_heads_,
                                                  num_heads_,
                                                  head_dim_,
                                                  /*num_kv_head_replicas=*/1,
                                                  /*bias=*/true,
                                                  /*gather_output=*/false,
                                                  context.get_parallel_args(),
                                                  context.get_tensor_options(),
                                                  context.get_quant_args()));
    proj_ = register_module("proj",
                            RowParallelLinear(hidden_size,
                                              hidden_size,
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              context.get_quant_args(),
                                              process_group_,
                                              context.get_tensor_options()));
    proj_bias_ = register_parameter(
        "proj_bias",
        torch::empty({hidden_size}, context.get_tensor_options()),
        /*requires_grad=*/false);
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& cosine,
                        torch::Tensor& sine,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec) {
    (void)cu_seq_len;
    torch::Tensor qkv = qkv_proj_->forward(hidden_states);
    std::vector<torch::Tensor> chunks = qkv.chunk(3, -1);
    const int64_t token_count = hidden_states.size(0);
    torch::Tensor query =
        chunks[0].reshape({token_count, num_heads_, head_dim_});
    torch::Tensor key = chunks[1].reshape({token_count, num_heads_, head_dim_});
    torch::Tensor value =
        chunks[2].reshape({token_count, num_heads_, head_dim_});
    const torch::Tensor query_fp32 = query.to(torch::kFloat32);
    const torch::Tensor key_fp32 = key.to(torch::kFloat32);
    const torch::Tensor cosine_fp32 = cosine.to(torch::kFloat32).unsqueeze(1);
    const torch::Tensor sine_fp32 = sine.to(torch::kFloat32).unsqueeze(1);
    query = (query_fp32 * cosine_fp32 +
             joyimage_qwen3_detail::rotate_half(query_fp32) * sine_fp32)
                .to(query.scalar_type());
    key = (key_fp32 * cosine_fp32 +
           joyimage_qwen3_detail::rotate_half(key_fp32) * sine_fp32)
              .to(key.scalar_type());

    torch::Tensor output = torch::zeros_like(query);
    CHECK_GE(cu_seq_len_vec.size(), 2);
    for (size_t index = 0; index + 1 < cu_seq_len_vec.size(); ++index) {
      const int32_t start = cu_seq_len_vec[index];
      const int32_t end = cu_seq_len_vec[index + 1];
      if (end <= start) {
        continue;
      }
      const torch::Tensor local_query =
          query.slice(0, start, end).permute({1, 0, 2}).unsqueeze(0);
      const torch::Tensor local_key =
          key.slice(0, start, end).permute({1, 0, 2}).unsqueeze(0);
      const torch::Tensor local_value =
          value.slice(0, start, end).permute({1, 0, 2}).unsqueeze(0);
      const torch::Tensor result =
          torch::scaled_dot_product_attention(local_query,
                                              local_key,
                                              local_value,
                                              /*attn_mask=*/std::nullopt,
                                              /*dropout_p=*/0.0,
                                              /*is_causal=*/false,
                                              /*scale=*/scaling_)
              .squeeze(0)
              .permute({1, 0, 2})
              .contiguous();
      output.slice(0, start, end).copy_(result);
    }
    output = output.reshape({token_count, -1});
    return joyimage_qwen3_detail::unsharded_row_matmul(
        output, proj_, process_group_, proj_bias_);
  }

  void load_state_dict(const StateDict& state_dict) {
    qkv_proj_->load_state_dict(state_dict.get_dict_with_prefix("qkv."));
    if (state_dict.get_tensor("qkv.bias").defined()) {
      qkv_bias_is_loaded_ = true;
    }
    const StateDict proj_dict = state_dict.get_dict_with_prefix("proj.");
    proj_->load_state_dict(
        joyimage_qwen3_detail::weight_only_state_dict(proj_dict));
    torch::Tensor bias = proj_dict.get_tensor("bias");
    if (bias.defined()) {
      proj_bias_.data().copy_(bias);
      proj_bias_is_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(qkv_proj_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "qkv.weight";
    CHECK(qkv_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "qkv.bias";
    CHECK(proj_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(proj_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "proj.bias";
  }

 private:
  int64_t num_heads_ = 0;
  int64_t head_dim_ = 0;
  double scaling_ = 0.0;
  ProcessGroup* process_group_ = nullptr;
  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear proj_{nullptr};
  torch::Tensor proj_bias_;
  bool qkv_bias_is_loaded_ = false;
  bool proj_bias_is_loaded_ = false;
};
TORCH_MODULE(JoyImageQwen3VisionAttention);

class JoyImageQwen3VisionLayerImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3VisionLayerImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    attention_ =
        register_module("self_attn", JoyImageQwen3VisionAttention(context));
    norm1_ = register_module("norm1",
                             RMSNorm(args.mm_hidden_size(),
                                     args.rms_norm_eps(),
                                     context.get_tensor_options()));
    norm2_ = register_module("norm2",
                             RMSNorm(args.mm_hidden_size(),
                                     args.rms_norm_eps(),
                                     context.get_tensor_options()));
    norm1_->set_layernorm_mode();
    norm2_->set_layernorm_mode();
    mlp_ =
        register_module("mlp",
                        JoyImageQwen3MLP(args.mm_hidden_size(),
                                         args.mm_intermediate_size(),
                                         /*is_gated=*/false,
                                         /*has_bias=*/true,
                                         args.mm_hidden_act(),
                                         context.get_quant_args(),
                                         context.get_parallel_args().tp_group_,
                                         context.get_tensor_options()));
  }

  torch::Tensor forward(torch::Tensor& hidden_states,
                        torch::Tensor& cosine,
                        torch::Tensor& sine,
                        torch::Tensor& cu_seq_len,
                        std::vector<int32_t>& cu_seq_len_vec) {
    torch::Tensor normalized = std::get<0>(norm1_->forward(hidden_states));
    torch::Tensor output =
        hidden_states +
        attention_->forward(
            normalized, cosine, sine, cu_seq_len, cu_seq_len_vec);
    normalized = std::get<0>(norm2_->forward(output));
    return output + mlp_->forward(normalized);
  }

  void load_state_dict(const StateDict& state_dict) {
    attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."),
                          {"linear_fc1."},
                          "linear_fc2.");
    norm1_->load_state_dict(state_dict.get_dict_with_prefix("norm1."));
    norm2_->load_state_dict(state_dict.get_dict_with_prefix("norm2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attention_->verify_loaded_weights(prefix + "attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    norm1_->verify_loaded_weights(prefix + "norm1.");
    norm2_->verify_loaded_weights(prefix + "norm2.");
  }

 private:
  JoyImageQwen3VisionAttention attention_{nullptr};
  JoyImageQwen3MLP mlp_{nullptr};
  RMSNorm norm1_{nullptr};
  RMSNorm norm2_{nullptr};
};
TORCH_MODULE(JoyImageQwen3VisionLayer);

}  // namespace layer
class JoyImageQwen3ModelImpl final
    : public LlmModelImplBase<layer::JoyImageQwen3DecoderLayer> {
 public:
  explicit JoyImageQwen3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<layer::JoyImageQwen3DecoderLayer>(
            "qwen3",
            context.get_model_args()) {
    // register submodules
    const ModelArgs& model_args = context.get_model_args();
    CHECK_EQ(model_args.head_dim(), 128)
        << "JoyImage Qwen3 text encoder requires head_dim=128";
    const torch::TensorOptions options = context.get_tensor_options();
    if (!mrope_section_.empty()) {
      mrope_dtype_ = options.dtype().toScalarType();
      const torch::TensorOptions cpu_long_options =
          torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
      const torch::Tensor frequency_indices =
          torch::arange(0, 128, 2, cpu_long_options).to(torch::kFloat32);
      mrope_inv_freq_ = register_buffer(
          "mrope_inv_freq",
          (1.0 / torch::pow(model_args.rope_theta(), frequency_indices / 128))
              .to(options.device()));
    }

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    for (int32_t index = 0; index < model_args.n_layers(); ++index) {
      layers_.emplace_back(layer::JoyImageQwen3DecoderLayer(context));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(embed_tokens_->is_weight_loaded())
        << "weight is not loaded for " << prefix + "embed_tokens.weight";
    norm_->verify_loaded_weights(prefix + "norm.");
    for (size_t index = 0; index < layers_.size(); ++index) {
      layers_[index]->verify_loaded_weights(prefix + "layers." +
                                            std::to_string(index) + ".");
    }
  }

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    torch::Tensor position_ids = positions;
    if (position_ids.dim() == 1) {
      position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand({3, 1, -1});
    } else if (position_ids.dim() == 2) {
      position_ids = position_ids.unsqueeze(1);
    }
    CHECK_EQ(position_ids.dim(), 3);
    CHECK_EQ(position_ids.size(0), 3);

    const int64_t batch_size = position_ids.size(1);
    torch::Tensor inv_freq_expanded = mrope_inv_freq_.view({1, 1, -1, 1})
                                          .expand({3, batch_size, -1, 1})
                                          .to(position_ids.device());
    torch::Tensor position_ids_expanded =
        position_ids.unsqueeze(2).to(torch::kFloat32);
    torch::Tensor freqs = torch::matmul(inv_freq_expanded.to(torch::kFloat32),
                                        position_ids_expanded)
                              .transpose(2, 3);
    torch::Tensor interleaved_freqs = freqs[0].clone();
    for (int32_t dimension = 1; dimension <= 2; ++dimension) {
      const int64_t length = mrope_section_[dimension] * 3;
      const torch::indexing::Slice indices(dimension, length, 3);
      interleaved_freqs.index_put_(
          {torch::indexing::Ellipsis, indices},
          freqs[dimension].index({torch::indexing::Ellipsis, indices}));
    }
    const torch::Tensor embedding =
        torch::cat({interleaved_freqs, interleaved_freqs}, -1);
    torch::Tensor cos_pos = embedding.cos().to(mrope_dtype_);
    torch::Tensor sin_pos = embedding.sin().to(mrope_dtype_);
    if (batch_size == 1) {
      cos_pos = cos_pos.squeeze(0);
      sin_pos = sin_pos.squeeze(0);
    }
    return std::make_pair(cos_pos, sin_pos);
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    ModelInputParams input_params_new = input_params;
    const std::vector<torch::Tensor>& deep_stacks =
        input_params.multimodal.deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    const torch::Tensor inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }
    auto& dp_token_nums = input_params_new.parallel.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    if (!input_params_new.attn_metadata) {
      input_params_new.attn_metadata =
          std::make_shared<layer::AttentionMetadata>(
              layer::AttentionMetadataBuilder::build(input_params_new,
                                                     model_args_.enable_mla()));
    }

    auto& attn_metadata = *(input_params_new.attn_metadata);
    const bool only_prefill =
        (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
    if (positions.dim() == 2 && only_prefill && !mrope_section_.empty()) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    std::optional<torch::Tensor> residual;
    for (size_t index = 0; index < layers_.size(); ++index) {
      if (!input_params_new.synchronize_layer(static_cast<uint32_t>(index))) {
        return ModelOutput();
      }
      auto& layer = layers_[index];
      h = layer(h,
                residual,
                positions,
                attn_metadata,
                kv_caches[index],
                input_params_new);
      if (!input_params_new.record_layer(static_cast<uint32_t>(index),
                                         h.device())) {
        return ModelOutput();
      }

      if (index < deep_stacks.size()) {
        if (residual.has_value()) {
          h = h + residual.value();
          residual.reset();
        }
        h = h + deep_stacks[index];
      }
    }
    auto [hidden_states, residual_out] = norm_(h, residual);
    return ModelOutput(hidden_states, residual_out);
  }

 private:
  torch::Tensor mrope_inv_freq_;
  torch::ScalarType mrope_dtype_ = torch::kFloat32;
};
TORCH_MODULE(JoyImageQwen3Model);

class JoyImageQwen3ForCausalLMImpl final
    : public LlmForCausalLMImplBase<JoyImageQwen3Model> {
 public:
  explicit JoyImageQwen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<JoyImageQwen3Model>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    LlmForCausalLMImplBase<JoyImageQwen3Model>::load_model(std::move(loader),
                                                           prefix);
    model_->verify_loaded_weights(prefix);
    if (!embedding_mode_) {
      CHECK(lm_head_->is_weight_loaded()) << "lm_head weight is not loaded";
    }
  }
};
TORCH_MODULE(JoyImageQwen3ForCausalLM);
class JoyImageQwen3VisionPatchEmbedImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3VisionPatchEmbedImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();

    in_channels_ = model_args.mm_num_channels();
    temporal_patch_size_ = model_args.mm_temporal_patch_size();
    patch_size_ = model_args.mm_patch_size();
    embed_dim_ = model_args.mm_hidden_size();
    const std::vector<int64_t> kernel_size = {
        temporal_patch_size_, patch_size_, patch_size_};
    conv_proj_ = register_module(
        "proj",
        torch::nn::Conv3d(
            torch::nn::Conv3dOptions(in_channels_, embed_dim_, kernel_size)
                .stride(kernel_size)
                .bias(true)));
    conv_proj_->weight.set_data(conv_proj_->weight.to(options));
    conv_proj_->bias.set_data(conv_proj_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.view(
        {-1, in_channels_, temporal_patch_size_, patch_size_, patch_size_});
    return conv_proj_(x.to(conv_proj_->weight.scalar_type()))
        .view({-1, embed_dim_});
  }

  void load_state_dict(const StateDict& state_dict) {
    torch::Tensor weight = state_dict.get_tensor("proj.weight");
    if (weight.defined()) {
      CHECK_EQ(conv_proj_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      conv_proj_->weight.data().copy_(weight);
      weight_is_loaded_ = true;
    }
    torch::Tensor bias = state_dict.get_tensor("proj.bias");
    if (bias.defined()) {
      bias = bias.reshape({bias.size(0)});
      CHECK_EQ(conv_proj_->bias.sizes(), bias.sizes())
          << "proj bias size mismatch for " << name();
      conv_proj_->bias.data().copy_(bias);
      bias_is_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "proj.bias";
  }

 private:
  int64_t in_channels_ = 0;
  int64_t temporal_patch_size_ = 0;
  int64_t patch_size_ = 0;
  int64_t embed_dim_ = 0;
  torch::nn::Conv3d conv_proj_{nullptr};
  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;
};
TORCH_MODULE(JoyImageQwen3VisionPatchEmbed);

class JoyImageQwen3VisionRotaryEmbeddingImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3VisionRotaryEmbeddingImpl(const ModelContext& context) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();

    const int64_t rotary_dim = model_args.mm_head_dim() / 2;

    const torch::TensorOptions cpu_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    const torch::Tensor reference_inv_freq =
        1.0 /
        torch::pow(/*base=*/10000.0,
                   torch::arange(0, rotary_dim, 2, cpu_options) / rotary_dim);
    reference_inv_freq_ = register_buffer(
        "reference_inv_freq", reference_inv_freq.to(options.device()));
  }

  torch::Tensor forward_positions(const torch::Tensor& position_ids) {
    return (position_ids.unsqueeze(-1) * reference_inv_freq_).flatten(1);
  }

 private:
  torch::Tensor reference_inv_freq_;
};
TORCH_MODULE(JoyImageQwen3VisionRotaryEmbedding);

class JoyImageQwen3VisionPatchMergerImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3VisionPatchMergerImpl(
      const ModelContext& context,
      bool use_postshuffle_norm = false) {
    const ModelArgs& model_args = context.get_model_args();
    const torch::TensorOptions options = context.get_tensor_options();
    const int64_t d_model = model_args.mm_projection_dim();
    const int64_t context_dim = model_args.mm_hidden_size();
    const int64_t spatial_merge_size = model_args.mm_spatial_merge_size();
    hidden_size_ = context_dim * spatial_merge_size * spatial_merge_size;
    use_postshuffle_norm_ = use_postshuffle_norm;
    if (use_postshuffle_norm) {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    } else {
      norm_ = register_module(
          "norm",
          torch::nn::LayerNorm(torch::nn::LayerNormOptions({context_dim})
                                   .elementwise_affine(true)
                                   .eps(1e-6)));
    }
    norm_->weight.set_data(norm_->weight.to(options));
    norm_->bias.set_data(norm_->bias.to(options));

    fc1_ = register_module(
        "linear_fc1",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true)));
    activation_ = register_module("activation", torch::nn::GELU());
    fc2_ = register_module(
        "linear_fc2",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, d_model).bias(true)));
    fc1_->weight.set_data(fc1_->weight.to(options));
    fc1_->bias.set_data(fc1_->bias.to(options));
    fc2_->weight.set_data(fc2_->weight.to(options));
    fc2_->bias.set_data(fc2_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor x) {
    if (use_postshuffle_norm_) {
      x = norm_(x.view({-1, hidden_size_}));
    } else {
      x = norm_(x).view({-1, hidden_size_});
    }
    return fc2_->forward(activation_->forward(fc1_->forward(x)));
  }

  void load_state_dict(const StateDict& state_dict) {
    // norm
    const auto& norm_dict = state_dict.get_dict_with_prefix("norm.");
    const auto& norm_weight = norm_dict.get_tensor("weight");
    if (norm_weight.defined()) {
      CHECK_EQ(norm_->weight.sizes(), norm_weight.sizes())
          << "weight size mismatch for " << name();
      norm_->weight.data().copy_(norm_weight);
      norm_weight_is_loaded_ = true;
    }
    const auto norm_bias = norm_dict.get_tensor("bias");
    if (norm_bias.defined()) {
      CHECK_EQ(norm_->bias.sizes(), norm_bias.sizes())
          << "bias size mismatch for " << name();
      norm_->bias.data().copy_(norm_bias);
      norm_bias_is_loaded_ = true;
    }

    const auto& fc1_dict = state_dict.get_dict_with_prefix("linear_fc1.");
    const auto& fc1_weight = fc1_dict.get_tensor("weight");
    if (fc1_weight.defined()) {
      CHECK_EQ(fc1_->weight.sizes(), fc1_weight.sizes())
          << "weight size mismatch for " << name();
      fc1_->weight.data().copy_(fc1_weight);
      fc1_weight_is_loaded_ = true;
    }
    const auto fc1_bias = fc1_dict.get_tensor("bias");
    if (fc1_bias.defined()) {
      CHECK_EQ(fc1_->bias.sizes(), fc1_bias.sizes())
          << "bias size mismatch for " << name();
      fc1_->bias.data().copy_(fc1_bias);
      fc1_bias_is_loaded_ = true;
    }

    const auto& fc2_dict = state_dict.get_dict_with_prefix("linear_fc2.");
    const auto& fc2_weight = fc2_dict.get_tensor("weight");
    if (fc2_weight.defined()) {
      CHECK_EQ(fc2_->weight.sizes(), fc2_weight.sizes())
          << "weight size mismatch for " << name();
      fc2_->weight.data().copy_(fc2_weight);
      fc2_weight_is_loaded_ = true;
    }
    const auto fc2_bias = fc2_dict.get_tensor("bias");
    if (fc2_bias.defined()) {
      CHECK_EQ(fc2_->bias.sizes(), fc2_bias.sizes())
          << "bias size mismatch for " << name();
      fc2_->bias.data().copy_(fc2_bias);
      fc2_bias_is_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(norm_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "norm.weight";
    CHECK(norm_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "norm.bias";
    CHECK(fc1_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "linear_fc1.weight";
    CHECK(fc1_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "linear_fc1.bias";
    CHECK(fc2_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "linear_fc2.weight";
    CHECK(fc2_bias_is_loaded_)
        << "bias is not loaded for " << prefix + "linear_fc2.bias";
  }

 private:
  int64_t hidden_size_ = 0;
  bool use_postshuffle_norm_;
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear fc1_{nullptr};
  torch::nn::GELU activation_{nullptr};
  torch::nn::Linear fc2_{nullptr};
  bool norm_weight_is_loaded_ = false;
  bool norm_bias_is_loaded_ = false;
  bool fc1_weight_is_loaded_ = false;
  bool fc1_bias_is_loaded_ = false;
  bool fc2_weight_is_loaded_ = false;
  bool fc2_bias_is_loaded_ = false;
};
TORCH_MODULE(JoyImageQwen3VisionPatchMerger);

class JoyImageQwen3VisionTransformerImpl final : public torch::nn::Module {
 public:
  explicit JoyImageQwen3VisionTransformerImpl(const ModelContext& context)
      : options_(context.get_tensor_options()) {
    const ModelArgs& model_args = context.get_model_args();
    const int64_t hidden_size = model_args.mm_hidden_size();
    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    const std::vector<int64_t>& visual_indexes =
        model_args.mm_deepstack_visual_indexes();
    deepstack_visual_indexes_.insert(deepstack_visual_indexes_.end(),
                                     visual_indexes.begin(),
                                     visual_indexes.end());
    const int64_t num_position_embeddings =
        model_args.mm_num_position_embeddings();
    num_grid_per_side_ = static_cast<int64_t>(
        std::sqrt(static_cast<double>(num_position_embeddings)));

    patch_embed_ =
        register_module("patch_embed", JoyImageQwen3VisionPatchEmbed(context));
    rotary_pos_emb_ = register_module(
        "rotary_pos_emb", JoyImageQwen3VisionRotaryEmbedding(context));

    deepstack_mergers_ =
        register_module("deepstack_mergers", torch::nn::ModuleList());

    emb_ = register_module(
        "embedding",
        torch::nn::Embedding(num_position_embeddings, hidden_size));
    emb_->weight.set_data(emb_->weight.to(options_));

    merger_ =
        register_module("merger", JoyImageQwen3VisionPatchMerger(context));

    layers_.reserve(model_args.mm_num_hidden_layers());
    for (int32_t index = 0; index < model_args.mm_num_hidden_layers();
         ++index) {
      layers_.emplace_back(layer::JoyImageQwen3VisionLayer(context));
    }
    deepstack_merger_layers_.reserve(deepstack_visual_indexes_.size());
    for (size_t index = 0; index < deepstack_visual_indexes_.size(); ++index) {
      JoyImageQwen3VisionPatchMerger merger =
          JoyImageQwen3VisionPatchMerger(context,
                                         /*use_postshuffle_norm=*/true);
      deepstack_mergers_->push_back(merger);
      deepstack_merger_layers_.emplace_back(merger);
    }
  }

  torch::Tensor rot_pos_emb(const torch::Tensor& grid_thw_cpu,
                            const torch::Device& device) {
    std::vector<torch::Tensor> pos_ids_vec;
    const int64_t count = grid_thw_cpu.size(0);
    pos_ids_vec.reserve(count);

    const torch::TensorOptions options =
        torch::TensorOptions().dtype(torch::kLong).device(device);

    for (int64_t idx = 0; idx < count; ++idx) {
      const int64_t t = grid_thw_cpu[idx][0].item<int64_t>();
      const int64_t h = grid_thw_cpu[idx][1].item<int64_t>();
      const int64_t w = grid_thw_cpu[idx][2].item<int64_t>();

      torch::Tensor hpos_ids =
          torch::arange(h, options).unsqueeze(1).expand({-1, w});
      hpos_ids = hpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      torch::Tensor wpos_ids =
          torch::arange(w, options).unsqueeze(0).expand({h, -1});
      wpos_ids = wpos_ids
                     .reshape({h / spatial_merge_size_,
                               spatial_merge_size_,
                               w / spatial_merge_size_,
                               spatial_merge_size_})
                     .permute({0, 2, 1, 3})
                     .flatten();

      pos_ids_vec.emplace_back(
          torch::stack({hpos_ids, wpos_ids}, -1).repeat({t, 1}));
    }

    const torch::Tensor pos_ids = torch::cat(pos_ids_vec, 0);
    return rotary_pos_emb_->forward_positions(pos_ids);
  }

  torch::Tensor fast_pos_embed_interpolate(const torch::Tensor& grid_cpu,
                                           const torch::Device& device) {
    const int64_t count = grid_cpu.size(0);
    CHECK_GT(count, 0) << "Qwen3-VL grid_thw must not be empty";
    std::array<std::vector<torch::Tensor>, 4> index_parts;
    std::array<std::vector<torch::Tensor>, 4> weight_parts;
    for (int32_t corner = 0; corner < 4; ++corner) {
      index_parts[corner].reserve(count);
      weight_parts[corner].reserve(count);
    }

    for (int64_t index = 0; index < count; ++index) {
      const int64_t t = grid_cpu[index][0].item<int64_t>();
      const int64_t h = grid_cpu[index][1].item<int64_t>();
      const int64_t w = grid_cpu[index][2].item<int64_t>();
      CHECK_GT(t, 0);
      CHECK_GT(h, 0);
      CHECK_GT(w, 0);
      CHECK_EQ(h % spatial_merge_size_, 0);
      CHECK_EQ(w % spatial_merge_size_, 0);
      const torch::TensorOptions options =
          torch::TensorOptions().dtype(torch::kFloat32).device(device);
      const torch::Tensor h_grid = torch::linspace(
          0, static_cast<double>(num_grid_per_side_ - 1), h, options);
      const torch::Tensor w_grid = torch::linspace(
          0, static_cast<double>(num_grid_per_side_ - 1), w, options);
      const torch::Tensor h_floor = h_grid.to(torch::kInt32);
      const torch::Tensor w_floor = w_grid.to(torch::kInt32);
      const torch::Tensor h_ceil =
          (h_floor + 1).clamp_max(num_grid_per_side_ - 1);
      const torch::Tensor w_ceil =
          (w_floor + 1).clamp_max(num_grid_per_side_ - 1);
      const torch::Tensor h_frac = h_grid - h_floor;
      const torch::Tensor w_frac = w_grid - w_floor;
      const torch::Tensor h_floor_offset = h_floor * num_grid_per_side_;
      const torch::Tensor h_ceil_offset = h_ceil * num_grid_per_side_;

      const std::array<torch::Tensor, 4> corner_indices = {
          (h_floor_offset.unsqueeze(1) + w_floor.unsqueeze(0)).flatten(),
          (h_floor_offset.unsqueeze(1) + w_ceil.unsqueeze(0)).flatten(),
          (h_ceil_offset.unsqueeze(1) + w_floor.unsqueeze(0)).flatten(),
          (h_ceil_offset.unsqueeze(1) + w_ceil.unsqueeze(0)).flatten()};
      const std::array<torch::Tensor, 4> corner_weights = {
          ((1 - h_frac).unsqueeze(1) * (1 - w_frac).unsqueeze(0)).flatten(),
          ((1 - h_frac).unsqueeze(1) * w_frac.unsqueeze(0)).flatten(),
          (h_frac.unsqueeze(1) * (1 - w_frac).unsqueeze(0)).flatten(),
          (h_frac.unsqueeze(1) * w_frac.unsqueeze(0)).flatten()};

      const torch::Tensor h_index =
          torch::arange(h, torch::TensorOptions().device(device))
              .view({h / spatial_merge_size_, spatial_merge_size_});
      const torch::Tensor w_index =
          torch::arange(w, torch::TensorOptions().device(device))
              .view({w / spatial_merge_size_, spatial_merge_size_});
      const torch::Tensor reorder = (h_index.unsqueeze(2).unsqueeze(3) * w +
                                     w_index.unsqueeze(0).unsqueeze(0))
                                        .transpose(1, 2)
                                        .flatten()
                                        .repeat({t});

      for (int32_t corner = 0; corner < 4; ++corner) {
        index_parts[corner].emplace_back(
            corner_indices[corner].index({reorder}));
        weight_parts[corner].emplace_back(
            corner_weights[corner].index({reorder}));
      }
    }

    std::vector<torch::Tensor> indices;
    std::vector<torch::Tensor> weights;
    indices.reserve(4);
    weights.reserve(4);
    for (int32_t corner = 0; corner < 4; ++corner) {
      indices.emplace_back(torch::cat(index_parts[corner]));
      weights.emplace_back(torch::cat(weight_parts[corner]));
    }
    const torch::Tensor bilinear_indices = torch::stack(indices);
    const torch::Tensor bilinear_weights = torch::stack(weights);
    return (emb_(bilinear_indices) * bilinear_weights.unsqueeze(-1))
        .sum(0)
        .to(options_.dtype());
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor grid_thw) {  // [batch,thw]
    hidden_states = patch_embed_(hidden_states);
    const torch::Tensor grid_thw_cpu = grid_thw.cpu();
    const torch::Tensor pos_embeds =
        fast_pos_embed_interpolate(grid_thw_cpu, grid_thw.device());
    hidden_states = hidden_states + pos_embeds;
    //   compute position embedding
    const torch::Tensor rotary_pos_emb =
        rot_pos_emb(grid_thw_cpu, grid_thw.device());
    // compute cu_seqlens
    torch::Tensor cu_seqlens =
        torch::repeat_interleave(
            grid_thw.index({torch::indexing::Slice(), 1}) *
                grid_thw.index({torch::indexing::Slice(), 2}),
            grid_thw.index({torch::indexing::Slice(), 0}))
            .cumsum(0, torch::kInt32);
    namespace F = torch::nn::functional;
    cu_seqlens = F::pad(
        cu_seqlens, F::PadFuncOptions({1, 0}).mode(torch::kConstant).value(0));

    torch::Tensor cosine = rotary_pos_emb.cos().repeat({1, 2});
    torch::Tensor sine = rotary_pos_emb.sin().repeat({1, 2});

    torch::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    std::vector<int32_t> cu_seqlens_vec(
        cu_seqlens_cpu.data_ptr<int32_t>(),  // full seqlen vec
        cu_seqlens_cpu.data_ptr<int32_t>() + cu_seqlens_cpu.numel());
    std::vector<torch::Tensor> deepstack_feature_lists;
    deepstack_feature_lists.reserve(deepstack_visual_indexes_.size());
    for (size_t idx = 0; idx < layers_.size(); ++idx) {
      hidden_states =
          layers_[idx](hidden_states, cosine, sine, cu_seqlens, cu_seqlens_vec);
      auto it = std::find(deepstack_visual_indexes_.begin(),
                          deepstack_visual_indexes_.end(),
                          idx);

      if (it != deepstack_visual_indexes_.end()) {
        const size_t index = static_cast<size_t>(
            std::distance(deepstack_visual_indexes_.begin(), it));
        deepstack_feature_lists.emplace_back(
            deepstack_merger_layers_[index](hidden_states));
      }
    }
    // adapter
    hidden_states = merger_(hidden_states);
    std::vector<torch::Tensor> tensors;
    tensors.reserve(deepstack_feature_lists.size() + 1);
    tensors.emplace_back(hidden_states);
    tensors.insert(tensors.end(),
                   deepstack_feature_lists.begin(),
                   deepstack_feature_lists.end());
    return torch::cat(tensors, /*dim=*/1);
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));
    for (size_t idx = 0; idx < layers_.size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    merger_->load_state_dict(state_dict.get_dict_with_prefix("merger."));

    for (size_t idx = 0; idx < deepstack_merger_layers_.size(); ++idx) {
      deepstack_merger_layers_[idx]->load_state_dict(
          state_dict.get_dict_with_prefix("deepstack_merger_list." +
                                          std::to_string(idx) + "."));
    }

    const auto& emb_dict = state_dict.get_dict_with_prefix("pos_embed.");
    const auto& emb_weight = emb_dict.get_tensor("weight");
    if (emb_weight.defined()) {
      CHECK_EQ(emb_->weight.sizes(), emb_weight.sizes())
          << "weight size mismatch for " << name();
      emb_->weight.data().copy_(emb_weight);
      pos_embed_is_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    for (size_t index = 0; index < layers_.size(); ++index) {
      layers_[index]->verify_loaded_weights(prefix + "blocks." +
                                            std::to_string(index) + ".");
    }
    merger_->verify_loaded_weights(prefix + "merger.");
    for (size_t index = 0; index < deepstack_merger_layers_.size(); ++index) {
      deepstack_merger_layers_[index]->verify_loaded_weights(
          prefix + "deepstack_merger_list." + std::to_string(index) + ".");
    }
    CHECK(pos_embed_is_loaded_)
        << "weight is not loaded for " << prefix + "pos_embed.weight";
  }

 private:
  int64_t spatial_merge_size_ = 0;
  std::vector<int64_t> deepstack_visual_indexes_;
  int64_t num_grid_per_side_ = 0;
  bool pos_embed_is_loaded_ = false;

  JoyImageQwen3VisionPatchEmbed patch_embed_{nullptr};
  JoyImageQwen3VisionRotaryEmbedding rotary_pos_emb_{nullptr};
  torch::nn::Embedding emb_{nullptr};

  std::vector<layer::JoyImageQwen3VisionLayer> layers_;

  torch::nn::ModuleList deepstack_mergers_{nullptr};
  std::vector<JoyImageQwen3VisionPatchMerger> deepstack_merger_layers_;
  JoyImageQwen3VisionPatchMerger merger_{nullptr};

  torch::TensorOptions options_;
};
TORCH_MODULE(JoyImageQwen3VisionTransformer);

class JoyImageQwen3VLForConditionalGenerationImpl final
    : public Qwen3VLForConditionalGenerationBase<JoyImageQwen3VisionTransformer,
                                                 JoyImageQwen3ForCausalLM> {
 public:
  explicit JoyImageQwen3VLForConditionalGenerationImpl(
      const ModelContext& context)
      : Qwen3VLForConditionalGenerationBase(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader) {
    Qwen3VLForConditionalGenerationBase::load_model(std::move(loader));
    visual_->verify_loaded_weights("model.visual.");
  }
};
TORCH_MODULE(JoyImageQwen3VLForConditionalGeneration);

}  // namespace xllm
