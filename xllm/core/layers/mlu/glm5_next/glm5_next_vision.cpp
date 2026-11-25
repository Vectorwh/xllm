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

#include "core/layers/mlu/glm5_next/glm5_next_vision.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/kernels/mlu/mlu_ops_api.h"

namespace xllm::layer {

class Glm5NextVisionNormImpl final : public torch::nn::Module {
 public:
  Glm5NextVisionNormImpl(int64_t dim, double eps) : eps_(eps) {
    weight_ = register_parameter("weight", torch::ones({dim}));
  }

  // If present, residual receives the rounded sum for the skip connection.
  torch::Tensor forward(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& residual = std::nullopt) {
    const torch::Tensor x = input.reshape({-1, input.size(-1)}).contiguous();
    torch::Tensor normalized = torch::empty_like(x);
    kernel::mlu::fused_layernorm(
        x,
        normalized,
        residual,
        weight_,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        residual,
        std::nullopt,
        std::nullopt,
        "rmsnorm",
        eps_,
        /*store_output_before_norm=*/residual.has_value(),
        /*store_output_after_norm=*/false,
        /*dynamic_quant=*/false);
    return normalized.view(input.sizes());
  }

 private:
  double eps_;
  torch::Tensor weight_;
};

namespace {

torch::Tensor activate(const torch::Tensor& input, const std::string& act) {
  if (act == "silu") {
    return torch::silu(input);
  }
  if (act == "gelu_pytorch_tanh") {
    return torch::gelu(input, "tanh");
  }
  CHECK_EQ(act, "gelu");
  return torch::gelu(input);
}

class Glm5NextVisionMlpImpl final : public torch::nn::Module {
 public:
  Glm5NextVisionMlpImpl(int64_t dim,
                        int64_t intermediate,
                        bool bias,
                        std::string act,
                        double limit)
      : act_(std::move(act)), limit_(limit) {
    gate_ = register_module(
        "gate_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, intermediate).bias(bias)));
    up_ = register_module(
        "up_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, intermediate).bias(bias)));
    down_ = register_module(
        "down_proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(intermediate, dim).bias(bias)));
  }

  torch::Tensor forward(const torch::Tensor& input) {
    const auto projected = torch::linear(input, gate_up_weight_, gate_up_bias_)
                               .chunk(/*chunks=*/2, /*dim=*/-1);
    return down_(activate(torch::clamp_max(projected[0], limit_), act_) *
                 torch::clamp(projected[1], -limit_, limit_));
  }

  void pack_weights() {
    torch::NoGradGuard no_grad;
    // Parameter views preserve checkpoint names and accept sharded loads
    // directly into the packed storage, without concatenating each forward.
    gate_up_weight_ = torch::cat({gate_->weight, up_->weight}, /*dim=*/0);
    const int64_t width = gate_->weight.size(0);
    gate_->weight.set_data(
        gate_up_weight_.narrow(/*dim=*/0, /*start=*/0, width));
    up_->weight.set_data(gate_up_weight_.narrow(/*dim=*/0, width, width));
    if (gate_->bias.defined()) {
      gate_up_bias_ = torch::cat({gate_->bias, up_->bias}, /*dim=*/0);
      gate_->bias.set_data(gate_up_bias_.narrow(/*dim=*/0, /*start=*/0, width));
      up_->bias.set_data(gate_up_bias_.narrow(/*dim=*/0, width, width));
    }
  }

 private:
  std::string act_;
  double limit_;
  torch::nn::Linear gate_{nullptr}, up_{nullptr}, down_{nullptr};
  torch::Tensor gate_up_weight_, gate_up_bias_;
};

class Glm5NextVisionAttentionImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionAttentionImpl(const ModelArgs& args)
      : heads_(args.mm_num_attention_heads()),
        head_dim_(args.mm_hidden_size() / heads_) {
    const int64_t dim = args.mm_hidden_size();
    qkv_ =
        register_module("qkv",
                        torch::nn::Linear(torch::nn::LinearOptions(dim, 3 * dim)
                                              .bias(args.mm_attention_bias())));
    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(dim, dim).bias(args.mm_attention_bias())));
    q_norm_ = register_module("q_norm",
                              std::make_shared<Glm5NextVisionNormImpl>(
                                  head_dim_, args.mm_layer_norm_eps()));
    k_norm_ = register_module("k_norm",
                              std::make_shared<Glm5NextVisionNormImpl>(
                                  head_dim_, args.mm_layer_norm_eps()));
  }

  torch::Tensor forward(const torch::Tensor& input,
                        const Glm5NextVisionMetadata& metadata) {
    const auto qkv = qkv_(input)
                         .view({input.size(0), 3, heads_, head_dim_})
                         .unbind(/*dim=*/1);
    auto rotate = [&](const torch::Tensor& x) {
      const torch::Tensor fp32 = x.to(torch::kFloat32);
      // Expanded axial caches are indexed by the full flattened token axis.
      // Use BSHD mode so RoPE does not reset its cache index at frame bounds.
      torch::Tensor rotated = fp32.unsqueeze(/*dim=*/0).contiguous();
      torch::Tensor unused_key;
      kernel::mlu::apply_rotary(rotated,
                                unused_key,
                                metadata.sin.squeeze(/*dim=*/1),
                                metadata.cos.squeeze(/*dim=*/1),
                                std::nullopt,
                                std::nullopt,
                                /*interleaved=*/false,
                                /*discrete=*/false,
                                /*dynamic_ntk=*/false,
                                input.size(0));
      return rotated.squeeze(/*dim=*/0).to(x.scalar_type());
    };
    const torch::Tensor q = rotate(q_norm_->forward(qkv[0]));
    const torch::Tensor k = rotate(k_norm_->forward(qkv[1]));
    torch::Tensor output = torch::empty_like(q);
    std::optional<torch::Tensor> lse;
    kernel::mlu::batch_prefill(
        q,
        k,
        qkv[2].contiguous(),
        output,
        lse,
        metadata.cu_seq_lens,
        metadata.cu_seq_lens,
        /*alibi_slope=*/std::nullopt,
        /*attn_bias=*/std::nullopt,
        /*q_quant_scale=*/std::nullopt,
        /*k_quant_scale=*/std::nullopt,
        /*v_quant_scale=*/std::nullopt,
        /*out_quant_scale=*/std::nullopt,
        /*block_table=*/std::nullopt,
        metadata.max_seq_len,
        metadata.max_seq_len,
        /*scale=*/1.0f / std::sqrt(static_cast<float>(head_dim_)),
        /*is_causal=*/false,
        /*window_size_left=*/-1,
        /*window_size_right=*/-1,
        /*compute_dtype=*/"float",
        /*return_lse=*/false);
    return proj_(output.reshape({input.size(0), -1}));
  }

 private:
  int64_t heads_;
  int64_t head_dim_;
  torch::nn::Linear qkv_{nullptr}, proj_{nullptr};
  std::shared_ptr<Glm5NextVisionNormImpl> q_norm_, k_norm_;
};

class Glm5NextVisionBlockImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionBlockImpl(const ModelArgs& args) {
    norm1_ =
        register_module("norm1",
                        std::make_shared<Glm5NextVisionNormImpl>(
                            args.mm_hidden_size(), args.mm_layer_norm_eps()));
    norm2_ =
        register_module("norm2",
                        std::make_shared<Glm5NextVisionNormImpl>(
                            args.mm_hidden_size(), args.mm_layer_norm_eps()));
    attn_ = register_module(
        "attn", std::make_shared<Glm5NextVisionAttentionImpl>(args));
    mlp_ = register_module(
        "mlp",
        std::make_shared<Glm5NextVisionMlpImpl>(args.mm_hidden_size(),
                                                args.mm_intermediate_size(),
                                                args.mm_attention_bias(),
                                                args.mm_hidden_act(),
                                                args.mm_swiglu_limit()));
  }

  torch::Tensor forward(torch::Tensor x,
                        const Glm5NextVisionMetadata& metadata) {
    torch::Tensor residual = attn_->forward(norm1_->forward(x), metadata);
    const torch::Tensor normalized = norm2_->forward(x, residual);
    return residual + mlp_->forward(normalized);
  }

  void pack_weights() { mlp_->pack_weights(); }

 private:
  std::shared_ptr<Glm5NextVisionNormImpl> norm1_, norm2_;
  std::shared_ptr<Glm5NextVisionAttentionImpl> attn_;
  std::shared_ptr<Glm5NextVisionMlpImpl> mlp_;
};

}  // namespace

class Glm5NextVisionMergerImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionMergerImpl(const ModelArgs& args) {
    const int64_t dim = args.mm_projection_dim();
    proj_ = register_module(
        "proj",
        torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
    norm_ = register_module(
        "post_projection_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    // Register gate/up/down directly to match checkpoint merger.* names.
    mlp_ =
        std::make_shared<Glm5NextVisionMlpImpl>(dim,
                                                args.mm_projector_hidden_size(),
                                                /*bias=*/false,
                                                args.mm_hidden_act(),
                                                args.mm_swiglu_limit());
    for (const auto& child : mlp_->named_children()) {
      register_module(child.key(), child.value());
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return mlp_->forward(torch::gelu(norm_(proj_(x))));
  }

  void pack_weights() { mlp_->pack_weights(); }

 private:
  torch::nn::Linear proj_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::shared_ptr<Glm5NextVisionMlpImpl> mlp_;
};

Glm5NextVisionImpl::Glm5NextVisionImpl(const ModelContext& context)
    : args_(context.get_model_args()), options_(context.get_tensor_options()) {
  CHECK_GT(args_.mm_num_attention_heads(), 0);
  CHECK_EQ(args_.mm_hidden_size() % args_.mm_num_attention_heads(), 0);
  const int64_t head_dim =
      args_.mm_hidden_size() / args_.mm_num_attention_heads();
  CHECK_EQ(head_dim % 4, 0);
  CHECK_GT(args_.mm_spatial_merge_size(), 0);
  CHECK_GT(args_.mm_patch_size(), 0);
  CHECK_GT(args_.mm_temporal_patch_size(), 0);
  CHECK_GT(args_.mm_projector_hidden_size(), 0);
  CHECK_GT(args_.mm_swiglu_limit(), 0);
  CHECK(args_.mm_hidden_act() == "silu" || args_.mm_hidden_act() == "gelu" ||
        args_.mm_hidden_act() == "gelu_pytorch_tanh");
  const int64_t patch = args_.mm_patch_size();
  const int64_t temporal = args_.mm_temporal_patch_size();
  auto patch_embed =
      register_module("patch_embed", std::make_shared<torch::nn::Module>());
  patch_projection_ = patch_embed->register_module(
      "proj",
      torch::nn::Conv3d(torch::nn::Conv3dOptions(args_.mm_num_channels(),
                                                 args_.mm_hidden_size(),
                                                 {temporal, patch, patch})
                            .stride({temporal, patch, patch})
                            .bias(true)));
  blocks_ = register_module("blocks", torch::nn::ModuleList());
  for (int64_t i = 0; i < args_.mm_num_hidden_layers(); ++i) {
    blocks_->push_back(std::make_shared<Glm5NextVisionBlockImpl>(args_));
  }
  post_layernorm_ =
      register_module("post_layernorm",
                      std::make_shared<Glm5NextVisionNormImpl>(
                          args_.mm_hidden_size(), args_.mm_layer_norm_eps()));
  downsample_ = register_module(
      "downsample",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(args_.mm_hidden_size(),
                                                 args_.mm_projection_dim(),
                                                 args_.mm_spatial_merge_size())
                            .stride(args_.mm_spatial_merge_size())
                            .bias(true)));
  merger_ = register_module("merger",
                            std::make_shared<Glm5NextVisionMergerImpl>(args_));
  const torch::Tensor exponents =
      torch::arange(0, head_dim / 2, 2, torch::kFloat32) /
      static_cast<double>(head_dim / 2);
  inv_freq_ =
      register_buffer("inv_freq", (1.0 / torch::pow(10000.0, exponents)));
  to(options_.device(), options_.dtype().toScalarType());
  // RoPE frequencies must retain float32 precision after casting model weights.
  inv_freq_.set_data(
      (1.0 / torch::pow(10000.0, exponents)).to(options_.device()));
  for (const auto& block : *blocks_) {
    block->as<Glm5NextVisionBlockImpl>()->pack_weights();
  }
  merger_->pack_weights();
}

Glm5NextVisionMetadata Glm5NextVisionImpl::prepare_metadata(
    const torch::Tensor& grid_thw) const {
  torch::NoGradGuard no_grad;
  CHECK_EQ(grid_thw.dim(), 2);
  CHECK_EQ(grid_thw.size(1), 3);
  CHECK_GT(grid_thw.size(0), 0);
  const torch::Tensor grid =
      grid_thw.to(torch::kCPU, torch::kInt64).contiguous();
  const auto sizes = grid.accessor<int64_t, 2>();
  const int64_t merge = args_.mm_spatial_merge_size();
  std::vector<torch::Tensor> positions;
  Glm5NextVisionMetadata metadata;
  std::vector<int32_t> offsets{0};
  positions.reserve(grid.size(0));
  offsets.reserve(grid.sum().item<int64_t>() + 1);
  metadata.output_sizes.reserve(grid.size(0));
  const auto index_options = torch::TensorOptions().dtype(torch::kInt64);
  for (int64_t i = 0; i < grid.size(0); ++i) {
    const int64_t t = sizes[i][0];
    const int64_t h = sizes[i][1];
    const int64_t w = sizes[i][2];
    CHECK_GT(t, 0);
    CHECK_GT(h, 0);
    CHECK_GT(w, 0);
    CHECK_EQ(h % merge, 0);
    CHECK_EQ(w % merge, 0);
    CHECK_LE(h, std::numeric_limits<int32_t>::max() / w);
    const int64_t patches_per_frame = h * w;
    CHECK_LE(t,
             (std::numeric_limits<int32_t>::max() - metadata.num_patches) /
                 patches_per_frame);
    const torch::Tensor rows =
        torch::arange(h, index_options).unsqueeze(1).expand({h, w});
    const torch::Tensor cols =
        torch::arange(w, index_options).unsqueeze(0).expand({h, w});
    auto reorder = [&](const torch::Tensor& x) {
      return x.reshape({h / merge, merge, w / merge, merge})
          .transpose(1, 2)
          .flatten();
    };
    positions.emplace_back(
        torch::stack({reorder(rows), reorder(cols)}, -1).repeat({t, 1}));
    metadata.num_patches += t * patches_per_frame;
    metadata.max_seq_len = std::max(metadata.max_seq_len, patches_per_frame);
    metadata.output_sizes.emplace_back(t * patches_per_frame / (merge * merge));
    for (int64_t frame = 0; frame < t; ++frame) {
      offsets.emplace_back(
          static_cast<int32_t>(offsets.back() + patches_per_frame));
    }
  }
  const torch::Tensor position_ids =
      torch::cat(positions, 0).to(options_.device());
  const torch::Tensor frequencies =
      (position_ids.unsqueeze(-1) * inv_freq_).flatten(1);
  const torch::Tensor angles =
      torch::cat({frequencies, frequencies}, -1).unsqueeze(1);
  metadata.cos = angles.cos();
  metadata.sin = angles.sin();
  metadata.cu_seq_lens =
      torch::tensor(offsets, torch::kInt32).to(options_.device());
  return metadata;
}

torch::Tensor Glm5NextVisionImpl::forward(const torch::Tensor& pixels,
                                          const torch::Tensor& grid_thw) {
  return forward_prepared(pixels, prepare_metadata(grid_thw));
}

torch::Tensor Glm5NextVisionImpl::forward_prepared(
    const torch::Tensor& pixels,
    const Glm5NextVisionMetadata& metadata) {
  torch::NoGradGuard no_grad;
  CHECK_EQ(pixels.dim(), 2);
  CHECK_EQ(pixels.size(0), metadata.num_patches);
  CHECK_EQ(pixels.size(1),
           args_.mm_num_channels() * args_.mm_temporal_patch_size() *
               args_.mm_patch_size() * args_.mm_patch_size());
  CHECK_EQ(metadata.cos.size(0), metadata.num_patches);
  CHECK_EQ(metadata.cos.size(-1),
           args_.mm_hidden_size() / args_.mm_num_attention_heads());
  CHECK_EQ(metadata.sin.sizes(), metadata.cos.sizes());
  const int64_t merge = args_.mm_spatial_merge_size();
  // These convolutions consume one non-overlapping patch per output position.
  torch::Tensor x = torch::linear(pixels.to(options_),
                                  patch_projection_->weight.flatten(1),
                                  patch_projection_->bias);
  for (const auto& block : *blocks_) {
    x = block->as<Glm5NextVisionBlockImpl>()->forward(x, metadata);
  }
  x = post_layernorm_->forward(x);
  x = x.view({-1, merge, merge, args_.mm_hidden_size()}).permute({0, 3, 1, 2});
  x = torch::linear(x.reshape({-1, args_.mm_hidden_size() * merge * merge}),
                    downsample_->weight.flatten(1),
                    downsample_->bias);
  return merger_->forward(x);
}

void Glm5NextVisionImpl::load_state_dict(const StateDict& state_dict) {
  torch::NoGradGuard no_grad;
  for (const auto& parameter : named_parameters()) {
    const torch::Tensor weight = state_dict.get_tensor(parameter.key());
    if (!weight.defined()) {
      continue;
    }
    CHECK(weight.is_floating_point())
        << "Vision weight must be unquantized: " << parameter.key();
    CHECK(weight.sizes() == parameter.value().sizes())
        << "Vision weight shape mismatch: " << parameter.key();
    parameter.value().copy_(weight);
    loaded_weights_.insert(parameter.key());
  }
}

void Glm5NextVisionImpl::verify_loaded_weights() const {
  for (const auto& parameter : named_parameters()) {
    CHECK(loaded_weights_.count(parameter.key()))
        << "Missing vision weight: model.visual." << parameter.key();
  }
}

}  // namespace xllm::layer
