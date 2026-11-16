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
#include "core/layers/mlu/deepseek_v41/deepseek_v41_vision.h"

#include <glog/logging.h>
#include <torch/nn/functional.h>

#include <algorithm>
#include <utility>

#include "core/util/utils.h"

namespace xllm::layer {
namespace {

torch::Tensor apply_rotary(const torch::Tensor& x,
                           const torch::Tensor& cos,
                           const torch::Tensor& sin) {
  const auto parts = x.to(torch::kFloat32).chunk(2, /*dim=*/-1);
  return torch::cat(
             {parts[0] * cos - parts[1] * sin, parts[1] * cos + parts[0] * sin},
             /*dim=*/-1)
      .to(x.scalar_type());
}

}  // namespace

DeepseekV41VisionImpl::DeepseekV41VisionImpl(const ModelContext& context)
    : args_(context.get_model_args()), options_(context.get_tensor_options()) {
  const int64_t hidden = args_.mm_hidden_size();
  const int64_t inter = args_.mm_intermediate_size();
  const int64_t output = args_.mm_projection_dim();
  const int64_t patch = args_.mm_patch_size();
  const int64_t ratio = args_.mm_downsample_ratio();
  CHECK_GT(hidden, 0);
  CHECK_GT(inter, 0);
  CHECK_GT(output, 0);
  CHECK_GT(patch, 0);
  CHECK_GT(ratio, 0);
  CHECK_GT(args_.mm_num_hidden_layers(), 0);
  CHECK_GT(args_.mm_num_attention_heads(), 0);
  CHECK_GT(args_.mm_num_channels(), 0);
  CHECK_EQ(args_.mm_num_attention_heads() * args_.mm_head_dim(), hidden);
  CHECK_EQ(args_.mm_head_dim() % 4, 0);
  CHECK_GT(args_.mm_rope_theta(), 0);

  weights_.reserve(10 + 8 * args_.mm_num_hidden_layers());
  loaded_.reserve(10 + 8 * args_.mm_num_hidden_layers());
  add_weight("vision.patch_embed.proj.weight",
             {hidden, args_.mm_num_channels() * patch * patch});
  add_weight("vision.patch_embed.proj.bias", {hidden});
  add_weight("vision.norm.weight", {hidden}, /*use_fp32=*/true);
  add_weight("image_start", {output});
  add_weight("image_end", {output});
  add_weight("image_newline", {output});
  add_weight("aligner.w1.weight", {output, hidden * ratio * ratio});
  add_weight("aligner.w1.bias", {output});
  add_weight("aligner.w2.weight", {output, output});
  add_weight("aligner.w2.bias", {output});
  for (int64_t i = 0; i < args_.mm_num_hidden_layers(); ++i) {
    const std::string prefix = "vision.blocks." + std::to_string(i) + ".";
    add_weight(prefix + "attn.wqkv.weight", {3 * hidden, hidden});
    add_weight(prefix + "attn.wqkv.bias", {3 * hidden});
    add_weight(prefix + "attn.wo.weight", {hidden, hidden});
    add_weight(prefix + "attn.wo.bias", {hidden});
    add_weight(prefix + "mlp.w1.weight", {2 * inter, hidden});
    add_weight(prefix + "mlp.w2.weight", {hidden, inter});
    add_weight(prefix + "norm1.weight", {hidden}, /*use_fp32=*/true);
    add_weight(prefix + "norm2.weight", {hidden}, /*use_fp32=*/true);
  }
}

void DeepseekV41VisionImpl::add_weight(const std::string& name,
                                       std::vector<int64_t> shape,
                                       bool use_fp32) {
  std::string parameter_name = name;
  std::replace(parameter_name.begin(), parameter_name.end(), '.', '_');
  // Python DeepseekV41VisionNorm retains FP32 checkpoint weights even in BF16.
  const auto weight_options =
      use_fp32 ? options_.dtype(torch::kFloat32) : options_;
  weights_.emplace(name,
                   register_parameter(parameter_name,
                                      torch::empty(shape, weight_options),
                                      /*requires_grad=*/false));
}

void DeepseekV41VisionImpl::load_state_dict(const StateDict& state) {
  for (auto& [name, weight] : weights_) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      value = state.get_tensor("model." + name);
    }
    if (!value.defined()) {
      continue;
    }
    // Conv2d exports and flattened Linear checkpoints use the same patch basis.
    if (name == "vision.patch_embed.proj.weight" && value.dim() == 4) {
      CHECK_EQ(value.size(0), args_.mm_hidden_size());
      CHECK_EQ(value.size(1), args_.mm_num_channels());
      CHECK_EQ(value.size(2), args_.mm_patch_size());
      CHECK_EQ(value.size(3), args_.mm_patch_size());
      value = value.flatten(1);
    }
    CHECK_EQ(value.sizes(), weight.sizes()) << "Invalid vision weight " << name;
    weight.copy_(value);
    loaded_.insert(name);
  }
}

void DeepseekV41VisionImpl::verify_loaded_weights() const {
  for (const auto& [name, weight] : weights_) {
    CHECK(loaded_.count(name)) << "Missing vision weight: " << name;
  }
}

std::vector<torch::Tensor> DeepseekV41VisionImpl::encode(
    const torch::Tensor& pixels,
    const torch::Tensor& image_grid) const {
  CHECK(pixels.dim() == 2 || pixels.dim() == 4);
  CHECK(image_grid.dim() == 1 || image_grid.dim() == 2);
  torch::Tensor grid = image_grid.to(torch::kCPU, torch::kInt64).contiguous();
  if (grid.dim() == 1) {
    grid = grid.reshape({1, -1});
  }
  CHECK_GE(grid.size(1), 2);
  const int64_t patch_area =
      args_.mm_num_channels() * args_.mm_patch_size() * args_.mm_patch_size();
  CHECK_EQ(pixels.numel(), pixels.size(0) * patch_area);
  torch::Tensor patches =
      pixels.reshape({pixels.size(0), patch_area}).to(options_);
  const auto dims = grid.accessor<int64_t, 2>();
  int64_t total_patches = 0;
  for (int64_t i = 0; i < grid.size(0); ++i) {
    CHECK_GT(dims[i][grid.size(1) - 2], 0);
    CHECK_GT(dims[i][grid.size(1) - 1], 0);
    total_patches += dims[i][grid.size(1) - 2] * dims[i][grid.size(1) - 1];
  }
  CHECK_EQ(total_patches, patches.size(0)) << "Image grid/patch count mismatch";
  std::vector<torch::Tensor> embeddings;
  embeddings.reserve(grid.size(0));
  int64_t offset = 0;
  for (int64_t i = 0; i < grid.size(0); ++i) {
    const int64_t n_h = dims[i][grid.size(1) - 2];
    const int64_t n_w = dims[i][grid.size(1) - 1];
    embeddings.emplace_back(
        forward(patches.narrow(0, offset, n_h * n_w), n_h, n_w));
    offset += n_h * n_w;
  }
  return embeddings;
}

torch::Tensor DeepseekV41VisionImpl::rms_norm(
    const torch::Tensor& x,
    const torch::Tensor& weight) const {
  torch::Tensor xf = x.to(torch::kFloat32);
  // DeepseekV41VisionConfig uses 1e-6 independently of text rms_norm_eps
  // (1e-20).
  return (xf * torch::rsqrt(xf.square().mean(-1, /*keepdim=*/true) + 1e-6) *
          weight.to(torch::kFloat32))
      .to(x.scalar_type());
}

torch::Tensor DeepseekV41VisionImpl::vision_forward(
    const torch::Tensor& patches,
    int64_t n_h,
    int64_t n_w) const {
  const int64_t heads = args_.mm_num_attention_heads();
  const int64_t head_dim = args_.mm_head_dim();
  const int64_t rope_dim = head_dim / 2;
  const auto floats = options_.dtype(torch::kFloat32);
  torch::Tensor inv_freq = torch::pow(
      args_.mm_rope_theta(), -torch::arange(0, rope_dim, 2, floats) / rope_dim);
  torch::Tensor hpos =
      torch::arange(n_h, floats).unsqueeze(1).expand({n_h, n_w});
  torch::Tensor wpos =
      torch::arange(n_w, floats).unsqueeze(0).expand({n_h, n_w});
  torch::Tensor freqs =
      (torch::stack({hpos, wpos}, -1).reshape({-1, 2, 1}) * inv_freq)
          .flatten(1);
  torch::Tensor cos = freqs.cos().unsqueeze(1);
  torch::Tensor sin = freqs.sin().unsqueeze(1);
  torch::Tensor x = torch::nn::functional::linear(
      patches,
      weights_.at("vision.patch_embed.proj.weight"),
      weights_.at("vision.patch_embed.proj.bias"));
  for (int64_t i = 0; i < args_.mm_num_hidden_layers(); ++i) {
    const std::string prefix = "vision.blocks." + std::to_string(i) + ".";
    torch::Tensor normalized =
        rms_norm(x, weights_.at(prefix + "norm1.weight"));
    const auto qkv =
        torch::nn::functional::linear(normalized,
                                      weights_.at(prefix + "attn.wqkv.weight"),
                                      weights_.at(prefix + "attn.wqkv.bias"))
            .chunk(3, /*dim=*/-1);
    torch::Tensor q =
        apply_rotary(qkv[0].reshape({-1, heads, head_dim}), cos, sin)
            .transpose(0, 1)
            .unsqueeze(0);
    torch::Tensor k =
        apply_rotary(qkv[1].reshape({-1, heads, head_dim}), cos, sin)
            .transpose(0, 1)
            .unsqueeze(0);
    torch::Tensor v =
        qkv[2].reshape({-1, heads, head_dim}).transpose(0, 1).unsqueeze(0);
    torch::Tensor attention =
        torch::scaled_dot_product_attention(q,
                                            k,
                                            v,
                                            /*attn_mask=*/std::nullopt,
                                            /*dropout_p=*/0.0,
                                            /*is_causal=*/false);
    attention = attention.squeeze(0).transpose(0, 1).reshape({n_h * n_w, -1});
    x = x +
        torch::nn::functional::linear(attention,
                                      weights_.at(prefix + "attn.wo.weight"),
                                      weights_.at(prefix + "attn.wo.bias"));
    normalized = rms_norm(x, weights_.at(prefix + "norm2.weight"));
    torch::Tensor projected = torch::nn::functional::linear(
        normalized, weights_.at(prefix + "mlp.w1.weight"));
    // Match Python's fused SwiGLU: round once after SiLU and multiplication.
    const auto gate_up = projected.to(torch::kFloat32).chunk(2, /*dim=*/-1);
    torch::Tensor activated =
        (torch::silu(gate_up[0]) * gate_up[1]).to(projected.scalar_type());
    x = x + torch::nn::functional::linear(
                activated, weights_.at(prefix + "mlp.w2.weight"));
  }
  return rms_norm(x, weights_.at("vision.norm.weight"));
}

torch::Tensor DeepseekV41VisionImpl::align(const torch::Tensor& features,
                                           int64_t n_h,
                                           int64_t n_w) const {
  const int64_t ratio = args_.mm_downsample_ratio();
  const int64_t h = util::ceil_div(n_h, ratio);
  const int64_t w = util::ceil_div(n_w, ratio);
  torch::Tensor x = features.reshape({n_h, n_w, -1}).permute({2, 0, 1});
  x = torch::constant_pad_nd(x, {0, w * ratio - n_w, 0, h * ratio - n_h}, 0);
  // Same channel-major spatial windows as unfold(kernel=ratio, stride=ratio).
  x = x.reshape({args_.mm_hidden_size(), h, ratio, w, ratio})
          .permute({1, 3, 0, 2, 4})
          .reshape({h * w, -1});
  x = torch::gelu(torch::nn::functional::linear(
      x, weights_.at("aligner.w1.weight"), weights_.at("aligner.w1.bias")));
  return torch::nn::functional::linear(
      x, weights_.at("aligner.w2.weight"), weights_.at("aligner.w2.bias"));
}

torch::Tensor DeepseekV41VisionImpl::forward(const torch::Tensor& patches,
                                             int64_t n_h,
                                             int64_t n_w) const {
  CHECK_GT(n_h, 0);
  CHECK_GT(n_w, 0);
  CHECK_EQ(patches.size(0), n_h * n_w);
  torch::Tensor features = align(vision_forward(patches, n_h, n_w), n_h, n_w);
  const int64_t h = util::ceil_div(n_h, args_.mm_downsample_ratio());
  const int64_t w = util::ceil_div(n_w, args_.mm_downsample_ratio());
  std::vector<torch::Tensor> rows;
  rows.reserve(2 + h * (w + 1));
  rows.emplace_back(weights_.at("image_start"));
  // Row-pair interleave, omitting all padding roles from the prompt layout.
  for (int64_t pair = 0; pair < h; pair += 2) {
    for (int64_t col = 0; col <= w; ++col) {
      const int64_t row_end = std::min(pair + 2, h);
      for (int64_t row = pair; row < row_end; ++row) {
        rows.emplace_back(col == w ? weights_.at("image_newline")
                                   : features[row * w + col]);
      }
    }
  }
  rows.emplace_back(weights_.at("image_end"));
  return torch::stack(rows, /*dim=*/0);
}

}  // namespace xllm::layer
