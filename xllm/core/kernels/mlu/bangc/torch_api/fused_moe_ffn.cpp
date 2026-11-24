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

#include "kernels/mlu/bangc/include/fused_moe_ffn.h"

#include <ATen/DeviceGuard.h>
#include <cnrt.h>
#include <glog/logging.h>

#include <algorithm>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/src/fused_moe_ffn/launcher.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu::bangc {
namespace {
constexpr int32_t kGroup = 128;
constexpr float kSwiGluLimit = 10.0f;
constexpr int32_t kGateTile = 192;
constexpr int32_t kDownTile = 256;

torch::Tensor pack_weight(const torch::Tensor& weight,
                          int32_t tile,
                          int32_t quarters) {
  const int64_t experts = weight.size(0);
  const int64_t tiles = weight.size(1) / tile;
  const int64_t groups = weight.size(2) / (kGroup / 2);
  // Within each WRAM quarter, order packed nibbles by LT, group and lane.
  return weight
      .view({experts, tiles, tile / 64, 16, 4, quarters, groups / quarters, 64})
      .permute({0, 1, 5, 3, 6, 2, 4, 7})
      .contiguous()
      .view({experts, tiles, quarters, 16, groups / quarters, tile / 16, 64});
}

torch::Tensor pack_scale_cpu(const torch::Tensor& scale, int32_t tile) {
  return scale.to(torch::kCPU)
      .view({scale.size(0), scale.size(1), scale.size(2) / tile, tile})
      .permute({1, 2, 0, 3})
      .contiguous();
}
}  // namespace

std::optional<PackedMoeFfn> PackedMoeFfn::try_pack(
    MoeFfnWeights weights,
    const MoeFfnPackOptions& options) {
  if (options.weight_bits != 4 || options.group_size != kGroup ||
      options.output_dtype != torch::kBFloat16 || !options.is_gated ||
      options.activation != "silu" || options.swiglu_limit != kSwiGluLimit) {
    return std::nullopt;
  }

  auto& [w13, w2, scale13, scale2, smooth] = weights;
  if (!detail::check_tensor(w13, torch::kInt8) || w13.dim() != 3) {
    return std::nullopt;
  }
  const int64_t experts = w13.size(0);
  const auto device = w13.device();
  if (!detail::check_tensor(w2, torch::kInt8, device) ||
      !detail::check_tensor(scale13, torch::kFloat32, device) ||
      !detail::check_tensor(scale2, torch::kFloat32, device) ||
      !detail::check_tensor(smooth, torch::kFloat32, device) || experts <= 0 ||
      experts > kFfnMaxExperts ||
      w13.sizes() !=
          torch::IntArrayRef({experts, 2 * kFfnIntermediate, kFfnHidden / 2}) ||
      w2.sizes() !=
          torch::IntArrayRef({experts, kFfnHidden, kFfnIntermediate / 2}) ||
      scale13.sizes() !=
          torch::IntArrayRef(
              {kFfnHidden / kGroup, experts, 2 * kFfnIntermediate}) ||
      scale2.sizes() != torch::IntArrayRef(
                            {kFfnIntermediate / kGroup, experts, kFfnHidden}) ||
      smooth.sizes() != torch::IntArrayRef({experts, kFfnIntermediate}) ||
      !detail::supports_mtp592_union8(device)) {
    return std::nullopt;
  }
  torch::Tensor s13 = pack_scale_cpu(scale13, kGateTile);
  torch::Tensor s2 = pack_scale_cpu(scale2, kDownTile);
  // The kernel has one scale format for both projections. Keep FP32 if
  // either conversion would change even one checkpoint scale.
  auto short13 = s13.to(torch::kBFloat16);
  auto short2 = s2.to(torch::kBFloat16);
  const bool exact = short13.to(torch::kFloat32).equal(s13) &&
                     short2.to(torch::kFloat32).equal(s2);
  auto packed_w13 = pack_weight(w13, kGateTile, /*quarters=*/4);
  auto packed_w2 = pack_weight(w2, kDownTile, /*quarters=*/3);
  s13 = (exact ? short13 : s13).to(scale13.device());
  s2 = (exact ? short2 : s2).to(scale2.device());
  w13.set_data(packed_w13);
  w2.set_data(packed_w2);
  scale13.set_data(s13);
  scale2.set_data(s2);
  return PackedMoeFfn(std::move(weights));
}

torch::Tensor PackedMoeFfn::forward(const torch::Tensor& input,
                                    const torch::Tensor& input_scale,
                                    const torch::Tensor& counts) const {
  const auto& [w13, w2, scale13, scale2, smooth] = weights_;
  const auto device = w13.device();
  const torch::DeviceGuard guard(device);
  CHECK(detail::check_tensor(input, torch::kInt8, device) && input.dim() == 2 &&
        input.size(1) == kFfnHidden);
  const int64_t rows = input.size(0);
  const int64_t experts = w13.size(0);
  CHECK(detail::check_tensor(input_scale, torch::kFloat32, device) &&
        input_scale.numel() >= rows);
  CHECK(detail::check_tensor(counts, torch::kInt32, device) &&
        counts.numel() == experts);
  auto output =
      torch::empty({rows, kFfnHidden}, input.options().dtype(torch::kBFloat16));
  if (rows == 0) {
    return output;
  }
  // Dispatch buffers may be larger than the routed prefix. Each launch
  // intersects device-resident expert counts with its row range, so capture
  // and replay never need to read counts on the host.
  const int64_t chunk = std::min<int64_t>(rows, kMaxLaunchRows);
  auto workspace = torch::empty(
      {chunk * (kFfnIntermediate + static_cast<int64_t>(sizeof(float))) + 128},
      input.options().dtype(torch::kByte));
  const cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  for (int64_t offset = 0; offset < rows; offset += chunk) {
    const int32_t n = static_cast<int32_t>(std::min(chunk, rows - offset));
    fused_moe_ffn_launch(
        queue,
        output.data_ptr<torch::BFloat16>() + offset * kFfnHidden,
        workspace.data_ptr(),
        input.data_ptr<int8_t>() + offset * kFfnHidden,
        input_scale.data_ptr<float>() + offset,
        w13.data_ptr<int8_t>(),
        w2.data_ptr<int8_t>(),
        scale13.data_ptr(),
        scale2.data_ptr(),
        smooth.data_ptr<float>(),
        counts.data_ptr<int32_t>(),
        n,
        static_cast<int32_t>(experts),
        kSwiGluLimit,
        scale13.scalar_type() == torch::kBFloat16,
        offset);
    CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
  }
  return output;
}
}  // namespace xllm::kernel::mlu::bangc
