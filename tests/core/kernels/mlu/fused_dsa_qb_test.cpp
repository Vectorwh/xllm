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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
struct Inputs {
  torch::Tensor qr, weight, positions, rope, scale, smooth;
};

Inputs make_inputs(int32_t rows, int32_t heads, bool quantized) {
  torch::manual_seed(592);
  const int32_t cols = heads * 512;
  const auto channels = torch::arange(cols, torch::kLong);
  const auto angles =
      torch::arange(17, torch::kFloat32).unsqueeze(1) *
      (torch::arange(32, torch::kFloat32) + 1).repeat_interleave(2) * 0.01;
  Inputs in;
  in.qr = (torch::randint(-123, 124, {rows, 1280}).to(torch::kFloat32) / 4)
              .to(torch::kBFloat16);
  in.weight =
      torch::zeros({cols, 1280}, quantized ? torch::kInt8 : torch::kBFloat16);
  in.weight.index_put_({channels, (channels * 13).remainder(1280)}, 3);
  in.weight.index_put_({channels, (channels * 13 + 1).remainder(1280)}, -7);
  in.positions = (torch::arange(rows, torch::kInt32) * 11 + 3).remainder(17);
  in.rope = torch::cat({angles.cos(), angles.sin()}, 1);
  if (quantized) {
    // Give rows exact, distinct dynamic scales and keep a zero row.
    in.qr.select(1, 0).copy_(127 * (torch::arange(rows).remainder(2) + 1));
    in.qr.select(0, 0).zero_();
    in.smooth = (torch::arange(1280).remainder(3) + 1).to(torch::kFloat32);
    in.scale = (channels.remainder(11) + 1).to(torch::kFloat32) / 64;
  }
  return in;
}

torch::Tensor reference(const Inputs& in) {
  torch::Tensor qr = in.qr.to(torch::kFloat64);
  torch::Tensor token_scale;
  if (in.scale.defined()) {
    qr = qr * in.smooth;
    token_scale = std::get<0>(qr.abs().max(1, true)) / 127;
    qr = (qr / torch::where(token_scale == 0, 1., token_scale))
             .round()
             .clamp(-127, 127);
  }
  torch::Tensor q = torch::matmul(qr, in.weight.to(torch::kFloat64).t());
  if (in.scale.defined()) {
    q = q * token_scale * in.scale;
  }
  q = q.to(torch::kBFloat16).view({qr.size(0), -1, 512}).to(torch::kFloat32);
  const auto rows = in.positions.to(torch::kLong).clamp(0, in.rope.size(0) - 1);
  const auto table = in.rope.index_select(0, rows).unsqueeze(1);
  const auto even = q.slice(-1, 448, 512, 2).clone();
  const auto odd = q.slice(-1, 449, 512, 2).clone();
  const auto cos = table.slice(-1, 0, 64, 2);
  const auto sin = table.slice(-1, 64, 128, 2);
  q.slice(-1, 448, 512, 2).copy_(even * cos - odd * sin);
  q.slice(-1, 449, 512, 2).copy_(odd * cos + even * sin);
  return q.to(torch::kBFloat16);
}

Inputs on_mlu(const Inputs& in) {
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  return {in.qr.to(device),
          in.weight.to(device),
          in.positions.to(device),
          in.rope.to(device),
          in.scale.defined() ? in.scale.to(device) : torch::Tensor(),
          in.smooth.defined() ? in.smooth.to(device) : torch::Tensor()};
}

torch::Tensor run(const Inputs& in) {
  return fused_dsa_qb(
      in.qr, in.weight, in.positions, in.rope, in.scale, in.smooth);
}

void check_output(const torch::Tensor& out, const Inputs& cpu) {
  const torch::Tensor expected = reference(cpu);
  ASSERT_EQ(out.sizes(), expected.sizes());
  ASSERT_EQ(out.scalar_type(), torch::kBFloat16);
  ASSERT_TRUE(out.is_contiguous());
  const torch::Tensor actual = out.cpu();
  EXPECT_TRUE(
      torch::equal(actual.slice(-1, 0, 448), expected.slice(-1, 0, 448)));
  // FP32 multiply/add contraction in the rotated tail can differ by one BF16
  // ULP.
  EXPECT_TRUE(torch::allclose(actual.slice(-1, 448).to(torch::kFloat32),
                              expected.slice(-1, 448).to(torch::kFloat32),
                              0.008,
                              1e-5));
}

void set_boundary_positions(Inputs& in) {
  in.positions[1] = -1;
  in.positions[-1] = in.rope.size(0);
}

TEST(FusedDSAQbTest, ReplaysInt8LocalShardWithUpdatedInputs) {
  Inputs cpu = make_inputs(/*rows=*/24, /*heads=*/8, /*quantized=*/true);
  const Inputs in = on_mlu(cpu);
  torch_mlu::synchronize();
  torch_mlu::mlu::MLUStreamGuard stream(
      torch_mlu::getStreamFromPool(false, in.qr.device().index()));
  check_output(run(in), cpu);
  torch_mlu::MLUGraph graph;
  graph.capture_begin();
  const torch::Tensor out = run(in);
  graph.capture_end();
  cpu.qr.neg_();
  cpu.positions = (cpu.positions + 1).remainder(17);
  set_boundary_positions(cpu);
  in.qr.copy_(cpu.qr);
  in.positions.copy_(cpu.positions);
  graph.replay();
  check_output(out, cpu);
}

TEST(FusedDSAQbTest, QuantizesPrefillWithLocalChannelScales) {
  Inputs cpu = make_inputs(/*rows=*/129, /*heads=*/16, /*quantized=*/true);
  set_boundary_positions(cpu);
  check_output(run(on_mlu(cpu)), cpu);
}

TEST(FusedDSAQbTest, ClampsBfloat16DecodePositions) {
  Inputs cpu = make_inputs(/*rows=*/8, /*heads=*/8, /*quantized=*/false);
  set_boundary_positions(cpu);
  check_output(run(on_mlu(cpu)), cpu);
}

TEST(FusedDSAQbTest, RotatesBfloat16PrefillAtFullWidth) {
  Inputs cpu = make_inputs(/*rows=*/97, /*heads=*/64, /*quantized=*/false);
  set_boundary_positions(cpu);
  check_output(run(on_mlu(cpu)), cpu);
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
