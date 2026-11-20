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

#include <limits>
#include <utility>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
struct Inputs {
  torch::Tensor x, positions, rope, weight, scale, smooth_a, smooth_b;

  Inputs to(const torch::Device& device) const {
    return {x.to(device),
            positions.to(device),
            rope.to(device),
            weight.to(device),
            scale.to(device),
            smooth_a.to(device),
            smooth_b.to(device)};
  }
};

using Output = std::tuple<torch::Tensor, torch::Tensor>;

Inputs make_inputs(int64_t rows, int64_t groups) {
  torch::manual_seed(41);
  auto x = torch::randint(-64, 65, {rows, groups, 8, 512})
               .mul(torch::arange(1, groups + 1).view({1, groups, 1, 1}))
               .to(torch::kBFloat16)
               .flatten(1, 2);
  x[0].zero_();
  auto angles = torch::arange(41, torch::kFloat32).unsqueeze(1) *
                torch::arange(1, 33, torch::kFloat32).repeat_interleave(2) *
                0.013;
  return {x,
          (torch::arange(rows, torch::kInt32) * 7).remainder(41),
          torch::cat({angles.cos(), angles.sin()}, 1),
          torch::randint(-8, 9, {groups, 1024, 4096}, torch::kInt8),
          torch::rand({1024 * groups}) * 0.01 + 0.002,
          torch::rand({4096}) + 0.25,
          torch::rand({1024 * groups}) + 0.25};
}

Output quantize(const torch::Tensor& x, const torch::Tensor& smooth) {
  auto values = x.to(torch::kFloat32) * smooth;
  auto peak = values.abs().amax(-1, true).to(torch::kFloat64);
  auto multiplier =
      torch::where(peak > 0, 127.0 / peak, 0.).to(torch::kFloat32);
  return {(values * multiplier).round().clamp(-127, 127).to(torch::kInt8),
          (peak.squeeze(-1) / 127.0).to(torch::kFloat32)};
}

Output reference(const Inputs& in) {
  const auto groups = in.weight.size(0);
  auto positions = in.positions.clamp(0, in.rope.size(0) - 1).to(torch::kLong);
  auto table = in.rope.index_select(0, positions).unsqueeze(1);
  auto cos = table.slice(-1, 0, 64, 2);
  auto sin = table.slice(-1, 64, 128, 2);
  auto even = in.x.slice(-1, 448, 512, 2).to(torch::kFloat32);
  auto odd = in.x.slice(-1, 449, 512, 2).to(torch::kFloat32);
  auto tail = torch::stack({even * cos + odd * sin, odd * cos - even * sin}, -1)
                  .flatten(-2)
                  .to(torch::kBFloat16);
  auto rotated = torch::cat({in.x.slice(-1, 0, 448), tail}, -1);
  auto [a, sa] = quantize(rotated.view({-1, groups, 4096}), in.smooth_a);
  // Weights are bounded by 8, so each integer sum fits exactly in FP32.
  auto accum = torch::bmm(a.transpose(0, 1).to(torch::kFloat32),
                          in.weight.transpose(1, 2).to(torch::kFloat32))
                   .transpose(0, 1);
  auto scale = sa.unsqueeze(-1) * in.scale.view({1, groups, 1024});
  return quantize((accum * scale).to(torch::kBFloat16).flatten(1), in.smooth_b);
}

Output run(const Inputs& in) {
  return fused_dsa_output_projection(in.x,
                                     in.positions,
                                     in.rope,
                                     in.weight,
                                     in.scale,
                                     in.smooth_a,
                                     in.smooth_b);
}

void check(const Output& output, const Inputs& cpu, const Inputs& device) {
  const auto& [q, s] = output;
  const auto [expected, scale] = reference(cpu);
  ASSERT_EQ(q.sizes(), expected.sizes());
  ASSERT_EQ(s.sizes(), scale.sizes());
  ASSERT_EQ(q.scalar_type(), torch::kInt8);
  ASSERT_EQ(s.scalar_type(), torch::kFloat32);
  ASSERT_TRUE(q.is_contiguous() && s.is_contiguous());
  auto actual = q.cpu().to(torch::kFloat32);
  auto actual_scale = s.cpu();
  auto delta = (actual - expected).abs();
  EXPECT_LE(delta.max().item<int>(), 1);
  EXPECT_LE(delta.count_nonzero().item<int64_t>() * 1000, q.numel());
  EXPECT_TRUE(torch::allclose(actual_scale, scale, 1e-5, 1e-8));
  auto error =
      (actual * actual_scale.unsqueeze(1) - expected * scale.unsqueeze(1))
          .abs();
  auto bound = delta * scale.unsqueeze(1) +
               actual.abs() * (1e-8 + 1e-5 * scale.unsqueeze(1));
  EXPECT_TRUE(
      (error <= bound + (actual * actual_scale.unsqueeze(1)).abs() * 2.4e-7)
          .all()
          .item<bool>());
  EXPECT_EQ(actual[0].count_nonzero().item<int>(), 0);
  EXPECT_EQ(actual_scale[0].item<float>(), 0);
  for (auto member : {&Inputs::x,
                      &Inputs::positions,
                      &Inputs::rope,
                      &Inputs::weight,
                      &Inputs::scale,
                      &Inputs::smooth_a,
                      &Inputs::smooth_b}) {
    EXPECT_TRUE(torch::equal((device.*member).cpu(), cpu.*member));
  }
}

void check_eager(int64_t rows, int64_t groups) {
  auto cpu = make_inputs(rows, groups);
  auto in =
      cpu.to(torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  check(run(in), cpu, in);
}

TEST(FusedDSAOutputProjectionTest, ReplaysUpdatedActivationAndPositions) {
  auto cpu = make_inputs(6, 1);
  auto in =
      cpu.to(torch::Device(torch::kPrivateUse1, torch_mlu::current_device()));
  torch_mlu::synchronize();
  torch_mlu::mlu::MLUStreamGuard stream(
      torch_mlu::getStreamFromPool(false, in.x.device().index()));
  check(run(in), cpu, in);
  torch_mlu::MLUGraph graph;
  graph.capture_begin();
  auto output = run(in);
  graph.capture_end();
  cpu.x.neg_();
  cpu.positions = torch::tensor({0, -1, 39, 40, 41, 42}, torch::kInt32);
  in.x.copy_(cpu.x);
  in.positions.copy_(cpu.positions);
  graph.replay();
  check(output, cpu, in);
}

TEST(FusedDSAOutputProjectionTest, ClampsPositionsAcrossDispatchPaths) {
  const torch::Device device(torch::kPrivateUse1, torch_mlu::current_device());
  for (const auto& [rows, groups] :
       {std::pair{48, 1}, std::pair{48, 2}, std::pair{769, 1}}) {
    SCOPED_TRACE(::testing::Message()
                 << "rows=" << rows << ", groups=" << groups);
    auto in = make_inputs(rows, groups).to(device);
    const int32_t last = static_cast<int32_t>(in.rope.size(0) - 1);
    auto pattern = torch::tensor({std::numeric_limits<int32_t>::min(),
                                  -1,
                                  0,
                                  last - 1,
                                  last,
                                  last + 1,
                                  last + 2,
                                  std::numeric_limits<int32_t>::max()},
                                 torch::kInt32);
    auto positions =
        pattern.repeat({(rows + pattern.numel() - 1) / pattern.numel()})
            .slice(0, 0, rows);
    in.positions = positions.to(device);
    auto bounded = in;
    bounded.positions = positions.clamp(0, last).to(device);
    const auto [expected, expected_scales] = run(bounded);
    const auto [actual, actual_scales] = run(in);
    EXPECT_TRUE(torch::equal(actual, expected));
    EXPECT_TRUE(torch::equal(actual_scales, expected_scales));
    EXPECT_TRUE(torch::equal(in.positions.cpu(), positions));
  }
}

TEST(FusedDSAOutputProjectionTest, QuantizesAcrossGroups) {
  check_eager(20, 2);
}

TEST(FusedDSAOutputProjectionTest, ProjectsPrefillTail) {
  check_eager(1025, 1);
}

TEST(FusedDSAOutputProjectionTest, ProjectsLastSmallMultiGroupBatch) {
  check_eager(96, 8);
}

TEST(FusedDSAOutputProjectionTest, ProjectsFirstLargeMultiGroupBatch) {
  check_eager(97, 8);
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
