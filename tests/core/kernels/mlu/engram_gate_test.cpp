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

#include "kernels/mlu/bangc/include/engram_gate.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <vector>

namespace xllm::kernel::mlu::bangc {
namespace {

torch::Tensor reference(const torch::Tensor& hidden,
                        const torch::Tensor& kv,
                        const torch::Tensor& q,
                        const torch::Tensor& k,
                        double eps = 1e-5) {
  const int64_t heads = hidden.size(1);
  const int64_t width = hidden.size(2);
  const auto h = hidden.to(torch::kFloat32);
  const auto keys =
      kv.slice(1, 0, heads * width).reshape_as(h).to(torch::kFloat32);
  const auto values = kv.slice(1, heads * width).to(torch::kFloat32);
  auto dot = (h * keys * q.to(torch::kFloat32) * k.to(torch::kFloat32)).sum(-1);
  dot *= torch::rsqrt(h.square().mean(-1) + eps) *
         torch::rsqrt(keys.square().mean(-1) + eps) / std::sqrt(width);
  const auto gate = torch::sigmoid(torch::where(dot < 0, -1.0, 1.0) *
                                   dot.abs().clamp_min(1e-6).sqrt());
  return (h + gate.unsqueeze(-1) * values.unsqueeze(1)).to(torch::kBFloat16);
}

TEST(EngramGateTest, MatchesCpuReferenceAcrossShapesAndGateSigns) {
  torch::manual_seed(319);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto cpu = torch::TensorOptions().dtype(torch::kBFloat16);
  const std::vector<std::vector<int64_t>> shapes = {{0, 2, 8},
                                                    {1, 2, 8},
                                                    {3, 3, 37},
                                                    {6, 4, 5120},
                                                    {31, 4, 5120},
                                                    {32, 4, 5120},
                                                    {33, 4, 5120},
                                                    {129, 4, 5120},
                                                    {2, 2, 8209}};
  for (const auto& shape : shapes) {
    const int64_t tokens = shape[0];
    const int64_t heads = shape[1];
    const int64_t width = shape[2];
    auto h = torch::randn(shape, cpu);
    auto kv = torch::randn({tokens, (heads + 1) * width}, cpu);
    auto q = torch::randn({heads, width}, cpu);
    auto k = torch::randn({heads, width}, cpu);
    for (const double sign : {-1.0, 0.0, 1.0}) {
      const auto signed_q = (q * sign).to(torch::kBFloat16);
      const auto expected = reference(h, kv, signed_q, k);
      const auto actual = engram_gate(h.to(device),
                                      kv.to(device),
                                      signed_q.to(device),
                                      k.to(device),
                                      1e-5)
                              .cpu();
      // FP32 reduction trees can differ; the final BF16 cast can differ by one
      // representable value. Near zero keep an absolute FP32 error allowance.
      EXPECT_TRUE(torch::allclose(actual, expected, 0.008, 1e-4))
          << "shape=" << torch::IntArrayRef(shape) << " sign=" << sign;
    }
  }
}

TEST(EngramGateTest, GraphReplayUsesUpdatedInputs) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto cpu = torch::TensorOptions().dtype(torch::kBFloat16);
  const std::vector<std::vector<int64_t>> shapes = {{2, 2, 37}, {33, 4, 5120}};
  for (const auto& shape : shapes) {
    const int64_t tokens = shape[0];
    const int64_t heads = shape[1];
    const int64_t width = shape[2];
    auto h = torch::ones(shape, cpu).to(device);
    auto kv = torch::ones({tokens, (heads + 1) * width}, cpu).to(device);
    auto q = torch::ones({heads, width}, cpu).to(device);
    auto k = torch::ones({heads, width}, cpu).to(device);
    engram_gate(h, kv, q, k, 1e-5);
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    torch::Tensor output;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      output = engram_gate(h, kv, q, k, 1e-5);
      graph.capture_end();
    }
    for (const double factor : {-0.5, 0.0, 0.75}) {
      q.fill_(factor);
      h.fill_(factor);
      const auto expected = reference(h.cpu(), kv.cpu(), q.cpu(), k.cpu());
      graph.replay();
      EXPECT_TRUE(torch::allclose(output.cpu(), expected, 0.008, 1e-4))
          << "shape=" << torch::IntArrayRef(shape) << " factor=" << factor;
    }
  }
}

TEST(EngramGateTest, PreservesNonfiniteValuesInGenericAndTokenPaths) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto cpu = torch::TensorOptions().dtype(torch::kBFloat16);
  for (const int64_t tokens : {1, 33}) {
    auto h = torch::ones({tokens, 4, 5120}, cpu);
    auto kv = torch::ones({tokens, 5 * 5120}, cpu);
    auto q = torch::ones({4, 5120}, cpu);
    auto k = torch::ones({4, 5120}, cpu);
    h[0][0][0] = std::numeric_limits<float>::infinity();
    kv[0][5120] = std::numeric_limits<float>::quiet_NaN();
    q[2][0] = std::numeric_limits<float>::quiet_NaN();
    const auto expected = reference(h, kv, q, k);
    const auto actual =
        engram_gate(
            h.to(device), kv.to(device), q.to(device), k.to(device), 1e-5)
            .cpu();
    EXPECT_TRUE(torch::allclose(actual, expected, 0.008, 1e-4, true))
        << "tokens=" << tokens;
  }
}

TEST(EngramGateTest, PreservesMultiplyOrderAtWeightExtremes) {
  struct TestCase {
    float hidden;
    float key;
    float q;
    float k;
    bool cancel;
  };
  const std::vector<TestCase> cases = {{0.0f, 1.0f, 1e30f, 1e30f, false},
                                       {0.0f, 1.0f, -1e30f, 1e30f, false},
                                       {1.0f, 0.0f, 1e30f, 1e30f, false},
                                       {1e10f, 1e10f, 1e20f, 1e-20f, true},
                                       {1e10f, 1e10f, 1e-30f, -1e-30f, false},
                                       {1e20f, 1e20f, 1e-30f, 1e-30f, false}};
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto cpu = torch::TensorOptions().dtype(torch::kBFloat16);
  for (const int64_t tokens : {1, 33, 65}) {
    for (const auto& test_case : cases) {
      auto h = torch::full({tokens, 4, 5120}, test_case.hidden, cpu);
      auto kv = torch::ones({tokens, 5 * 5120}, cpu);
      kv.slice(1, 0, 4 * 5120).fill_(test_case.key);
      auto q = torch::full({4, 5120}, test_case.q, cpu);
      auto k = torch::full({4, 5120}, test_case.k, cpu);
      if (test_case.cancel) {
        k.slice(1, 1, 5120, 2).mul_(-1);
      }
      const auto expected = reference(h, kv, q, k);
      const auto actual =
          engram_gate(
              h.to(device), kv.to(device), q.to(device), k.to(device), 1e-5)
              .cpu();
      EXPECT_TRUE(torch::allclose(actual, expected, 0.008, 1e-4, true))
          << "tokens=" << tokens << " hidden=" << test_case.hidden
          << " key=" << test_case.key << " q=" << test_case.q
          << " k=" << test_case.k << " cancel=" << test_case.cancel;
    }
  }
}

TEST(EngramGateTest, PreservesNormalizationOrderAcrossPositiveFloatEps) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto cpu = torch::TensorOptions().dtype(torch::kBFloat16);
  const std::vector<double> eps_values = {
      std::numeric_limits<float>::denorm_min(),
      1e-40,
      1e-20,
      1e-5,
      1e20,
      1e38,
      3.4e38};
  const std::vector<std::vector<float>> cases = {{0.0f, 0.0f, 1.0f},
                                                 {0.0f, 1.0f, 1.0f},
                                                 {1e-20f, 1e-20f, 1e-2f},
                                                 {-0.5f, 1.0f, 1.0f}};
  for (const int64_t tokens : {1, 33}) {
    for (const auto& test_case : cases) {
      auto h = torch::full({tokens, 4, 5120}, test_case[0], cpu);
      auto kv = torch::ones({tokens, 5 * 5120}, cpu);
      kv.slice(1, 0, 4 * 5120).fill_(test_case[1]);
      auto q = torch::full({4, 5120}, test_case[2], cpu);
      auto k = torch::full({4, 5120}, test_case[2], cpu);
      for (const double eps : eps_values) {
        const auto expected = reference(h, kv, q, k, eps);
        const auto actual =
            engram_gate(
                h.to(device), kv.to(device), q.to(device), k.to(device), eps)
                .cpu();
        EXPECT_TRUE(torch::allclose(actual, expected, 0.008, 1e-4, true))
            << "tokens=" << tokens << " eps=" << eps
            << " hidden=" << test_case[0] << " key=" << test_case[1];
      }
    }
  }
}

TEST(EngramGateTest, RoundsGateProductBeforeLargeResidualCancellation) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto cpu = torch::TensorOptions().dtype(torch::kBFloat16);
  const std::vector<std::vector<int64_t>> shapes = {{1, 2, 8}, {33, 4, 5120}};
  for (const auto& shape : shapes) {
    const int64_t tokens = shape[0];
    const int64_t heads = shape[1];
    const int64_t width = shape[2];
    auto h = torch::zeros(shape, cpu);
    h.select(2, 0).fill_(std::ldexp(1.0f, 60));
    h.select(2, 1).fill_(std::ldexp(1.0f, 44));
    h.select(2, 2).fill_(std::ldexp(1.0f, 36));
    auto kv = torch::ones({tokens, (heads + 1) * width}, cpu);
    kv.slice(1, heads * width).zero_();
    kv.select(1, heads * width).fill_(-3.0f * std::ldexp(1.0f, 60));
    auto q = torch::ones({heads, width}, cpu);
    auto k = torch::ones({heads, width}, cpu);
    k.select(1, 0).fill_(-0.48046875f);
    k.select(1, 1).fill_(0.8125f);
    k.select(1, 2).fill_(15.3125f);
    // The independent formula yields float32(1/3). Rounding gate*value first
    // produces exact cancellation; fused multiply-add leaves a large residue.
    const auto expected = reference(h, kv, q, k);
    ASSERT_EQ(expected[0][0][0].item<float>(), 0.0f);
    const auto actual =
        engram_gate(
            h.to(device), kv.to(device), q.to(device), k.to(device), 1e-5)
            .cpu();
    EXPECT_TRUE(torch::allclose(actual, expected, 0.008, 1e-4))
        << "shape=" << torch::IntArrayRef(shape);
    EXPECT_EQ(actual[0][0][0].item<float>(), 0.0f);
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu::bangc
