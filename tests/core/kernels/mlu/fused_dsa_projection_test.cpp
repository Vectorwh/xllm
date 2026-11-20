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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
class FusedDSAProjectionTest : public ::testing::Test {
 protected:
  const torch::Device device{torch::kPrivateUse1, 0};
  const torch::DeviceGuard guard{device};
  const torch::TensorOptions bf16 =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const torch::Tensor w_kv = torch::zeros({512, 5120}, bf16);
  const torch::Tensor w_qa = torch::zeros({1280, 5120}, bf16);
  const torch::Tensor gamma_kv = torch::ones({512}, bf16);
  const torch::Tensor gamma_q = torch::ones({1280}, bf16);
  const torch::Tensor rope =
      torch::cat({torch::ones({2, 64}, bf16.dtype(torch::kFloat32)),
                  torch::zeros({2, 64}, bf16.dtype(torch::kFloat32))},
                 1);

  torch::Tensor compressor_input(int64_t rows) {
    auto x = torch::zeros({rows, 5120}, bf16);
    for (int64_t column : {0, 1, 4096}) {
      x.select(1, column).fill_(1);
    }
    return x;
  }
};

TEST_F(FusedDSAProjectionTest, NormalizesAndRotatesSelectedPositions) {
  torch::manual_seed(41);
  constexpr float kEps = 1e-5f;
  // Keep the projected mean square comparable to kEps.
  auto x = torch::randn({6, 5120}, bf16) / 16384;
  w_kv.normal_();
  w_qa.normal_();
  gamma_kv.normal_();
  gamma_q.normal_();
  rope[1].slice(0, 0, 64).zero_();
  rope[1].slice(0, 64).fill_(1);
  auto positions = torch::tensor({1, 0, 1, 1, 0, 0}, bf16.dtype(torch::kInt32));
  auto out = fused_dsa_projection(
      x, w_kv, w_qa, gamma_kv, gamma_q, positions, rope, kEps);
  ASSERT_FALSE(out.raw_scores.defined());
  const auto normalize = [&](const torch::Tensor& weight,
                             const torch::Tensor& gamma) {
    auto z = torch::matmul(x.cpu().to(torch::kFloat64),
                           weight.cpu().to(torch::kFloat64).t())
                 .to(torch::kBFloat16)
                 .to(torch::kFloat64);
    return (z * torch::rsqrt(z.square().mean(1, true) + kEps) *
            gamma.cpu().to(torch::kFloat64))
        .to(torch::kBFloat16);
  };
  auto kv = normalize(w_kv, gamma_kv);
  const auto qr = normalize(w_qa, gamma_q);
  const auto tail = kv.slice(1, 448).reshape({6, 32, 2});
  const auto rotated = torch::stack({-tail.select(2, 1), tail.select(2, 0)}, 2);
  tail.copy_(torch::where(
      positions.cpu().view({6, 1, 1}).to(torch::kBool), rotated, tail));
  EXPECT_EQ(out.kv.scalar_type(), torch::kBFloat16);
  EXPECT_EQ(out.qr.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::allclose(out.kv.cpu(), kv, 0.015, 0.002));
  EXPECT_TRUE(torch::allclose(out.qr.cpu(), qr, 0.015, 0.002));
}

TEST_F(FusedDSAProjectionTest, RoundsRawAfterCompleteReduction) {
  auto x = compressor_input(1);
  auto weight = torch::zeros_like(w_kv);
  weight.select(1, 0).fill_(1);
  weight.select(1, 1).fill_(1.0 / 256);
  weight.select(1, 4096).fill_(-1);
  auto positions = torch::zeros({1}, bf16.dtype(torch::kInt32));
  auto out = fused_dsa_projection(x,
                                  w_kv,
                                  w_qa,
                                  gamma_kv,
                                  gamma_q,
                                  positions,
                                  rope,
                                  /*rms_norm_eps=*/1e-20f,
                                  weight);
  ASSERT_EQ(out.raw_scores.scalar_type(), torch::kBFloat16);
  ASSERT_TRUE(out.raw_scores.is_contiguous());
  EXPECT_TRUE(
      torch::equal(out.raw_scores, torch::full({1, 512}, 1.0 / 256, bf16)));
}

TEST_F(FusedDSAProjectionTest, LargeTileTailKeepsPackedFp32Outputs) {
  constexpr int64_t rows = 4097;
  auto x = compressor_input(rows);
  x[rows - 1].mul_(2);
  auto weight = torch::zeros_like(w_kv);
  weight.select(1, 0).fill_(1);
  weight.select(1, 4096).fill_(1.0 / 256);
  auto gate = -weight;
  auto positions = torch::zeros({rows}, bf16.dtype(torch::kInt32));
  const auto guarded = [&](int64_t width, torch::ScalarType dtype) {
    return torch::full({rows * width + 2}, -7, bf16.dtype(dtype));
  };
  auto kv = guarded(512, torch::kBFloat16);
  auto qr = guarded(1280, torch::kBFloat16);
  auto packed = guarded(1024, torch::kFloat32);
  DSAProjectionOutput buffers{kv.slice(0, 1, -1).view({rows, 512}),
                              qr.slice(0, 1, -1).view({rows, 1280}),
                              packed.slice(0, 1, -1).view({rows, 1024})};
  auto out = fused_dsa_projection(x,
                                  w_kv,
                                  w_qa,
                                  gamma_kv,
                                  gamma_q,
                                  positions,
                                  rope,
                                  /*rms_norm_eps=*/1e-20f,
                                  weight,
                                  gate,
                                  buffers);
  ASSERT_TRUE(out.raw_scores.is_same(buffers.raw_scores));
  auto raw = out.raw_scores.slice(1, 0, 512);
  auto scores = out.raw_scores.slice(1, 512);
  auto expected = torch::full_like(raw, 1 + 1.0 / 256);
  expected[rows - 1].mul_(2);
  EXPECT_EQ(raw.stride(0), 1024);
  EXPECT_EQ(scores.storage_offset() - raw.storage_offset(), 512);
  EXPECT_TRUE(torch::equal(raw, expected));
  EXPECT_TRUE(torch::equal(scores, -expected));
  EXPECT_TRUE(torch::equal(buffers.kv, torch::zeros_like(buffers.kv)));
  EXPECT_TRUE(torch::equal(buffers.qr, torch::zeros_like(buffers.qr)));
  for (const auto& backing : {kv, qr, packed}) {
    EXPECT_EQ(backing[0].item<float>(), -7);
    EXPECT_EQ(backing[-1].item<float>(), -7);
  }
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
