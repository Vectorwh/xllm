/* Copyright 2025-2026 The xLLM Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/
// clang-format off
#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include "core/kernels/cuda/cuda_ops_api.h"
#include "core/kernels/cuda/cutlass_extensions/common.hpp"

namespace {

class StaticScaledFP8QuantTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA is unavailable.";
    }
    if (xllm::kernel::cuda::get_sm_version_num() < 89) {
      GTEST_SKIP() << "FP8 requires SM89 or newer.";
    }
    device_ = torch::Device(torch::kCUDA);
  }

  torch::TensorOptions options(torch::ScalarType dtype) const {
    return torch::TensorOptions().device(device_).dtype(dtype);
  }

  void run_and_check(const torch::Tensor& input, float scale_value) const {
    torch::Tensor out = torch::empty(input.sizes(), options(torch::kFloat8_e4m3fn));
    const torch::Tensor scale = torch::tensor({scale_value}, options(torch::kFloat32));
    xllm::kernel::cuda::static_scaled_fp8_quant(out, input, scale);
    const torch::Tensor expected = (input.to(torch::kFloat32) / scale)
                                      .clamp(-448.0, 448.0)
                                      .to(torch::kFloat8_e4m3fn)
                                      .to(torch::kFloat32);
    EXPECT_TRUE(torch::equal(out.to(torch::kFloat32), expected));
  }

  torch::Device device_{torch::kCPU};
};

TEST_F(StaticScaledFP8QuantTest, BasicFloat32InputTest) {
  run_and_check(torch::randn({128, 256}, options(torch::kFloat32)), 1.0F);
}

TEST_F(StaticScaledFP8QuantTest, Float16InputTest) {
  run_and_check(torch::randn({64, 512}, options(torch::kFloat16)), 0.5F);
}

TEST_F(StaticScaledFP8QuantTest, BFloat16InputTest) {
  run_and_check(torch::randn({32, 128}, options(torch::kBFloat16)), 2.0F);
}

TEST_F(StaticScaledFP8QuantTest, DifferentScalesTest) {
  const torch::Tensor input = torch::randn({64, 256}, options(torch::kFloat32)) * 0.1F;
  // Dyadic scales isolate the conversion from an unrelated FP32 half-tie.
  for (const float scale : {0.125F, 8.0F}) {
    run_and_check(input, scale);
  }
}

// Cover both a row shorter than one FP8 vector and an unaligned row with a
// vectorized body plus scalar tail; unlike the old shape smoke this checks exact
// quantized values against the torch conversion reference.
TEST_F(StaticScaledFP8QuantTest, DifferentSizesTest) {
  run_and_check(torch::randn({1, 35}, options(torch::kFloat32)), 1.0F);
  run_and_check(torch::randn({16, 257}, options(torch::kFloat32)), 1.0F);
}

TEST_F(StaticScaledFP8QuantTest, BatchedTensor3DTest) {
  run_and_check(torch::randn({4, 32, 128}, options(torch::kFloat32)), 1.0F);
}

TEST_F(StaticScaledFP8QuantTest, LargeInputWithScaleTest) {
  const torch::Tensor input = torch::tensor(
      {{-1000.0F, -449.0F, -448.0F, -0.25F, -0.125F, -0.0F,
        0.0F, 0.125F, 0.25F, 448.0F, 449.0F, 1000.0F}},
      options(torch::kFloat32));
  run_and_check(input, 1.0F);
}

TEST_F(StaticScaledFP8QuantTest, ZeroValuesTest) {
  run_and_check(torch::zeros({16, 32}, options(torch::kFloat32)), 1.0F);
}

}  // namespace
