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

#include "core/kernels/ops_api.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm::kernel {
namespace {

TEST(MusaOpsApiTest, UngatedActivationPreservesOutputStorage) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  ActivationParams params;
  params.input = torch::tensor({{-2.0f, -0.5f, 0.0f, 1.0f}}, torch::kFloat32)
                     .to(device.unwrap());
  params.output = torch::empty_like(params.input);
  params.is_gated = false;
  params.act_mode = "silu";
  const torch::Tensor original = params.output;
  active(params);
  EXPECT_EQ(params.output.data_ptr(), original.data_ptr());
  EXPECT_TRUE(torch::allclose(original.cpu(), torch::silu(params.input.cpu())));
}

TEST(MusaOpsApiTest, NativeRmsNormPreservesGammaOffsetAndResidual) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  constexpr int64_t kHidden = 1024;
  constexpr double kEps = 1e-6;
  // Plain BF16, Gemma with/without residual, and Float vec8 residual loading.
  for (int32_t scenario = 0; scenario < 4; ++scenario) {
    const torch::ScalarType dtype =
        scenario == 2 ? torch::kFloat32 : torch::kBFloat16;
    const bool gamma_offset = scenario == 1 || scenario == 3;
    SCOPED_TRACE(torch::toString(dtype));
    const torch::Tensor input =
        torch::linspace(-1.5, 2.0, 2 * kHidden).reshape({2, kHidden}).to(dtype);
    const torch::Tensor residual = input.flip({1}).contiguous();
    const torch::Tensor weight = torch::linspace(-0.4, 0.4, kHidden).to(dtype);
    const double tolerance = dtype == torch::kFloat32 ? 1e-5 : 1e-2;
    FusedLayerNormParams params;
    params.input = input.to(device.unwrap());
    params.output = torch::empty_like(params.input);
    params.weight = weight.to(device.unwrap());
    params.eps = kEps;
    params.add_gamma_offset = gamma_offset;
    const torch::Tensor original_output = params.output;
    torch::Tensor sum = input.to(torch::kFloat32);
    if (scenario == 1 || scenario == 2) {
      params.residual = residual.to(device.unwrap());
      sum.add_(residual.to(torch::kFloat32));
    }
    fused_layernorm(params);
    const torch::Tensor gamma =
        weight.to(torch::kFloat32) + (gamma_offset ? 1.0 : 0.0);
    const torch::Tensor expected =
        (sum * torch::rsqrt(sum.square().mean(-1, true) + kEps) * gamma)
            .to(dtype);
    if (scenario == 0) {
      EXPECT_TRUE(torch::equal(params.output.cpu(), expected));
    } else {
      EXPECT_TRUE(
          torch::allclose(params.output.cpu(), expected, tolerance, tolerance));
    }
    if (scenario == 1 || scenario == 2) {
      EXPECT_TRUE(torch::equal(params.residual_out->cpu(), sum.to(dtype)));
    } else {
      EXPECT_EQ(params.output.data_ptr(), original_output.data_ptr());
      params.output = torch::Tensor();
      fused_layernorm(params);
      EXPECT_TRUE(torch::equal(params.output, original_output));
      EXPECT_FALSE(params.residual_out.has_value());
      EXPECT_TRUE(torch::equal(params.input.cpu(), input));
    }
  }
}

TEST(MusaOpsApiTest, LayerNormPreservesPreallocatedOutputs) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  FusedLayerNormParams params;
  params.input =
      torch::linspace(-1.0, 2.0, 256).reshape({4, 64}).to(device.unwrap());
  params.residual = params.input * 0.25;
  params.weight = torch::linspace(0.4, 1.2, 64).to(device.unwrap());
  params.beta = torch::linspace(-0.2, 0.2, 64).to(device.unwrap());
  params.output = torch::empty_like(params.input);
  params.residual_out = torch::empty_like(params.input);
  params.mode = "layernorm";
  params.eps = 1e-6;
  const torch::Tensor output = params.output;
  const torch::Tensor residual_out = params.residual_out.value();
  const torch::Tensor input = params.input.clone();
  const torch::Tensor residual = params.residual->clone();
  const torch::Tensor sum = input + residual;
  const torch::Tensor expected = torch::layer_norm(
      sum.cpu(), {64}, params.weight.cpu(), params.beta->cpu(), params.eps);
  fused_layernorm(params);
  EXPECT_EQ(params.output.data_ptr(), output.data_ptr());
  EXPECT_EQ(params.residual_out->data_ptr(), residual_out.data_ptr());
  EXPECT_TRUE(torch::allclose(params.output.cpu(), expected, 1e-5, 1e-5));
  EXPECT_TRUE(torch::equal(params.residual_out.value(), sum));
  EXPECT_TRUE(torch::equal(params.input, input));
  EXPECT_TRUE(torch::equal(params.residual.value(), residual));
  params.output = residual_out;
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(fused_layernorm(params), "must not share storage");
}

}  // namespace
}  // namespace xllm::kernel
