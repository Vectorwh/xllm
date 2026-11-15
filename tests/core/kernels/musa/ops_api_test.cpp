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

}  // namespace
}  // namespace xllm::kernel
