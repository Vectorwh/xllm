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

#include "core/layers/common/attention_mask.h"

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include <cstdint>
#include <limits>

namespace xllm::layer {
namespace {

TEST(AttentionMaskTest, AppendMaskPreservesCausalAndPaddingValues) {
  const torch::Device device(torch::kPrivateUse1, 0);
  for (const torch::Dtype dtype :
       {torch::kFloat32, torch::kBFloat16, torch::kFloat16}) {
    AttentionMask attention_mask(device, dtype);
    const float mask_value = dtype == torch::kFloat16
                                 ? -std::numeric_limits<float>::infinity()
                                 : -9984.0f;
    for (const int32_t kv_len : {3, 5}) {
      const torch::Tensor actual =
          attention_mask.gen_append_mask(3, kv_len, 7, dtype, device);
      torch::Tensor expected = torch::empty({3, 7}, torch::kFloat32);
      auto values = expected.accessor<float, 2>();
      for (int32_t row = 0; row < 3; ++row) {
        for (int32_t column = 0; column < 7; ++column) {
          values[row][column] = column <= kv_len - 3 + row ? 0.0f : mask_value;
        }
      }
      EXPECT_TRUE(torch::equal(actual.cpu(), expected.to(dtype)));
      EXPECT_EQ(actual.scalar_type(), dtype);
      EXPECT_EQ(actual.device(), device);
      EXPECT_TRUE(actual.is_contiguous());
    }
    const torch::Tensor empty =
        attention_mask.gen_append_mask(0, 5, 7, dtype, device);
    EXPECT_EQ(empty.size(0), 0);
    EXPECT_EQ(empty.size(1), 7);
  }
}

TEST(AttentionMaskTest, AppendMaskPeakMemoryStaysWithinOutputBudget) {
  const torch::Device device(torch::kPrivateUse1, 0);
  AttentionMask attention_mask(device, torch::kBFloat16);
  attention_mask.gen_append_mask(2, 2, 2, torch::kBFloat16, device);
  c10_npu::device_synchronize();
  c10_npu::NPUCachingAllocator::resetPeakStats(0);
  const int64_t initial_bytes = c10_npu::NPUCachingAllocator::getDeviceStats(0)
                                    .allocated_bytes[0]
                                    .current;
  const torch::Tensor mask = attention_mask.gen_append_mask(
      4096, 4096, 4096, torch::kBFloat16, device);
  c10_npu::device_synchronize();
  const int64_t peak_bytes =
      c10_npu::NPUCachingAllocator::getDeviceStats(0).allocated_bytes[0].peak;
  const int64_t output_bytes = mask.numel() * mask.element_size();
  constexpr int64_t kAllocatorMarginBytes = 2 * 1024 * 1024;
  EXPECT_LE(peak_bytes - initial_bytes, output_bytes + kAllocatorMarginBytes);
}

}  // namespace
}  // namespace xllm::layer
