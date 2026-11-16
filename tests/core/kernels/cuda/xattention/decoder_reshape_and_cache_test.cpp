/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kernels/cuda/xattention/xattention_ops_api.h"

namespace xllm::kernel::cuda {
namespace test {

class DecoderReshapeAndCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);
    dtype_ = torch::kFloat16;
  }

  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_ = torch::kFloat16;
};

// This test validates `xllm::kernel::cuda::decoder_reshape_and_cache` against
// a PyTorch reference implementation that follows the current CUDA kernel
// contract.
//
// Kernel input contract:
// - proj_k/proj_v: [batch_size, beam_size, kv_heads, head_dim], where
//
// Cache layout:
// - unshared_k/v_cache: [max_num_request, beam_size, max_decode_step, kv_heads,
//   head_dim]
//
//
// The test checks:
// 1) full-tensor equivalence (CUDA output vs reference output)
// 2) per-sequence slice correctness at the target decode step
void torch_reference(const torch::Tensor& proj_k,
                     const torch::Tensor& proj_v,
                     torch::Tensor& key_cache,
                     torch::Tensor& value_cache,
                     const torch::Tensor& step) {
  const int64_t position = step.item<int32_t>();
  key_cache.narrow(0, 0, proj_k.size(0)).select(2, position).copy_(proj_k);
  value_cache.narrow(0, 0, proj_v.size(0)).select(2, position).copy_(proj_v);
}

TEST_F(DecoderReshapeAndCacheTest, CorrectnessTest) {
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t max_num_request = 2;
  const int64_t max_decode_step = 3;

  torch::Tensor step = torch::tensor({1}, torch::kInt32).to(device_);

  const auto float_opts = torch::TensorOptions().device(device_).dtype(dtype_);

  torch::Tensor proj_k =
      torch::randn({batch_size, beam_size, kv_heads, head_dim}, float_opts);
  torch::Tensor proj_v =
      torch::randn({batch_size, beam_size, kv_heads, head_dim}, float_opts);

  torch::Tensor unshared_k_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      float_opts);
  torch::Tensor unshared_v_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      float_opts);

  torch::Tensor ref_k_cache = unshared_k_cache.clone();
  torch::Tensor ref_v_cache = unshared_v_cache.clone();

  decoder_reshape_and_cache(
      proj_k, proj_v, unshared_k_cache, unshared_v_cache, step);

  torch_reference(proj_k, proj_v, ref_k_cache, ref_v_cache, step);

  EXPECT_TRUE(torch::allclose(unshared_k_cache, ref_k_cache, 1e-5, 1e-5));
  EXPECT_TRUE(torch::allclose(unshared_v_cache, ref_v_cache, 1e-5, 1e-5));
}

}  // namespace test
}  // namespace xllm::kernel::cuda
