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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/kernels/cuda/xattention/xattention_ops_api.h"

class CacheSelctTest : public ::testing::Test {
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

void cache_select_reference(const torch::Tensor& beam_indices,
                            std::vector<torch::Tensor>& keys,
                            std::vector<torch::Tensor>& values,
                            const torch::Tensor& block_table,
                            int64_t decode_step,
                            int64_t beam_size) {
  const torch::Tensor parents =
      (beam_indices.reshape({-1, beam_size}) / beam_size).to(torch::kInt64);
  for (size_t layer = 0; layer < keys.size(); ++layer) {
    const torch::Tensor source_keys = keys[layer].clone();
    const torch::Tensor source_values = values[layer].clone();
    for (int64_t batch = 0; batch < block_table.size(0); ++batch) {
      const int64_t request = block_table[batch].item<int32_t>();
      keys[layer][request]
          .narrow(1, 0, decode_step + 1)
          .copy_(source_keys[request]
                     .index_select(0, parents[batch])
                     .narrow(1, 0, decode_step + 1));
      values[layer][request]
          .narrow(1, 0, decode_step + 1)
          .copy_(source_values[request]
                     .index_select(0, parents[batch])
                     .narrow(1, 0, decode_step + 1));
    }
  }
}

TEST_F(CacheSelctTest, CorrectnessTest) {
  // Small shapes are enough to catch indexing bugs, while keeping the test
  // fast.
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  int32_t current_step = 1;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t layer_num = 2;
  const int64_t max_num_request = 1;
  const int64_t max_decode_step = 3;
  const auto float_opts = torch::TensorOptions().device(device_).dtype(dtype_);
  const auto int_opts =
      torch::TensorOptions().device(device_).dtype(torch::kInt32);

  // The swap permutation exercises both in-place copy directions.
  torch::Tensor beam_index =
      torch::tensor({2, 1}, int_opts).view({batch_size * beam_size, 1});
  std::vector<torch::Tensor> base_k_cache;
  std::vector<torch::Tensor> base_v_cache;

  for (int64_t layer = 0; layer < layer_num; ++layer) {
    base_k_cache.push_back(torch::randn(
        {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
        float_opts));
    base_v_cache.push_back(torch::randn(
        {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
        float_opts));
  }

  std::vector<torch::Tensor> k_cache_normal;
  std::vector<torch::Tensor> v_cache_normal;
  k_cache_normal.reserve(layer_num);
  v_cache_normal.reserve(layer_num);
  for (int64_t layer = 0; layer < layer_num; ++layer) {
    k_cache_normal.push_back(base_k_cache[layer].clone());
    v_cache_normal.push_back(base_v_cache[layer].clone());
  }

  // CPU reference uses request-level block table: [batch_size, 1].
  torch::Tensor block_table =
      torch::arange(batch_size, int_opts).view({batch_size, 1});

  // CUDA kernel uses sequence-level block table shape [batch_size * beam_size,
  // 1].
  torch::Tensor seq_block_table =
      torch::arange(batch_size * beam_size, int_opts)
          .view({batch_size * beam_size, 1});

  cache_select_reference(beam_index,
                         k_cache_normal,
                         v_cache_normal,
                         block_table,
                         current_step,
                         beam_size);

  // CUDA kernel version (single launch over layer axis).
  std::vector<torch::Tensor> k_cache_cuda;
  std::vector<torch::Tensor> v_cache_cuda;
  k_cache_cuda.reserve(layer_num);
  v_cache_cuda.reserve(layer_num);
  for (int64_t layer = 0; layer < layer_num; ++layer) {
    k_cache_cuda.push_back(base_k_cache[layer].clone());
    v_cache_cuda.push_back(base_v_cache[layer].clone());
  }
  xllm::kernel::cuda::cache_select(beam_index,
                                   k_cache_cuda,
                                   v_cache_cuda,
                                   seq_block_table,
                                   current_step,
                                   beam_size,
                                   layer_num);
  for (int64_t layer = 0; layer < layer_num; ++layer) {
    EXPECT_TRUE(torch::allclose(
        k_cache_normal[layer], k_cache_cuda[layer], 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(
        v_cache_normal[layer], v_cache_cuda[layer], 1e-5, 1e-5));
  }
}
