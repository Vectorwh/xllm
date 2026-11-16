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
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {
namespace {

struct BlockCopyCaseConfig {
  int64_t num_layers;
  int64_t num_blocks;
  int64_t block_size;
  int64_t num_heads;
  int64_t head_dim;
  std::vector<int32_t> src_blocks;
  std::vector<int32_t> dst_blocks;
  std::vector<int32_t> cum_sum;
};

struct BlockCopyLaunchInputs {
  torch::Tensor key_ptr_tensor;
  torch::Tensor value_ptr_tensor;
  torch::Tensor src_tensor;
  torch::Tensor dst_tensor;
  torch::Tensor cum_sum_tensor;
  int64_t numel_per_block;
};

struct NativeBlockCopyLaunchInputs {
  torch::Tensor src_tensor;
  torch::Tensor dst_tensor;
};

std::vector<int64_t> flatten_src_blocks_for_native(
    const std::vector<int32_t>& src_blocks,
    const std::vector<int32_t>& cum_sum) {
  std::vector<int64_t> flat_src_blocks;
  flat_src_blocks.reserve(cum_sum.empty() ? 0 : cum_sum.back());
  for (size_t group_idx = 0; group_idx < src_blocks.size(); ++group_idx) {
    const int32_t begin = group_idx == 0 ? 0 : cum_sum[group_idx - 1];
    const int32_t end = cum_sum[group_idx];
    for (int32_t dst_idx = begin; dst_idx < end; ++dst_idx) {
      flat_src_blocks.push_back(src_blocks[group_idx]);
    }
  }
  return flat_src_blocks;
}

void native_block_copy(std::vector<torch::Tensor>& key_caches,
                       std::vector<torch::Tensor>& value_caches,
                       const NativeBlockCopyLaunchInputs& launch_inputs) {
  for (size_t layer_idx = 0; layer_idx < key_caches.size(); ++layer_idx) {
    auto selected_keys =
        torch::index_select(key_caches[layer_idx], 0, launch_inputs.src_tensor);
    auto selected_values = torch::index_select(
        value_caches[layer_idx], 0, launch_inputs.src_tensor);
    key_caches[layer_idx].index_copy_(
        0, launch_inputs.dst_tensor, selected_keys);
    value_caches[layer_idx].index_copy_(
        0, launch_inputs.dst_tensor, selected_values);
  }
}

BlockCopyLaunchInputs prepare_block_copy_launch_inputs(
    const std::vector<torch::Tensor>& key_caches,
    const std::vector<torch::Tensor>& value_caches,
    const std::vector<int32_t>& src_blocks,
    const std::vector<int32_t>& dst_blocks,
    const std::vector<int32_t>& cum_sum,
    const torch::Device& device) {
  std::vector<int64_t> key_ptrs;
  std::vector<int64_t> value_ptrs;
  key_ptrs.reserve(key_caches.size());
  value_ptrs.reserve(value_caches.size());
  for (size_t layer_idx = 0; layer_idx < key_caches.size(); ++layer_idx) {
    key_ptrs.push_back(
        reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr()));
    value_ptrs.push_back(
        reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr()));
  }

  const auto ptr_opts =
      torch::TensorOptions().device(device).dtype(torch::kInt64);
  const auto idx_opts =
      torch::TensorOptions().device(device).dtype(torch::kInt32);
  return {
      .key_ptr_tensor = torch::tensor(key_ptrs, ptr_opts),
      .value_ptr_tensor = torch::tensor(value_ptrs, ptr_opts),
      .src_tensor = torch::tensor(src_blocks, idx_opts),
      .dst_tensor = torch::tensor(dst_blocks, idx_opts),
      .cum_sum_tensor = torch::tensor(cum_sum, idx_opts),
      .numel_per_block = key_caches[0][0].numel(),
  };
}

NativeBlockCopyLaunchInputs prepare_native_block_copy_launch_inputs(
    const std::vector<int32_t>& src_blocks,
    const std::vector<int32_t>& dst_blocks,
    const std::vector<int32_t>& cum_sum,
    const torch::Device& device) {
  auto flat_src_blocks = flatten_src_blocks_for_native(src_blocks, cum_sum);
  std::vector<int64_t> flat_dst_blocks(dst_blocks.begin(), dst_blocks.end());
  return {
      .src_tensor = torch::tensor(
          flat_src_blocks,
          torch::TensorOptions().device(device).dtype(torch::kLong)),
      .dst_tensor = torch::tensor(
          flat_dst_blocks,
          torch::TensorOptions().device(device).dtype(torch::kLong)),
  };
}

void kernel_block_copy(const BlockCopyLaunchInputs& launch_inputs,
                       torch::ScalarType dtype) {
  xllm::kernel::cuda::block_copy(launch_inputs.key_ptr_tensor,
                                 launch_inputs.value_ptr_tensor,
                                 launch_inputs.src_tensor,
                                 launch_inputs.dst_tensor,
                                 launch_inputs.cum_sum_tensor,
                                 launch_inputs.numel_per_block,
                                 dtype);
}

std::vector<torch::Tensor> make_random_caches(const BlockCopyCaseConfig& config,
                                              const torch::Device& device,
                                              torch::ScalarType dtype) {
  std::vector<torch::Tensor> caches;
  caches.reserve(config.num_layers);
  const auto opts = torch::TensorOptions().device(device).dtype(dtype);
  for (int64_t layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
    caches.push_back(torch::randn({config.num_blocks,
                                   config.block_size,
                                   config.num_heads,
                                   config.head_dim},
                                  opts));
  }
  return caches;
}

void expect_caches_allclose(const std::vector<torch::Tensor>& lhs,
                            const std::vector<torch::Tensor>& rhs,
                            double rtol,
                            double atol) {
  ASSERT_EQ(lhs.size(), rhs.size());
  for (size_t idx = 0; idx < lhs.size(); ++idx) {
    EXPECT_TRUE(torch::allclose(lhs[idx], rhs[idx], rtol, atol))
        << "cache mismatch at layer=" << idx;
  }
}

void run_accuracy_compare_case(const BlockCopyCaseConfig& config,
                               torch::ScalarType dtype,
                               double rtol,
                               double atol) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test.";
  }

  torch::manual_seed(2026);
  const auto device = torch::Device(torch::kCUDA, 0);

  auto base_key_caches = make_random_caches(config, device, dtype);
  auto base_value_caches = make_random_caches(config, device, dtype);
  auto kernel_key_caches = base_key_caches;
  auto kernel_value_caches = base_value_caches;
  // Tensor-vector copies alias storage. Clone the oracle's tensors before
  // executing the kernel so native index_copy cannot mask a no-op kernel.
  std::vector<torch::Tensor> native_key_caches;
  std::vector<torch::Tensor> native_value_caches;
  native_key_caches.reserve(base_key_caches.size());
  native_value_caches.reserve(base_value_caches.size());
  for (size_t layer = 0; layer < base_key_caches.size(); ++layer) {
    native_key_caches.emplace_back(base_key_caches[layer].clone());
    native_value_caches.emplace_back(base_value_caches[layer].clone());
  }

  auto kernel_launch_inputs =
      prepare_block_copy_launch_inputs(kernel_key_caches,
                                       kernel_value_caches,
                                       config.src_blocks,
                                       config.dst_blocks,
                                       config.cum_sum,
                                       device);
  auto native_launch_inputs = prepare_native_block_copy_launch_inputs(
      config.src_blocks, config.dst_blocks, config.cum_sum, device);

  kernel_block_copy(kernel_launch_inputs, dtype);
  native_block_copy(
      native_key_caches, native_value_caches, native_launch_inputs);
  torch::cuda::synchronize();

  expect_caches_allclose(kernel_key_caches, native_key_caches, rtol, atol);
  expect_caches_allclose(kernel_value_caches, native_value_caches, rtol, atol);
}

}  // namespace

TEST(BlockCopyTest, KernelMatchesReferenceFp16) {
  run_accuracy_compare_case(
      BlockCopyCaseConfig{
          .num_layers = 3,
          .num_blocks = 8,
          .block_size = 3,
          .num_heads = 1,
          .head_dim = 7,
          .src_blocks = {1, 4},
          .dst_blocks = {2, 3, 6},
          .cum_sum = {2, 3},
      },
      torch::kHalf,
      1e-5,
      1e-5);
}

TEST(BlockCopyTest, KernelMatchesReferenceFp32) {
  run_accuracy_compare_case(
      BlockCopyCaseConfig{
          .num_layers = 4,
          .num_blocks = 10,
          .block_size = 8,
          .num_heads = 3,
          .head_dim = 16,
          .src_blocks = {1, 4, 7},
          .dst_blocks = {2, 3, 5, 8, 9},
          .cum_sum = {2, 4, 5},
      },
      torch::kFloat,
      1e-6,
      1e-6);
}

TEST(BlockCopyTest, KernelMatchesNativeFp16) {
  run_accuracy_compare_case(
      BlockCopyCaseConfig{
          .num_layers = 6,
          .num_blocks = 32,
          .block_size = 16,
          .num_heads = 4,
          .head_dim = 32,
          .src_blocks = {1, 4, 9, 12},
          .dst_blocks = {2, 3, 5, 6, 10, 11, 20},
          .cum_sum = {2, 4, 6, 7},
      },
      torch::kHalf,
      1e-5,
      1e-5);
}

}  // namespace test
}  // namespace xllm::kernel::dcu
