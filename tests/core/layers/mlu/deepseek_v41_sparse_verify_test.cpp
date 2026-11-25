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

#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_indexer.h"
#include "layers/mlu/tests_utils.h"

namespace xllm::layer {
namespace {

TEST(DeepseekV41GraphTest, SparseVerificationIgnoresCausallyMaskedSelections) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.compress_ratios({2}).index_topk(512).candidate_source_layer_id(-1);
  DeepseekV41Indexer indexer(ModelContext(parallel, args, QuantArgs(), options),
                             0);
  Csa2Context shared;
  const auto ints = options.dtype(torch::kInt32);
  shared.num_valid_input_rows = torch::full({1}, 4, ints);
  shared.block_tables.resize(2);
  shared.block_tables[1] = torch::zeros({1, 1}, ints);
  shared.index_history[1] = torch::full({1}, 2, ints);
  indexer->select_indices(torch::ones({4, 32, 128}, options),
                          torch::ones({4, 32}, options),
                          torch::ones({128, 128}, options),
                          torch::arange(4, ints),
                          torch::zeros({4}, ints),
                          true,
                          shared);
  auto indices = shared.packed_topk.slice(1, 510).to(torch::kInt64);
  auto expected_indices =
      torch::tensor({-1, -1, -1, 0, -1, 0, 0, 1}, torch::kInt64).view({4, 2});
  ASSERT_TRUE(torch::equal(indices.cpu(), expected_indices));
  namespace bangc = kernel::mlu::bangc;
  auto swa = torch::zeros({256, 512}, options);
  auto compressed = torch::zeros_like(swa);
  compressed[63].fill_(64);
  compressed[64].fill_(4);
  compressed[65].fill_(8);
  bangc::PagedAttentionMetadata m;
  m.swa_table = torch::zeros({1, 1}, ints);
  m.swa_columns = torch::ones({1}, ints);
  m.compressed_table = torch::ones({1, 1}, ints);
  m.topk = shared.packed_topk;
  m.prefix = torch::tensor({0, 4}, ints);
  m.begins = torch::zeros({1}, ints);
  m.valid_rows = shared.num_valid_input_rows;
  const auto out = bangc::fused_paged_sparsed_attention(
      bangc::PagedAttentionStage::VERIFY,
      torch::zeros({4, 8, 512}, options),
      torch::zeros({4, 512}, options),
      swa,
      compressed,
      torch::zeros({8}, options.dtype(torch::kFloat32)),
      m,
      2);
  const auto expected =
      torch::tensor({0.0f, 1.0f, 0.8f, 12.0f / 7}, torch::kFloat32)
          .view({4, 1, 1})
          .expand({4, 8, 512});
  EXPECT_TRUE(
      torch::allclose(out.cpu().to(torch::kFloat32), expected, 0.005, 0.001));
}
}  // namespace
}  // namespace xllm::layer
