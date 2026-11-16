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
#include <torch_npu/torch_npu.h>

#include <string>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangRopeWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
  }
};

struct RopeTestCase {
  std::string name;
  int64_t num_tokens;
  int64_t num_heads;
  int64_t full_head_dim;
  int64_t rope_dim;
  int64_t start_dim;
  int64_t seed;
};

torch::Tensor torch_rope_ref(const torch::Tensor& x,
                             const torch::Tensor& sin,
                             const torch::Tensor& cos) {
  auto cos_ref = cos;
  auto sin_ref = sin;
  if (cos_ref.dim() == 2) {
    cos_ref = cos_ref.unsqueeze(1);
    sin_ref = sin_ref.unsqueeze(1);
  }

  auto x_fp32 = x.to(torch::kFloat32);
  auto cos_fp32 = cos_ref.to(torch::kFloat32);
  auto sin_fp32 = sin_ref.to(torch::kFloat32);

  auto x_reshaped =
      x_fp32.view({x_fp32.size(0), x_fp32.size(1), x_fp32.size(2) / 2, 2});
  auto x0 = x_reshaped.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              0});
  auto x1 = x_reshaped.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              1});
  auto x_rotated = torch::stack({-x1, x0}, /*dim=*/-1).flatten(-2);

  auto out = x_fp32 * cos_fp32 + x_rotated * sin_fp32;
  return out.to(torch::kBFloat16);
}

torch::Tensor maybe_narrow(const torch::Tensor& tensor,
                           int64_t start_dim,
                           int64_t rope_dim) {
  if (start_dim == 0 && rope_dim == tensor.size(2)) {
    return tensor;
  }
  return tensor.narrow(/*dim=*/2, /*start=*/start_dim, /*length=*/rope_dim);
}

void run_apply_rotary_case(const RopeTestCase& test_case) {
  const torch::Device device("npu:0");
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  torch::manual_seed(test_case.seed);
  const torch::Tensor input = torch::randn(
      {test_case.num_tokens, test_case.num_heads, test_case.full_head_dim},
      options);
  const torch::Tensor sin_cache =
      torch::randn({test_case.num_tokens, test_case.rope_dim}, options);
  const torch::Tensor cos_cache =
      torch::randn({test_case.num_tokens, test_case.rope_dim}, options);
  const torch::Tensor expected = torch_rope_ref(
      maybe_narrow(input, test_case.start_dim, test_case.rope_dim),
      sin_cache,
      cos_cache);

  torch::Tensor actual = input.clone();
  torch::Tensor rotated =
      maybe_narrow(actual, test_case.start_dim, test_case.rope_dim);
  rope_in_place(rotated, sin_cache, cos_cache);
  EXPECT_TRUE(torch::allclose(rotated, expected, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "max_diff="
      << (rotated.to(torch::kFloat32) - expected.to(torch::kFloat32))
             .abs()
             .max()
             .item<float>();
  if (test_case.start_dim > 0) {
    EXPECT_TRUE(torch::equal(actual.narrow(2, 0, test_case.start_dim),
                             input.narrow(2, 0, test_case.start_dim)));
  }
}

TEST_F(TileLangRopeWrapperTest, ApplyRotaryMatchesNpuReferenceVariant128x128) {
  const std::vector<RopeTestCase> cases = {
      {.name = "baseline_16x4_hd128_rd128",
       .num_tokens = 16,
       .num_heads = 4,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 20260213},
      {.name = "large_tokens_2051x2_hd128_rd128",
       .num_tokens = 2051,
       .num_heads = 2,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 20260214},
      {.name = "tiny_1x1_hd128_rd128",
       .num_tokens = 1,
       .num_heads = 1,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 101},
      {.name = "odd_tokens_7x3_hd128_rd128",
       .num_tokens = 7,
       .num_heads = 3,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 102},
      {.name = "token_dim_64x4_hd128_rd128",
       .num_tokens = 64,
       .num_heads = 4,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 107},
      {.name = "chunk_boundary_8x5_hd128_rd128",
       .num_tokens = 8,
       .num_heads = 5,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 103},
      {.name = "cross_chunk_9x5_hd128_rd128",
       .num_tokens = 9,
       .num_heads = 5,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 104},
      {.name = "head_dim_4x64_hd128_rd128",
       .num_tokens = 4,
       .num_heads = 64,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 108},
      {.name = "medium_127x8_hd128_rd128",
       .num_tokens = 127,
       .num_heads = 8,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 105},
      {.name = "large_heads_33x16_hd128_rd128",
       .num_tokens = 33,
       .num_heads = 16,
       .full_head_dim = 128,
       .rope_dim = 128,
       .start_dim = 0,
       .seed = 106},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name
                                      << ", num_tokens=" << test_case.num_tokens
                                      << ", num_heads=" << test_case.num_heads);
    run_apply_rotary_case(test_case);
  }
}

TEST_F(TileLangRopeWrapperTest, ApplyRotaryMatchesNpuReferenceVariant576x64) {
  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kFullHeadDim = 576;
  constexpr int64_t kStartDim = 512;
  constexpr int64_t kRopeDim = 64;

  const std::vector<RopeTestCase> cases = {
      {.name = "1x576_start512_rope64",
       .num_tokens = 1,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260226},
      {.name = "47x576_start512_rope64",
       .num_tokens = 47,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260301},
      {.name = "48x576_start512_rope64",
       .num_tokens = 48,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260302},
      {.name = "49x576_start512_rope64",
       .num_tokens = 49,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260303},
      {.name = "95x576_start512_rope64",
       .num_tokens = 95,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260304},
      {.name = "96x576_start512_rope64",
       .num_tokens = 96,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260305},
      {.name = "97x576_start512_rope64",
       .num_tokens = 97,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260306},
      {.name = "2048x576_start512_rope64",
       .num_tokens = 2048,
       .num_heads = kNumHeads,
       .full_head_dim = kFullHeadDim,
       .rope_dim = kRopeDim,
       .start_dim = kStartDim,
       .seed = 20260225},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name
                                      << ", num_tokens=" << test_case.num_tokens
                                      << ", num_heads=" << test_case.num_heads);
    run_apply_rotary_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
