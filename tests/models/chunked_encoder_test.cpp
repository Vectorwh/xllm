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

#include "models/vlm/utils/chunked_encoder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace xllm {
namespace {

torch::Tensor encode_spatial_units(const torch::Tensor& pixels,
                                   const torch::Tensor& grid_thw) {
  const auto grid_values = grid_thw.to(torch::kInt64);
  const auto grids = grid_values.accessor<int64_t, 2>();
  std::vector<torch::Tensor> outputs;
  outputs.reserve(
      grid_thw.select(/*dim=*/1, /*index=*/0).sum().item<int64_t>());
  int64_t offset = 0;
  for (int64_t item = 0; item < grid_thw.size(0); ++item) {
    const int64_t spatial_tokens = grids[item][1] * grids[item][2];
    for (int64_t unit = 0; unit < grids[item][0]; ++unit) {
      const auto input = pixels.narrow(/*dim=*/0, offset, spatial_tokens);
      const auto weights = torch::softmax(
          torch::matmul(input, input.transpose(/*dim0=*/0, /*dim1=*/1)),
          /*dim=*/1);
      const auto hidden = torch::matmul(weights, input) + input;
      const auto merged =
          hidden.reshape({-1, 4, pixels.size(1)}).mean(/*dim=*/1);
      outputs.emplace_back(
          torch::cat({merged, merged.tanh(), merged.square()}, /*dim=*/1));
      offset += spatial_tokens;
    }
  }
  return torch::cat(outputs, /*dim=*/0);
}

TEST(ChunkedEncoderDeathTest, RejectsPixelRowsThatDoNotMatchGridPatches) {
  const auto grids = torch::tensor({3, 2, 2}, torch::kInt64).reshape({1, 3});
  const auto encode = [](const torch::Tensor& pixels,
                         const torch::Tensor& /*unused*/) { return pixels; };
  for (const int64_t rows : {11, 13}) {
    const auto pixels = torch::zeros({rows, 4});
    EXPECT_DEATH(
        run_chunked_encoder(pixels, grids, /*max_units_per_chunk=*/1, encode),
        "Pixel rows must match the patch count described by grid_thw");
  }
}

TEST(ChunkedEncoderTest, MatchesSpatialAttentionAndPackedDeepstackOutputs) {
  const auto pixels =
      torch::arange(44 * 4, torch::kFloat32).reshape({44, 4}) / 100;
  for (const torch::ScalarType grid_dtype : {torch::kInt32, torch::kInt64}) {
    const auto grids =
        torch::tensor({2, 2, 4, 3, 4, 2, 1, 2, 2}, grid_dtype).reshape({3, 3});
    const auto expected = encode_spatial_units(pixels, grids);
    for (const int64_t budget : {1, 2, 4}) {
      int64_t calls = 0;
      const auto actual = run_chunked_encoder(
          pixels,
          grids,
          budget,
          [&](const torch::Tensor& chunk_pixels,
              const torch::Tensor& chunk_grids) {
            EXPECT_EQ(chunk_grids.scalar_type(), grid_dtype);
            EXPECT_EQ(chunk_grids.device(), grids.device());
            const int64_t units = chunk_grids.select(/*dim=*/1, /*index=*/0)
                                      .sum()
                                      .item<int64_t>();
            EXPECT_LE(units, budget);
            ++calls;
            return encode_spatial_units(chunk_pixels, chunk_grids);
          });
      EXPECT_EQ(calls, (6 + budget - 1) / budget);
      EXPECT_EQ(actual.scalar_type(), pixels.scalar_type());
      EXPECT_EQ(actual.device(), pixels.device());
      EXPECT_TRUE(torch::allclose(actual, expected));
    }
  }
}

TEST(ChunkedEncoderTest, DoesNotSplitAnOversizedSpatialAttentionUnit) {
  const auto pixels =
      torch::arange(64 * 4, torch::kFloat32).reshape({64, 4}) / 100;
  const auto grids = torch::tensor({1, 8, 8}, torch::kInt64).reshape({1, 3});
  const auto expected = encode_spatial_units(pixels, grids);
  int64_t calls = 0;
  const auto actual = run_chunked_encoder(
      pixels,
      grids,
      /*max_units_per_chunk=*/1,
      [&](const torch::Tensor& chunk_pixels, const torch::Tensor& chunk_grids) {
        ++calls;
        EXPECT_EQ(chunk_pixels.size(0), 64);
        return encode_spatial_units(chunk_pixels, chunk_grids);
      });
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(torch::allclose(actual, expected));
}

TEST(ChunkedEncoderTest, UsesOriginalTensorsWhenDisabledOrWithinBudget) {
  const auto pixels = torch::arange(12 * 4, torch::kFloat32).reshape({12, 4});
  const auto grids = torch::tensor({3, 2, 2}, torch::kInt64).reshape({1, 3});
  const auto expected = encode_spatial_units(pixels, grids);
  for (const int64_t budget : {-1, 0, 3, 4}) {
    int64_t calls = 0;
    const auto actual = run_chunked_encoder(
        pixels,
        grids,
        budget,
        [&](const torch::Tensor& chunk_pixels,
            const torch::Tensor& chunk_grids) {
          ++calls;
          EXPECT_TRUE(chunk_pixels.is_same(pixels));
          EXPECT_TRUE(chunk_grids.is_same(grids));
          return encode_spatial_units(chunk_pixels, chunk_grids);
        });
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(torch::equal(actual, expected));
  }
}

TEST(ChunkedEncoderTest, PreservesBfloat16OutputsAndNoncontiguousInputs) {
  const auto pixels =
      (torch::arange(4 * 12, torch::kFloat32).reshape({4, 12}) / 100)
          .to(torch::kBFloat16)
          .transpose(/*dim0=*/0, /*dim1=*/1);
  const auto grids = torch::tensor({3, 2, 2}, torch::kInt64).reshape({1, 3});
  const auto encode = [](const torch::Tensor& chunk_pixels,
                         const torch::Tensor& /*unused*/) {
    return chunk_pixels.reshape({-1, 4, 4}).sum(/*dim=*/1);
  };
  const auto expected = encode(pixels, grids);
  const auto actual =
      run_chunked_encoder(pixels, grids, /*max_units_per_chunk=*/1, encode);
  EXPECT_TRUE(torch::equal(actual, expected));
  EXPECT_EQ(actual.scalar_type(), torch::kBFloat16);
}

}  // namespace
}  // namespace xllm
