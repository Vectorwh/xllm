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

#include "models/dit/utils/lanczos_resample.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace lanczos {
namespace {

torch::Tensor rgba_fixture() {
  return torch::tensor(
             std::vector<uint8_t>{255, 10,  20,  0,   10,  240, 30,  1,
                                  50,  60,  250, 128, 200, 20,  30,  255,
                                  100, 150, 200, 254, 250, 200, 100, 64},
             torch::kUInt8)
      .reshape({2, 3, 4})
      .permute({2, 0, 1})
      .unsqueeze(0);
}

void expect_pil_pixels(const torch::Tensor& actual,
                       int64_t width,
                       int64_t height,
                       const std::vector<uint8_t>& hwc_pixels) {
  const auto expected = torch::tensor(hwc_pixels, torch::kUInt8)
                            .reshape({height, width, actual.size(1)})
                            .permute({2, 0, 1})
                            .unsqueeze(0);
  EXPECT_TRUE(actual.device().is_cpu());
  EXPECT_EQ(actual.scalar_type(), torch::kUInt8);
  EXPECT_TRUE(torch::equal(actual, expected));
}

// Golden pixels from Pillow 10.3.0 Image.resize(..., Resampling.LANCZOS).
// Include colored transparent pixels and alpha values 1, 64, 128, 254 and 255.
TEST(LanczosResize, RgbaUpsampleMatchesPilWithoutMutatingNoncontiguousInput) {
  const auto source = rgba_fixture();
  const auto original = source.clone();
  ASSERT_FALSE(source.is_contiguous());
  const auto resized = resize_rgba_like_pil(source, /*width=*/5, /*height=*/4);
  expect_pil_pixels(
      resized,
      /*width=*/5,
      /*height=*/4,
      {0,   1,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0,   29,  25,
       255, 79,  35,  52,  255, 159, 209, 4,   16,  61,  172, 73,  103, 59,
       97,  148, 195, 60,  75,  102, 236, 95,  77,  70,  229, 119, 213, 3,
       7,   192, 170, 72,  102, 196, 100, 150, 200, 195, 117, 172, 209, 117,
       205, 133, 125, 67,  247, 3,   6,   255, 201, 86,  121, 255, 118, 175,
       234, 255, 139, 208, 197, 133, 255, 255, 0,   27});
  EXPECT_TRUE(torch::equal(source, original));
}

TEST(LanczosResize, RgbaDownsampleMatchesPil) {
  expect_pil_pixels(
      resize_rgba_like_pil(rgba_fixture(), /*width=*/2, /*height=*/1),
      /*width=*/2,
      /*height=*/1,
      {167, 71, 101, 128, 102, 135, 214, 107});
}

TEST(LanczosResize, HorizontalOnlyMatchesPil) {
  expect_pil_pixels(
      resize_rgba_like_pil(rgba_fixture(), /*width=*/5, /*height=*/2),
      /*width=*/5,
      /*height=*/2,
      {63,  63,  255, 4,   0,   0,   0,   0,   0,   255, 0,   1,  50,  62,
       249, 86,  50,  59,  249, 142, 215, 3,   6,   249, 171, 73, 103, 255,
       100, 149, 199, 254, 131, 194, 202, 126, 255, 202, 5,   44});
}

TEST(LanczosResize, VerticalOnlyMatchesPil) {
  expect_pil_pixels(
      resize_rgba_like_pil(rgba_fixture(), /*width=*/3, /*height=*/4),
      /*width=*/3,
      /*height=*/4,
      {0,   0,  0,  0,   0,   0,   0,   0,   33,  47,  255, 139,
       203, 21, 30, 59,  97,  148, 195, 60,  76,  78,  230, 113,
       199, 19, 29, 196, 100, 150, 200, 195, 174, 145, 154, 79,
       235, 24, 35, 255, 118, 175, 234, 255, 255, 255, 33,  53});
}

TEST(LanczosResize, SameSizePreservesTransparentColorsAndLowAlpha) {
  const auto source = rgba_fixture();
  const auto resized = resize_rgba_like_pil(source, /*width=*/3, /*height=*/2);
  EXPECT_TRUE(torch::equal(source, resized));
}

TEST(LanczosResize, RgbMatchesPilAndOpaqueRgba) {
  const auto rgb = rgba_fixture().slice(1, 0, 3).contiguous();
  const auto resized = resize_rgba_like_pil(rgb, /*width=*/5, /*height=*/4);
  expect_pil_pixels(
      resized,
      /*width=*/5,
      /*height=*/4,
      {255, 0, 29, 173, 112, 0,   0,   255, 0,   0,   134, 184, 28,  4,   255,
       246, 2, 22, 166, 93,  23,  31,  219, 70,  53,  156, 172, 107, 72,  215,
       225, 6, 12, 156, 69,  74,  79,  171, 160, 149, 184, 157, 210, 162, 124,
       209, 9, 5,  149, 50,  113, 116, 134, 230, 221, 206, 145, 255, 230, 54});
  const auto opaque =
      torch::cat({rgb, torch::full({1, 1, 2, 3}, 255, torch::kUInt8)}, 1);
  const auto resized_rgba =
      resize_rgba_like_pil(opaque, /*width=*/5, /*height=*/4);
  EXPECT_TRUE(torch::equal(resized, resized_rgba.slice(1, 0, 3)));
  EXPECT_TRUE(resized_rgba.slice(1, 3, 4).eq(255).all().item<bool>());
}

}  // namespace
}  // namespace lanczos
