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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/kernels/musa/mrope.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/ops_api.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm::kernel {
namespace {

TEST(MusaRopeTest, FusedQkNormPreservesStagedRoundingForBothPositionTypes) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  constexpr int64_t kHeadDim = 256;
  constexpr int64_t kRotaryDim = 64;
  constexpr double kEpsilon = 1e-6;
  const torch::Device target = device.unwrap();
  const torch::Tensor positions = torch::tensor({1, 0}, torch::kInt64);
  const torch::Tensor source =
      (torch::arange(2 * 4 * kHeadDim).remainder(17).to(torch::kFloat32) / 16 -
       0.5)
          .view({2, 4 * kHeadDim});
  for (const torch::ScalarType dtype : {torch::kBFloat16, torch::kFloat16}) {
    SCOPED_TRACE(dtype);
    const torch::Tensor input = source.to(dtype).to(target);
    const torch::Tensor weight =
        (torch::arange(kHeadDim).remainder(7).to(torch::kFloat32) / 16 - 0.25)
            .to(dtype)
            .to(target);
    torch::Tensor cache =
        (torch::arange(2 * kRotaryDim).remainder(11).to(torch::kFloat32) / 16)
            .view({2, kRotaryDim})
            .to(dtype)
            .to(target);
    for (const bool interleaved : {false, true}) {
      SCOPED_TRACE(interleaved);
      torch::Tensor expected = input.clone();
      torch::Tensor query = expected.narrow(1, 0, kHeadDim);
      torch::Tensor key = expected.narrow(1, 2 * kHeadDim, kHeadDim);
      musa::gemma_rms_norm(
          query, input.slice(1, 0, kHeadDim), weight, kEpsilon);
      musa::gemma_rms_norm(
          key, input.slice(1, 2 * kHeadDim, 3 * kHeadDim), weight, kEpsilon);
      torch::Tensor device_positions = positions.to(target);
      musa::partial_rotary_embedding_inplace(device_positions,
                                             query,
                                             key,
                                             cache,
                                             kHeadDim,
                                             kRotaryDim,
                                             /*is_neox=*/!interleaved);
      for (const torch::ScalarType position_type :
           {torch::kInt32, torch::kInt64}) {
        SCOPED_TRACE(position_type);
        torch::Tensor actual = input.clone();
        musa::fused_qk_norm_rope(actual,
                                 /*num_heads_q=*/1,
                                 /*num_heads_k=*/1,
                                 /*num_heads_v=*/1,
                                 kHeadDim,
                                 kEpsilon,
                                 weight,
                                 weight,
                                 cache,
                                 interleaved,
                                 positions.to(position_type).to(target),
                                 /*k_head_offset=*/2);
        EXPECT_TRUE(torch::equal(actual.cpu(), expected.cpu()));
      }
    }
  }
}

TEST(MusaRopeTest, MatchesReferenceWithAndWithoutKey) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const torch::Device target = device.unwrap();
  const torch::Tensor positions = torch::tensor({1, 0}, torch::kInt64);
  const torch::Tensor cache =
      torch::tensor({{1.0f, 0.5f, 0.25f, -0.5f}, {0.75f, 0.25f, -0.25f, 1.0f}});
  const torch::Tensor selected = cache.index_select(0, positions);
  const torch::Tensor cos = selected.slice(-1, 0, 2).unsqueeze(1);
  const torch::Tensor sin = selected.slice(-1, 2, 4).unsqueeze(1);
  const auto reference = [&](const torch::Tensor& input) {
    const torch::Tensor x = input.slice(-1, 0, 2);
    const torch::Tensor y = input.slice(-1, 2, 4);
    return torch::cat({x * cos - y * sin, y * cos + x * sin}, -1);
  };
  const torch::Tensor query =
      torch::arange(16, torch::kFloat32).view({2, 2, 4});
  const torch::Tensor key =
      torch::arange(8, torch::kFloat32).view({2, 1, 4}) + 0.5;
  for (const bool with_key : {false, true}) {
    SCOPED_TRACE(with_key);
    RotaryParams params;
    params.q = query.to(target);
    if (with_key) {
      params.k = key.to(target);
    }
    params.position_ids = positions.to(target);
    params.precomputed_cos_sin = cache.to(target);
    params.interleaved = false;
    apply_rotary(params);
    EXPECT_TRUE(torch::equal(params.q.cpu(), reference(query)));
    if (with_key) {
      EXPECT_TRUE(torch::equal(params.k.cpu(), reference(key)));
    } else {
      EXPECT_FALSE(params.k.defined());
    }
  }
}

TEST(MusaRopeTest, PartialMropeMatchesReferenceForStridedPositions) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  const torch::Tensor cache =
      torch::arange(48, torch::kFloat32).view({4, 12}) / 64;
  const torch::Tensor positions =
      torch::tensor({{0, 1, 2}, {3, 2, 1}, {1, 0, 3}}, torch::kInt64);
  const torch::Tensor query =
      torch::arange(96, torch::kFloat32).view({3, 2, 16}) / 16;
  const torch::Tensor key = query.slice(/*dim=*/1, 0, 1).clone() + 0.5;
  for (const bool interleaved : {false, true}) {
    SCOPED_TRACE(interleaved);
    const torch::ScalarType dtype =
        interleaved ? torch::kFloat16 : torch::kBFloat16;
    const torch::Tensor typed_cache = cache.to(dtype).to(torch::kFloat32);
    const torch::Tensor padded =
        torch::zeros({3, 6}, torch::kInt64).to(device.unwrap());
    torch::Tensor strided_positions = padded.slice(/*dim=*/1, 0, 6, 2);
    strided_positions.copy_(positions.to(device.unwrap()));
    torch::Tensor device_query = query.to(dtype).flatten(1).to(device.unwrap());
    torch::Tensor device_key = key.to(dtype).flatten(1).to(device.unwrap());
    const auto reference = [&](const torch::Tensor& input) {
      torch::Tensor expected = input.to(dtype).to(torch::kFloat32);
      const torch::Tensor source = expected.clone();
      for (int64_t frequency = 0; frequency < 6; ++frequency) {
        const int64_t axis = interleaved ? frequency % 3 : frequency / 2;
        const torch::Tensor rows = typed_cache.index_select(0, positions[axis]);
        const torch::Tensor cosine = rows.select(1, frequency).unsqueeze(1);
        const torch::Tensor sine = rows.select(1, frequency + 6).unsqueeze(1);
        expected.select(2, frequency)
            .copy_(source.select(2, frequency) * cosine -
                   source.select(2, frequency + 6) * sine);
        expected.select(2, frequency + 6)
            .copy_(source.select(2, frequency + 6) * cosine +
                   source.select(2, frequency) * sine);
      }
      return expected.to(dtype).flatten(1);
    };
    musa::partial_mrope_embedding(strided_positions,
                                  device_query,
                                  device_key,
                                  cache.to(dtype).to(device.unwrap()),
                                  /*head_size=*/16,
                                  /*rotary_dim=*/12,
                                  /*mrope_section=*/{2, 2, 2},
                                  interleaved);
    EXPECT_TRUE(
        torch::allclose(device_query.cpu(), reference(query), 0.01, 0.01));
    EXPECT_TRUE(torch::allclose(device_key.cpu(), reference(key), 0.01, 0.01));
  }
}

}  // namespace
}  // namespace xllm::kernel
