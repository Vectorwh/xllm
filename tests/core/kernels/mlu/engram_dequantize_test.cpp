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

#include "kernels/mlu/bangc/include/engram_dequantize.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <vector>

namespace xllm::kernel::mlu::bangc {
namespace {

torch::Tensor make_lut() {
  return torch::arange(256, torch::kInt16)
      .to(torch::kUInt8)
      .view(torch::kFloat8_e4m3fn)
      .to(torch::kFloat32);
}

torch::Tensor reference(const torch::Tensor& bytes,
                        const torch::Tensor& scales,
                        const torch::Tensor& hashes,
                        int64_t lo,
                        int64_t hi) {
  auto decoded = bytes.view(torch::kFloat8_e4m3fn).to(torch::kFloat32);
  auto factors = torch::exp2(scales.to(torch::kFloat32) - 127)
                     .repeat_interleave(bytes.size(1) / scales.size(1), -1);
  auto result = (decoded * factors).to(torch::kBFloat16);
  return result
      .masked_fill(
          (hashes.flatten().lt(lo) | hashes.flatten().ge(hi)).unsqueeze(1), 0)
      .view({hashes.size(0), hashes.size(1), bytes.size(1)});
}

torch::Tensor pack_rows(const torch::Tensor& bytes,
                        const torch::Tensor& scales,
                        int64_t tokens,
                        int64_t columns,
                        int64_t shards,
                        int64_t padding) {
  const int64_t width = bytes.size(1) + scales.size(1);
  const int64_t local_columns = (columns + shards - 1) / shards;
  auto canonical =
      torch::cat({bytes, scales}, 1).view({tokens, columns, width});
  auto storage = torch::full(
      {shards * tokens * local_columns, width + padding}, 255, torch::kUInt8);
  auto ranked = storage.view({shards, tokens, local_columns, width + padding});
  for (int64_t column = 0; column < columns; ++column) {
    ranked.select(0, column / local_columns)
        .select(1, column % local_columns)
        .narrow(1, 0, width)
        .copy_(canonical.select(1, column));
  }
  return storage;
}

TEST(EngramDequantizeTest, AllFp8CodesAndScaleExponents) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(false, 0));
  auto bytes = torch::arange(256, torch::kInt16)
                   .to(torch::kUInt8)
                   .unsqueeze(0)
                   .repeat({256, 1});
  auto scales =
      torch::arange(256, torch::kInt16).to(torch::kUInt8).view({256, 1});
  auto hashes = torch::arange(256, torch::kInt64).view({256, 1});
  auto actual = engram_dequantize(bytes.to(device),
                                  scales.to(device),
                                  hashes.to(device),
                                  make_lut().to(device),
                                  0,
                                  256);
  EXPECT_TRUE(torch::allclose(
      actual.cpu(), reference(bytes, scales, hashes, 0, 256), 0, 0, true));
}

TEST(EngramDequantizeTest, TailsOwnershipAndEmptyRows) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto lut = make_lut().to(device);
  for (const auto& shape : std::vector<std::tuple<int64_t, int64_t, int64_t>>{
           {0, 32, 1}, {1, 32, 1}, {33, 256, 8}, {7, 4128, 129}, {3, 65, 5}}) {
    const auto [rows, dim, scale_dim] = shape;
    auto bytes = torch::arange(rows * dim, torch::kInt64)
                     .remainder(127)
                     .to(torch::kUInt8)
                     .view({rows, dim});
    auto scales = torch::full({rows, scale_dim}, 127, torch::kUInt8);
    auto hashes = torch::arange(rows, torch::kInt64).view({rows, 1});
    auto actual = engram_dequantize(bytes.to(device),
                                    scales.to(device),
                                    hashes.to(device),
                                    lut,
                                    1,
                                    std::max<int64_t>(1, rows));
    EXPECT_TRUE(torch::allclose(
        actual.cpu(),
        reference(bytes, scales, hashes, 1, std::max<int64_t>(1, rows)),
        0,
        0,
        true));
  }
}

TEST(EngramDequantizeTest, ProvenOwnedRowsMatchMaskedPath) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto bytes = torch::arange(64 * 256, torch::kInt64)
                   .remainder(256)
                   .to(torch::kUInt8)
                   .view({64, 256})
                   .to(device);
  auto scales = torch::arange(64 * 8, torch::kInt64)
                    .remainder(256)
                    .to(torch::kUInt8)
                    .view({64, 8})
                    .to(device);
  auto hashes = torch::arange(64, torch::kInt64).view({8, 8}).to(device);
  auto lut = make_lut().to(device);
  auto masked = engram_dequantize(bytes, scales, hashes, lut, 0, 64);
  auto owned = engram_dequantize(bytes, scales, hashes, lut, 0, 64, true);
  EXPECT_TRUE(torch::allclose(owned.cpu(), masked.cpu(), 0, 0, true));
  EXPECT_TRUE(
      torch::allclose(owned.cpu(),
                      reference(bytes.cpu(), scales.cpu(), hashes.cpu(), 0, 64),
                      0,
                      0,
                      true));
}

TEST(EngramDequantizeTest, Group32NormalFactorsPreserveFp8SpecialValues) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto bytes = torch::arange(64 * 256, torch::kInt64)
                   .remainder(256)
                   .to(torch::kUInt8)
                   .view({64, 256});
  auto hashes = torch::arange(64, torch::kInt64).view({8, 8});
  auto lut = make_lut().to(device);
  for (int64_t exponent : {10, 127, 246, 255}) {
    auto scales = torch::full({64, 8}, exponent, torch::kUInt8);
    for (bool all_owned : {false, true}) {
      auto output = engram_dequantize(bytes.to(device),
                                      scales.to(device),
                                      hashes.to(device),
                                      lut,
                                      0,
                                      64,
                                      all_owned);
      EXPECT_TRUE(torch::allclose(
          output.cpu(), reference(bytes, scales, hashes, 0, 64), 0, 0, true));
    }
  }
}

TEST(EngramDequantizeTest, ReplayReadsLivePayloadScalesAndHashes) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(false, 0));
  auto bytes = torch::full({6, 256}, 56, torch::kUInt8).to(device);
  auto scales = torch::full({6, 8}, 127, torch::kUInt8).to(device);
  auto hashes = torch::arange(6, torch::kInt64).view({2, 3}).to(device);
  auto lut = make_lut().to(device);
  engram_dequantize(bytes, scales, hashes, lut, 1, 5);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor output;
  graph.capture_begin();
  output = engram_dequantize(bytes, scales, hashes, lut, 1, 5);
  graph.capture_end();
  for (int64_t round = 0; round < 3; ++round) {
    bytes.fill_(56 + round);
    scales.fill_(126 + round);
    hashes.copy_(
        torch::arange(6, torch::kInt64).add(round).view({2, 3}).to(device));
    graph.replay();
    EXPECT_TRUE(
        torch::equal(output.cpu(),
                     reference(bytes.cpu(), scales.cpu(), hashes.cpu(), 1, 5)));
  }
}

TEST(EngramDequantizeTest, PackedRowsPreserveRankOrderPaddingAndOwnership) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto lut = make_lut().to(device);
  for (const auto& shape :
       std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>{
           {0, 24, 256, 8},
           {3, 0, 256, 8},
           {7, 7, 32, 1},
           {1, 5, 65, 5},
           {9, 5, 4128, 129},
           {192, 24, 256, 8}}) {
    const auto [tokens, columns, dim, scale_dim] = shape;
    const int64_t rows = tokens * columns;
    auto bytes = torch::arange(rows * dim, torch::kInt64)
                     .view({rows, dim})
                     .add(torch::arange(rows, torch::kInt64).unsqueeze(1))
                     .remainder(256)
                     .to(torch::kUInt8);
    auto scales = torch::full({rows, scale_dim}, 127, torch::kUInt8);
    auto hashes = torch::arange(rows, torch::kInt64).view({tokens, columns});
    for (int64_t shards : {1, 3, 8, 32}) {
      auto storage =
          pack_rows(bytes, scales, tokens, columns, shards, 13).to(device);
      auto packed = storage.narrow(1, 0, dim + scale_dim);
      for (bool all_owned : {false, true}) {
        const int64_t lo = all_owned ? 0 : std::min<int64_t>(rows, 2);
        const int64_t hi = all_owned ? rows : std::max<int64_t>(lo, rows - 3);
        auto output = engram_dequantize_packed(packed,
                                               hashes.to(device),
                                               lut,
                                               dim,
                                               scale_dim,
                                               lo,
                                               hi,
                                               shards,
                                               all_owned);
        EXPECT_TRUE(torch::allclose(output.cpu(),
                                    reference(bytes, scales, hashes, lo, hi),
                                    0,
                                    0,
                                    true));
      }
    }
  }
}

TEST(EngramDequantizeTest, PackedScaleExtremesAndSpecialValues) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kTokens = 192;
  constexpr int64_t kColumns = 24;
  constexpr int64_t kRows = kTokens * kColumns;
  auto bytes = torch::arange(256, torch::kInt16)
                   .to(torch::kUInt8)
                   .unsqueeze(0)
                   .repeat({kRows, 1});
  auto hashes = torch::arange(kRows, torch::kInt64).view({kTokens, kColumns});
  auto lut = make_lut().to(device);
  for (int64_t exponent : {0, 10, 127, 246, 255}) {
    auto scales = torch::full({kRows, 8}, exponent, torch::kUInt8);
    auto packed = pack_rows(bytes, scales, kTokens, kColumns, 8, 0).to(device);
    auto output = engram_dequantize_packed(
        packed, hashes.to(device), lut, 256, 8, 0, kRows, 8, true);
    EXPECT_TRUE(torch::allclose(
        output.cpu(), reference(bytes, scales, hashes, 0, kRows), 0, 0, true));
  }
}

TEST(EngramDequantizeTest, PackedInt64OwnershipBounds) {
  const torch::Device device(torch::kPrivateUse1, 0);
  const int64_t start = std::numeric_limits<int64_t>::max() - 100;
  auto bytes = torch::arange(35 * 32, torch::kInt64)
                   .remainder(256)
                   .to(torch::kUInt8)
                   .view({35, 32});
  auto scales =
      torch::arange(35, torch::kInt64).to(torch::kUInt8).view({35, 1});
  auto hashes = torch::arange(35, torch::kInt64).add(start).view({7, 5});
  hashes.data_ptr<int64_t>()[3] = (start + 3) & ((int64_t{1} << 48) - 1);
  hashes.data_ptr<int64_t>()[5] = std::numeric_limits<int64_t>::min();
  hashes.data_ptr<int64_t>()[6] = std::numeric_limits<int64_t>::max();
  auto unpacked = engram_dequantize(bytes.to(device),
                                    scales.to(device),
                                    hashes.to(device),
                                    make_lut().to(device),
                                    start + 2,
                                    start + 31);
  EXPECT_TRUE(
      torch::allclose(unpacked.cpu(),
                      reference(bytes, scales, hashes, start + 2, start + 31),
                      0,
                      0,
                      true));
  auto packed = pack_rows(bytes, scales, 7, 5, 8, 13).to(device);
  auto output = engram_dequantize_packed(packed.narrow(1, 0, 33),
                                         hashes.to(device),
                                         make_lut().to(device),
                                         32,
                                         1,
                                         start + 2,
                                         start + 31,
                                         8);
  EXPECT_TRUE(
      torch::allclose(output.cpu(),
                      reference(bytes, scales, hashes, start + 2, start + 31),
                      0,
                      0,
                      true));
}

TEST(EngramDequantizeTest, PackedSelectedTokensPreserveRankedRowsAndPadding) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto lut = make_lut().to(device);
  for (const auto& shape :
       std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>>{
           {7, 7, 32, 1},
           {7, 24, 256, 8},
           {63, 24, 256, 8},
           {3, 5, 65, 5},
           {3, 3, 17000, 170}}) {
    const auto [tokens, columns, dim, scale_dim] = shape;
    SCOPED_TRACE(::testing::Message()
                 << "tokens=" << tokens << " columns=" << columns
                 << " dim=" << dim << " scales=" << scale_dim);
    const int64_t rows = tokens * columns;
    auto bytes =
        torch::arange(rows * dim, torch::kInt64)
            .view({rows, dim})
            .add(torch::arange(rows, torch::kInt64).mul(7).unsqueeze(1))
            .remainder(256)
            .to(torch::kUInt8);
    auto scales = torch::arange(rows * scale_dim, torch::kInt64)
                      .add(121)
                      .remainder(256)
                      .to(torch::kUInt8)
                      .view({rows, scale_dim});
    auto hashes = torch::arange(rows, torch::kInt64).view({tokens, columns});
    auto device_hashes = hashes.to(device);
    // Repeated, reversed and empty selections all address the original token
    // dimension, even when rank-major storage has padded heads. The longer
    // selection also crosses the short/long-batch boundary for group32.
    const std::vector<torch::Tensor> selections{
        torch::arange(tokens, torch::kInt64).flip({0}),
        torch::arange(tokens * 2 + 3, torch::kInt64)
            .mul(5)
            .add(tokens - 1)
            .remainder(tokens),
        torch::empty({0}, torch::kInt64)};
    for (int64_t shards : {1, 3, 8, 32}) {
      SCOPED_TRACE(::testing::Message() << "shards=" << shards);
      auto storage =
          pack_rows(bytes, scales, tokens, columns, shards, 13).to(device);
      auto packed = storage.narrow(1, 0, dim + scale_dim);
      for (bool all_owned : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "all_owned=" << all_owned);
        const int64_t lo = all_owned ? 0 : 2;
        const int64_t hi = all_owned ? rows : rows - 3;
        auto expected = reference(bytes, scales, hashes, lo, hi);
        auto full = engram_dequantize_packed(packed,
                                             device_hashes,
                                             lut,
                                             dim,
                                             scale_dim,
                                             lo,
                                             hi,
                                             shards,
                                             all_owned);
        EXPECT_TRUE(torch::allclose(full.cpu(), expected, 0, 0, true));
        for (const auto& indices : selections) {
          auto actual = engram_dequantize_packed(packed,
                                                 device_hashes,
                                                 lut,
                                                 dim,
                                                 scale_dim,
                                                 lo,
                                                 hi,
                                                 shards,
                                                 all_owned,
                                                 indices.to(device));
          ASSERT_EQ(actual.size(0), indices.numel());
          ASSERT_EQ(actual.size(1), columns);
          ASSERT_EQ(actual.size(2), dim);
          EXPECT_TRUE(torch::allclose(
              actual.cpu(), expected.index_select(0, indices), 0, 0, true));
        }
      }
    }
  }
}

TEST(EngramDequantizeTest, PackedSelectedTokensPreserveScale16SpecialValues) {
  const torch::Device device(torch::kPrivateUse1, 0);
  constexpr int64_t kTokens = 63;
  constexpr int64_t kColumns = 24;
  constexpr int64_t kRows = kTokens * kColumns;
  auto bytes = torch::arange(kRows * 256, torch::kInt64)
                   .view({kRows, 256})
                   .add(torch::arange(kRows, torch::kInt64).unsqueeze(1))
                   .remainder(256)
                   .to(torch::kUInt8);
  auto hashes = torch::arange(kRows, torch::kInt64).view({kTokens, kColumns});
  auto indices = torch::arange(kTokens * 2 + 3, torch::kInt64)
                     .mul(5)
                     .add(kTokens - 1)
                     .remainder(kTokens);
  auto device_hashes = hashes.to(device);
  auto device_indices = indices.to(device);
  auto lut = make_lut().to(device);
  // The mapped output exceeds 32 full tiles. Normal exponents force the
  // BF16 scaling cycle, while all FP8 codes retain NaN and signed-zero cases.
  for (int64_t exponent : {10, 127, 246}) {
    auto scales = torch::full({kRows, 8}, exponent, torch::kUInt8);
    for (int64_t shards : {3, 8}) {
      auto packed =
          pack_rows(bytes, scales, kTokens, kColumns, shards, 0).to(device);
      for (bool all_owned : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "exponent=" << exponent << " shards=" << shards
                     << " all_owned=" << all_owned);
        const int64_t lo = all_owned ? 0 : 2;
        const int64_t hi = all_owned ? kRows : kRows - 3;
        auto output = engram_dequantize_packed(packed,
                                               device_hashes,
                                               lut,
                                               /*dim=*/256,
                                               /*scale_dim=*/8,
                                               lo,
                                               hi,
                                               shards,
                                               all_owned,
                                               device_indices);
        EXPECT_TRUE(torch::allclose(
            output.cpu(),
            reference(bytes, scales, hashes, lo, hi).index_select(0, indices),
            0,
            0,
            true));
      }
    }
  }
}

TEST(EngramDequantizeTest, PackedGraphReplayReadsLiveTokenSelection) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(false, 0));
  constexpr int64_t kTokens = 7;
  constexpr int64_t kColumns = 5;
  constexpr int64_t kRows = kTokens * kColumns;
  auto bytes = torch::arange(kRows * 256, torch::kInt64)
                   .view({kRows, 256})
                   .add(torch::arange(kRows, torch::kInt64).unsqueeze(1))
                   .remainder(127)
                   .to(torch::kUInt8);
  auto scales = torch::full({kRows, 8}, 127, torch::kUInt8);
  auto hashes = torch::arange(kRows, torch::kInt64).view({kTokens, kColumns});
  auto storage = pack_rows(bytes, scales, kTokens, kColumns, 8, 13).to(device);
  auto packed = storage.narrow(1, 0, 264);
  auto device_hashes = hashes.to(device);
  auto lut = make_lut().to(device);
  auto indices = torch::arange(17, torch::kInt64).remainder(kTokens);
  auto device_indices = indices.to(device);
  auto run = [&] {
    return engram_dequantize_packed(packed,
                                    device_hashes,
                                    lut,
                                    /*dim=*/256,
                                    /*scale_dim=*/8,
                                    /*table_start=*/1,
                                    kRows - 1,
                                    /*head_shard_count=*/8,
                                    /*all_owned=*/false,
                                    device_indices);
  };
  run();
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  graph.capture_begin();
  auto output = run();
  graph.capture_end();
  for (int64_t round = 0; round < 3; ++round) {
    indices = torch::arange(17, torch::kInt64)
                  .mul(round + 1)
                  .add(kTokens - 1 - round)
                  .remainder(kTokens);
    device_indices.copy_(indices.to(device));
    graph.replay();
    EXPECT_TRUE(torch::allclose(
        output.cpu(),
        reference(bytes, scales, hashes, 1, kRows - 1).index_select(0, indices),
        0,
        0,
        true));
  }
}

TEST(EngramDequantizeTest, PackedGraphReplayReadsLiveRankedRowsAndHashes) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(false, 0));
  constexpr int64_t kTokens = 192;
  constexpr int64_t kColumns = 24;
  constexpr int64_t kRows = kTokens * kColumns;
  auto bytes = torch::full({kRows, 256}, 56, torch::kUInt8);
  auto scales = torch::full({kRows, 8}, 127, torch::kUInt8);
  auto hashes = torch::arange(kRows, torch::kInt64).view({kTokens, kColumns});
  auto storage = pack_rows(bytes, scales, kTokens, kColumns, 8, 13).to(device);
  auto packed = storage.narrow(1, 0, 264);
  auto device_hashes = hashes.to(device);
  auto lut = make_lut().to(device);
  engram_dequantize_packed(packed, device_hashes, lut, 256, 8, 1, kRows - 1, 8);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor output;
  graph.capture_begin();
  output = engram_dequantize_packed(
      packed, device_hashes, lut, 256, 8, 1, kRows - 1, 8);
  graph.capture_end();
  for (int64_t round = 0; round < 4; ++round) {
    bytes.fill_(56 + round);
    scales.fill_(round == 0 ? 0 : 126 + round);
    hashes = torch::arange(kRows, torch::kInt64)
                 .add(round)
                 .view({kTokens, kColumns});
    storage.copy_(
        pack_rows(bytes, scales, kTokens, kColumns, 8, 13).to(device));
    device_hashes.copy_(hashes.to(device));
    graph.replay();
    EXPECT_TRUE(torch::allclose(output.cpu(),
                                reference(bytes, scales, hashes, 1, kRows - 1),
                                0,
                                0,
                                true));
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu::bangc
