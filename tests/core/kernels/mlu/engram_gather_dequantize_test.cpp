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

#include "kernels/mlu/bangc/include/engram_gather_dequantize.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

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
  auto output = torch::zeros({hashes.numel(), bytes.size(1)}, torch::kBFloat16);
  auto flat_hashes = hashes.flatten();
  for (int64_t row = 0; row < flat_hashes.numel(); ++row) {
    const int64_t hash = flat_hashes[row].item<int64_t>();
    if (hash < lo || hash >= hi) {
      continue;
    }
    const int64_t table_row = hash - lo;
    auto decoded =
        bytes[table_row].view(torch::kFloat8_e4m3fn).to(torch::kFloat32);
    auto factors = torch::exp2(scales[table_row].to(torch::kFloat32) - 127)
                       .repeat_interleave(bytes.size(1) / scales.size(1), -1);
    output[row].copy_((decoded * factors).to(torch::kBFloat16));
  }
  return output.view({hashes.size(0), hashes.size(1), bytes.size(1)});
}

void expect_match(const torch::Tensor& actual, const torch::Tensor& expected) {
  auto cpu = actual.cpu();
  EXPECT_TRUE(torch::allclose(cpu, expected, 0, 0, true));
  // Floating equality does not distinguish the positive-zero ownership mask
  // from decoded negative zero, so also compare every zero's representation.
  auto zeros = expected.eq(0);
  EXPECT_TRUE(torch::equal(cpu.view(torch::kInt16).masked_select(zeros),
                           expected.view(torch::kInt16).masked_select(zeros)));
}

TEST(EngramGatherDequantizeTest, AllFp8CodesAndScaleExponents) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto bytes = torch::arange(256, torch::kInt16)
                   .to(torch::kUInt8)
                   .unsqueeze(0)
                   .repeat({256, 1});
  auto hashes = torch::arange(256, torch::kInt64).view({256, 1});
  for (int64_t scale_dim : {1, 8}) {
    auto scales = torch::arange(256, torch::kInt16)
                      .to(torch::kUInt8)
                      .view({256, 1})
                      .repeat({1, scale_dim});
    auto actual = engram_gather_dequantize(bytes.to(device),
                                           scales.to(device),
                                           hashes.to(device),
                                           make_lut().to(device),
                                           0,
                                           256);
    expect_match(actual, reference(bytes, scales, hashes, 0, 256));
  }
}

TEST(EngramGatherDequantizeTest, TailsForeignHashesAndEmptyTables) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto lut = make_lut().to(device);
  for (const auto& shape :
       std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>>{
           {0, 3, 65, 5, 7},
           {3, 0, 37, 1, 7},
           {7, 3, 65, 5, 7},
           {33, 5, 37, 1, 7},
           {3, 3, 17000, 170, 7},
           {17, 3, 32, 1, 0},
           {192, 3, 256, 8, 129}}) {
    const auto [tokens, heads, dim, scale_dim, rows] = shape;
    auto bytes = torch::arange(rows * dim, torch::kInt64)
                     .remainder(256)
                     .to(torch::kUInt8)
                     .view({rows, dim});
    auto scales = torch::arange(rows * scale_dim, torch::kInt64)
                      .remainder(256)
                      .to(torch::kUInt8)
                      .view({rows, scale_dim});
    auto hashes = torch::arange(tokens * heads, torch::kInt64)
                      .remainder(rows + 2)
                      .add(10)
                      .view({tokens, heads});
    if (hashes.numel() > 0) {
      hashes.flatten()[0] = std::numeric_limits<int64_t>::min();
    }
    if (hashes.numel() > 1) {
      hashes.flatten()[1] = std::numeric_limits<int64_t>::max();
    }
    auto actual = engram_gather_dequantize(bytes.to(device),
                                           scales.to(device),
                                           hashes.to(device),
                                           lut,
                                           11,
                                           11 + rows);
    expect_match(actual, reference(bytes, scales, hashes, 11, 11 + rows));
  }
}

TEST(EngramGatherDequantizeTest, Int64BoundsDoNotWrapIntoOwnedRows) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto lut = make_lut().to(device);
  for (int64_t start : {std::numeric_limits<int64_t>::max() - 7,
                        (static_cast<int64_t>(1) << 48) - 3}) {
    const int64_t end = start + 7;
    for (int64_t dim : {1, 256}) {
      auto bytes = torch::arange(7 * dim, torch::kInt64)
                       .remainder(256)
                       .to(torch::kUInt8)
                       .view({7, dim});
      auto scales = torch::full({7, dim == 1 ? 1 : 8}, 127, torch::kUInt8);
      for (const auto& hashes : std::vector<torch::Tensor>{
               torch::tensor({std::numeric_limits<int64_t>::min(),
                              start,
                              end - 1,
                              end,
                              start & ((static_cast<int64_t>(1) << 48) - 1),
                              static_cast<int64_t>(-1)},
                             torch::kInt64)
                   .view({2, 3}),
               torch::arange(7, torch::kInt64).add(start).view({7, 1})}) {
        auto actual = engram_gather_dequantize(bytes.to(device),
                                               scales.to(device),
                                               hashes.to(device),
                                               lut,
                                               start,
                                               end);
        expect_match(actual, reference(bytes, scales, hashes, start, end));
      }
    }
  }
}

TEST(EngramGatherDequantizeTest, SelectedTokensMatchGatherThenIndexSelect) {
  const torch::Device device(torch::kPrivateUse1, 0);
  auto lut = make_lut().to(device);
  constexpr int64_t kTableStart = 11;
  for (const auto& shape :
       std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>>{
           {7, 7, 32, 1, 17},
           {7, 24, 256, 8, 17},
           {63, 24, 256, 8, 17},
           {3, 5, 65, 5, 17},
           {3, 3, 17000, 170, 17},
           {7, 5, 256, 8, 0}}) {
    const auto [tokens, columns, dim, scale_dim, table_rows] = shape;
    SCOPED_TRACE(::testing::Message()
                 << "tokens=" << tokens << " columns=" << columns
                 << " dim=" << dim << " scales=" << scale_dim
                 << " table_rows=" << table_rows);
    auto bytes =
        torch::arange(table_rows * dim, torch::kInt64)
            .view({table_rows, dim})
            .add(torch::arange(table_rows, torch::kInt64).mul(7).unsqueeze(1))
            .remainder(256)
            .to(torch::kUInt8);
    auto scales = torch::arange(table_rows * scale_dim, torch::kInt64)
                      .add(121)
                      .remainder(256)
                      .to(torch::kUInt8)
                      .view({table_rows, scale_dim});
    auto hashes = torch::arange(tokens * columns, torch::kInt64)
                      .remainder(table_rows + 2)
                      .add(kTableStart - 1)
                      .view({tokens, columns});
    hashes.flatten()[0] = std::numeric_limits<int64_t>::min();
    hashes.flatten()[1] = std::numeric_limits<int64_t>::max();
    auto device_bytes = bytes.to(device);
    auto device_scales = scales.to(device);
    auto device_hashes = hashes.to(device);
    auto expected =
        reference(bytes, scales, hashes, kTableStart, kTableStart + table_rows);
    auto full = engram_gather_dequantize(device_bytes,
                                         device_scales,
                                         device_hashes,
                                         lut,
                                         kTableStart,
                                         kTableStart + table_rows);
    expect_match(full, expected);
    for (const auto& indices : std::vector<torch::Tensor>{
             torch::arange(tokens, torch::kInt64).flip({0}),
             torch::arange(tokens * 2 + 3, torch::kInt64)
                 .mul(5)
                 .add(tokens - 1)
                 .remainder(tokens),
             torch::empty({0}, torch::kInt64)}) {
      auto output = engram_gather_dequantize(device_bytes,
                                             device_scales,
                                             device_hashes,
                                             lut,
                                             kTableStart,
                                             kTableStart + table_rows,
                                             indices.to(device));
      ASSERT_EQ(output.size(0), indices.numel());
      ASSERT_EQ(output.size(1), columns);
      ASSERT_EQ(output.size(2), dim);
      expect_match(output, expected.index_select(0, indices));
    }
  }
}

TEST(EngramGatherDequantizeTest, GraphReplayReadsLiveTokenSelection) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(false, 0));
  auto bytes = torch::arange(7 * 256, torch::kInt64)
                   .view({7, 256})
                   .add(torch::arange(7, torch::kInt64).unsqueeze(1))
                   .remainder(127)
                   .to(torch::kUInt8);
  auto scales = torch::full({7, 8}, 127, torch::kUInt8);
  auto hashes =
      torch::arange(35, torch::kInt64).remainder(9).add(10).view({7, 5});
  auto device_bytes = bytes.to(device);
  auto device_scales = scales.to(device);
  auto device_hashes = hashes.to(device);
  auto lut = make_lut().to(device);
  auto indices = torch::arange(17, torch::kInt64).remainder(7);
  auto device_indices = indices.to(device);
  auto run = [&] {
    return engram_gather_dequantize(device_bytes,
                                    device_scales,
                                    device_hashes,
                                    lut,
                                    /*table_start=*/11,
                                    /*table_end=*/18,
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
                  .add(6 - round)
                  .remainder(7);
    device_indices.copy_(indices.to(device));
    graph.replay();
    expect_match(
        output,
        reference(bytes, scales, hashes, 11, 18).index_select(0, indices));
  }
}

TEST(EngramGatherDequantizeTest, ReplayReadsLiveTableScalesAndHashes) {
  const torch::Device device(torch::kPrivateUse1, 0);
  torch_mlu::mlu::MLUStreamGuard stream(torch_mlu::getStreamFromPool(false, 0));
  auto bytes = torch::full({7, 256}, 56, torch::kUInt8).to(device);
  auto scales = torch::full({7, 8}, 127, torch::kUInt8).to(device);
  auto hashes = torch::arange(6, torch::kInt64).add(11).view({2, 3}).to(device);
  auto lut = make_lut().to(device);
  engram_gather_dequantize(bytes, scales, hashes, lut, 11, 18);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor output;
  graph.capture_begin();
  output = engram_gather_dequantize(bytes, scales, hashes, lut, 11, 18);
  graph.capture_end();
  void* const output_address = output.data_ptr();
  for (int64_t round = 0; round < 128; ++round) {
    bytes.fill_(round);
    scales.fill_(round);
    hashes.copy_(torch::arange(6, torch::kInt64)
                     .add(10 + round % 4)
                     .view({2, 3})
                     .to(device));
    graph.replay();
    EXPECT_EQ(output.data_ptr(), output_address);
    expect_match(output,
                 reference(bytes.cpu(), scales.cpu(), hashes.cpu(), 11, 18));
  }
}

}  // namespace
}  // namespace xllm::kernel::mlu::bangc
