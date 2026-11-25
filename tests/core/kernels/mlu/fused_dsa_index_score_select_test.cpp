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

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
using Mode = ScoreSelectMode;
using Phase = ScoreSelectPhase;
class ScoreSelectTest : public ::testing::Test {
 protected:
  const torch::Device device{torch::kPrivateUse1, 0};
  const torch::DeviceGuard guard{device};
  const torch::TensorOptions bf =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const torch::TensorOptions ints = bf.dtype(torch::kInt32);
  ScoreSelectInputs in;
  torch::Tensor topk, blocks, scratch;
  void prepare(int32_t rows, int32_t batch, int32_t pages) {
    in.query = torch::ones({rows, 32, 128}, bf);
    in.weights = torch::ones({rows, 32}, bf);
    in.cache = torch::ones({batch * pages * 128, 128}, bf);
    in.table =
        torch::arange(batch * pages, ints).flip({0}).view({batch, pages});
    in.positions = torch::zeros({rows}, ints);
    in.requests = torch::zeros_like(in.positions);
    in.valid_rows = torch::full({1}, rows, ints);
    in.history = torch::full({batch}, pages * 128, ints);
    topk = torch::empty({rows, 512}, ints);
    blocks = torch::empty({rows, 2048}, ints);
  }
  void reserve(Mode mode, Phase phase) {
    scratch = torch::empty(
        {score_select_workspace(topk.size(0), in.table.size(0), mode, phase)},
        ints.dtype(torch::kUInt8));
  }
  void run(Mode mode, Phase phase, int32_t ratio = 1) {
    fused_dsa_index_score_select(
        in,
        ratio,
        mode,
        phase,
        topk,
        mode == Mode::SOURCE ? blocks : torch::Tensor(),
        scratch);
  }
  void expect_row(int32_t row, int32_t begin, int32_t end) {
    auto expected = torch::full({512}, -1, torch::kInt32);
    expected.slice(0, 512 - (end - begin))
        .copy_(torch::arange(begin, end, torch::kInt32));
    EXPECT_TRUE(torch::equal(topk[row].cpu(), expected)) << row;
  }
};
TEST_F(ScoreSelectTest, PackedFrontBackRowsRespectPagesAndCausalHistory) {
  prepare(4, 2, 6);
  in.requests.copy_(torch::tensor({0, 1, 0, 1}, ints));
  in.positions.copy_(torch::tensor({0, 5, 699, 999}, ints));
  in.history[1].fill_(9);
  torch::manual_seed(17);
  auto logical = torch::randn({768, 128}, torch::kFloat32).to(torch::kBFloat16);
  in.cache.view({12, 128, 128})
      .index_copy_(0,
                   in.table[0].to(torch::kInt64),
                   logical.view({6, 128, 128}).to(device));
  in.query.normal_();
  in.weights.normal_();
  reserve(Mode::NONE, Phase::PREFILL);
  run(Mode::NONE, Phase::PREFILL);
  expect_row(0, 0, 1);
  expect_row(1, 0, 6);
  expect_row(3, 0, 9);
  auto ids = topk[2].cpu().to(torch::kInt64);
  ASSERT_TRUE(ids.ge(0).all().item<bool>() && ids.lt(700).all().item<bool>());
  ASSERT_TRUE(ids.slice(0, 1).gt(ids.slice(0, 0, 511)).all().item<bool>());
  auto scores = (torch::matmul(in.query[2].cpu().to(torch::kFloat32),
                               logical.slice(0, 0, 700).to(torch::kFloat32).t())
                     .relu() *
                 in.weights[2].cpu().to(torch::kFloat32).unsqueeze(1))
                    .sum(0);
  auto expected = std::get<0>(scores.topk(512));
  auto actual = std::get<0>(scores.index_select(0, ids).sort(-1, true));
  EXPECT_LE((actual - expected).norm().item<double>() /
                expected.norm().item<double>(),
            0.003);
  EXPECT_GE(
      actual[-1].item<double>(),
      expected[-1].item<double>() -
          std::max(0.05,
                   0.03 * std::max(1., expected.abs().max().item<double>())));
}
TEST_F(ScoreSelectTest, CandidateHandoffMasksPartialBlockAndPinsLatest) {
  prepare(1, 1, 129);
  in.history.fill_(16385);
  in.positions.fill_(16384);
  // Block 2048 scores below every full block but must survive SOURCE.
  in.cache.slice(0, 0, 128).zero_();
  reserve(Mode::SOURCE, Phase::PREFILL);
  run(Mode::SOURCE, Phase::PREFILL);
  auto expected = torch::arange(2048, torch::kInt32);
  expected[-1].fill_(2048);
  ASSERT_TRUE(torch::equal(blocks[0].cpu(), expected));
  in.candidates = blocks;
  reserve(Mode::FILTER, Phase::PREFILL);
  run(Mode::FILTER, Phase::PREFILL);
  expect_row(0, 0, 512);
  blocks.fill_(-1);
  blocks[0][0].fill_(2048);
  run(Mode::FILTER, Phase::PREFILL);
  expect_row(0, 16384, 16385);
  in.candidates = torch::Tensor();
  in.history.zero_();
  reserve(Mode::SOURCE, Phase::PREFILL);
  run(Mode::SOURCE, Phase::PREFILL);
  EXPECT_TRUE(blocks.eq(-1).all().item<bool>());
  expect_row(0, 0, 0);
}
TEST_F(ScoreSelectTest, VerifyReplayUsesDeviceValidRowsAndRequestSlots) {
  prepare(8, 3, 1);
  reserve(Mode::NONE, Phase::DECODE);
  in.requests.copy_(torch::tensor({0, 2, 2, 2, 2, 2, 2, -777}, ints));
  in.positions.copy_(torch::tensor({1, 0, 1, 2, 3, 4, 5, -777}, ints));
  in.history[1].fill_(-777);
  in.table[1].fill_(-777);
  in.valid_rows.fill_(7);
  run(Mode::NONE, Phase::DECODE, 2);
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    run(Mode::NONE, Phase::DECODE, 2);
    graph.capture_end();
  }
  for (int32_t live : {7, 0, 2}) {
    in.valid_rows.fill_(live);
    if (live == 2) {
      in.positions[1].fill_(7);
    }
    topk.fill_(42);
    graph.replay();
    torch_mlu::synchronize();
    for (int32_t r = 0; r < 8; ++r) {
      int32_t count = r == 0 ? 1 : r / 2;
      if (live == 2 && r == 1) {
        count = 4;
      }
      expect_row(r, 0, r < live ? count : 0);
    }
  }
}

TEST_F(ScoreSelectTest, UniformDecodePreservesLaterRequestPositions) {
  prepare(/*rows=*/40, /*batch=*/40, /*pages=*/4);
  in.requests.copy_(torch::arange(40, ints));
  in.history.fill_(500);
  in.positions.fill_(100);
  in.positions.slice(0, 32).fill_(400);
  reserve(Mode::NONE, Phase::DECODE);
  run(Mode::NONE, Phase::DECODE);

  for (int32_t row = 0; row < 40; ++row) {
    expect_row(row, 0, row < 32 ? 101 : 401);
  }
}

TEST_F(ScoreSelectTest, StreamingDecodeUsesLiveRequestHistory) {
  constexpr int32_t kBatch = 66;
  constexpr int32_t kLive = 4 * kBatch;
  constexpr int32_t kRows = kLive + 6;
  prepare(kRows, kBatch, /*pages=*/4);
  auto rows = torch::arange(kLive, torch::kInt32);
  auto requests = rows.floor_divide(4);
  auto positions = 300 + rows.remainder(4);
  auto history = 297 + torch::arange(kBatch, torch::kInt32).remainder(11);
  history[0].fill_(0);
  in.requests.fill_(-777).slice(0, 0, kLive).copy_(requests);
  in.positions.fill_(-777).slice(0, 0, kLive).copy_(positions);
  in.history.copy_(history);
  in.valid_rows.fill_(kLive);
  reserve(Mode::SOURCE, Phase::DECODE);
  run(Mode::SOURCE, Phase::DECODE);

  auto visible = torch::zeros({kRows, 1}, torch::kInt32);
  visible.select(1, 0)
      .slice(0, 0, kLive)
      .copy_(torch::minimum(
          positions + 1, history.index_select(0, requests.to(torch::kInt64))));
  auto expected_tokens =
      (torch::arange(512, torch::kInt32) + visible - 512).clamp_min(-1);
  auto ids = torch::arange(2048, torch::kInt32);
  auto expected_blocks = torch::where(ids * 8 < visible, ids, -1);
  EXPECT_TRUE(torch::equal(topk.cpu(), expected_tokens));
  EXPECT_TRUE(torch::equal(blocks.cpu(), expected_blocks));
}

TEST_F(ScoreSelectTest, FilterVerifyUsesEachRowsCandidates) {
  constexpr int32_t kRows = 18;
  prepare(kRows, /*batch=*/3, /*pages=*/125);
  auto rows = torch::arange(kRows, torch::kInt32);
  auto positions = 15994 + rows.remainder(6);
  in.requests.copy_(rows.floor_divide(6));
  in.positions.copy_(positions);
  in.history.fill_(15997);
  auto candidates = torch::full({kRows, 2048}, -1, torch::kInt32);
  candidates.slice(0, 0, 12).slice(1, 0, 48).copy_(
      torch::arange(48, torch::kInt32));
  candidates.slice(0, 0, 12).select(1, 48).copy_(100 + rows.slice(0, 0, 12));
  candidates.slice(0, 0, 12).select(1, 49).fill_(1999);
  in.candidates = candidates.to(device);
  reserve(Mode::FILTER, Phase::DECODE);
  run(Mode::FILTER, Phase::DECODE);

  auto actual = topk.cpu();
  auto offsets = torch::arange(8, torch::kInt32);
  for (int32_t r = 0; r < kRows; ++r) {
    auto slots = candidates[r].unsqueeze(1) * 8 + offsets;
    auto selected = slots.masked_select(slots.ge(0) & slots.le(positions[r]) &
                                        slots.lt(15997));
    auto expected = torch::full({512}, -1, torch::kInt32);
    expected.slice(0, 512 - selected.numel()).copy_(selected);
    EXPECT_TRUE(torch::equal(actual[r], expected)) << r;
  }
}

TEST_F(ScoreSelectTest, SourceDecodeMergesLateWinnersAndPinsLatestBlock) {
  constexpr int32_t kHistory = 132000;
  constexpr int32_t kHotStart = 128000;
  constexpr int32_t kHotCount = 1024;
  constexpr int32_t kPages = (kHistory + 127) / 128;
  prepare(/*rows=*/1, /*batch=*/1, kPages);
  in.table[0].copy_(torch::arange(kPages, ints));
  in.history.fill_(kHistory);
  in.positions.fill_(kHistory - 1);
  in.cache.slice(0, kHotStart, kHotStart + kHotCount).fill_(2);
  in.cache.slice(0, kHistory - 8, kHistory).zero_();
  reserve(Mode::SOURCE, Phase::DECODE);
  run(Mode::SOURCE, Phase::DECODE);

  expect_row(0, kHotStart, kHotStart + 512);
  auto expected = torch::cat({
      torch::arange(2048 - kHotCount / 8 - 1, torch::kInt32),
      torch::arange(kHotStart / 8, (kHotStart + kHotCount) / 8, torch::kInt32),
      torch::tensor({kHistory / 8 - 1}, torch::kInt32),
  });
  EXPECT_TRUE(torch::equal(blocks[0].cpu(), expected));
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
