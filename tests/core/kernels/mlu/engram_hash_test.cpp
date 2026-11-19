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

#include "kernels/mlu/bangc/include/engram_hash.h"

#include <framework/core/MLUStream.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace xllm::kernel::mlu::bangc {
namespace {

class EngramHashTest : public ::testing::Test {
 protected:
  torch::Tensor run(const torch::Tensor& tokens,
                    const torch::Tensor& starts,
                    const torch::Tensor& history) {
    return engram_hash(tokens,
                       starts,
                       history,
                       map_,
                       multipliers_,
                       primes_,
                       offsets_,
                       /*pad_token=*/0);
  }
  void SetUp() override {
    const torch::DeviceGuard guard(device_);
    map_ = torch::tensor({0, 3, 1, 6, 2, 7, 5, 4}, torch::kInt64).to(device_);
    multipliers_ = torch::tensor({{int64_t{1}, int64_t{3}, int64_t{5}},
                                  {int64_t{306244774661193731},
                                   int64_t{90071992547409927},
                                   int64_t{162129586585337867}}},
                                 torch::kInt64)
                       .to(device_);
    primes_ = torch::tensor({{int64_t{7}, int64_t{11}},
                             {int64_t{4294967311}, int64_t{1099511627791}}},
                            torch::kInt64)
                  .to(device_);
    offsets_ = torch::tensor({{int64_t{0}, int64_t{7}},
                              {int64_t{0}, int64_t{4294967311}}},
                             torch::kInt64)
                   .to(device_);
  }
  torch::Tensor golden() const {
    // Worked integer examples include products above 2^52 and buckets above
    // 2^32; float conversion or int32 truncation changes these exact results.
    return torch::tensor({int64_t{5},          int64_t{10},
                          int64_t{4},          int64_t{11},
                          int64_t{0},          int64_t{7},
                          int64_t{6},          int64_t{15},
                          int64_t{0},          int64_t{7},
                          int64_t{4},          int64_t{11},
                          int64_t{3351248952}, int64_t{1103799468125},
                          int64_t{1417674800}, int64_t{1103789883502},
                          int64_t{0},          int64_t{4294967311},
                          int64_t{3871342647}, int64_t{1103790374960},
                          int64_t{0},          int64_t{4294967311},
                          int64_t{16777243},   int64_t{1103789883434}},
                         torch::kInt64)
        .view({2, 6, 2});
  }
  const torch::Device device_{torch::kPrivateUse1, 0};
  torch::Tensor map_, multipliers_, primes_, offsets_;
};

TEST_F(EngramHashTest,
       AllLayersPreserveIntegerHashesAcrossEmptyRequestsAndTiles) {
  const torch::DeviceGuard guard(device_);
  for (const int32_t copies : {1, 43}) {
    auto host = torch::tensor({2, 5, 129264, 7, -1, 1234}, torch::kInt64)
                    .repeat({copies});
    std::vector<int32_t> starts{0};
    starts.reserve(4 * copies + 1);
    for (int32_t copy = 0; copy < copies; ++copy) {
      for (const int32_t end : {0, 3, 3, 6}) {
        starts.emplace_back(copy * 6 + end);
      }
    }
    const auto q = torch::tensor(starts, torch::kInt32).to(device_);
    const auto history =
        torch::tensor({{0, 0}, {3, 4}, {0, 0}, {1, 2}}, torch::kInt64)
            .repeat({copies, 1})
            .to(device_);
    for (const auto dtype : {torch::kInt32, torch::kInt64}) {
      const auto actual = run(host.to(device_, dtype), q, history).cpu();
      EXPECT_TRUE(torch::equal(actual, golden().repeat({1, copies, 1})));
      EXPECT_TRUE(actual.select(0, 0).is_contiguous());
    }
  }
}

TEST_F(EngramHashTest, FullWidthTokenIdsClampWithoutAliasingImageSentinels) {
  const torch::DeviceGuard guard(device_);
  auto tokens = torch::tensor({int64_t{4294967298},
                               int64_t{4295096560},
                               std::numeric_limits<int64_t>::max(),
                               std::numeric_limits<int64_t>::min()},
                              torch::kInt64)
                    .to(device_);
  const auto clamped = torch::tensor({7, 7, 7, -1}, torch::kInt64).to(device_);
  const auto starts = torch::tensor({0, 4}, torch::kInt32).to(device_);
  const auto history = torch::tensor({{3, 4}}, torch::kInt64).to(device_);
  EXPECT_TRUE(torch::equal(run(tokens, starts, history).cpu(),
                           run(clamped, starts, history).cpu()));
}

TEST_F(EngramHashTest,
       GraphReplayReadsUpdatedTokensHistoryAndRequestBoundaries) {
  const torch::DeviceGuard guard(device_);
  auto tokens =
      torch::tensor({2, 5, 129264, 7, -1, 1234}, torch::kInt64).to(device_);
  auto starts = torch::tensor({0, 0, 3, 3, 6}, torch::kInt32).to(device_);
  auto history = torch::tensor({{0, 0}, {3, 4}, {0, 0}, {1, 2}}, torch::kInt64)
                     .to(device_);
  EXPECT_TRUE(torch::equal(run(tokens, starts, history).cpu(), golden()));
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  torch::Tensor output;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0));
    graph.capture_begin();
    output = run(tokens, starts, history);
    graph.capture_end();
  }
  graph.replay();
  EXPECT_TRUE(torch::equal(output.cpu(), golden()));
  tokens.copy_(torch::tensor({7, 1, 129265, 4, -2, 9}, torch::kInt64));
  starts.copy_(torch::tensor({0, 0, 2, 2, 6}, torch::kInt32));
  history.copy_(torch::tensor({{0, 0}, {1, 2}, {0, 0}, {6, 3}}, torch::kInt64));
  graph.replay();
  const auto expected =
      torch::tensor({int64_t{6},          int64_t{15},
                     int64_t{1},          int64_t{7},
                     int64_t{0},          int64_t{7},
                     int64_t{2},          int64_t{9},
                     int64_t{0},          int64_t{7},
                     int64_t{4},          int64_t{11},
                     int64_t{3871342647}, int64_t{1103790374960},
                     int64_t{1841299492}, int64_t{1103791849554},
                     int64_t{0},          int64_t{4294967311},
                     int64_t{2155872277}, int64_t{1103798239268},
                     int64_t{0},          int64_t{4294967311},
                     int64_t{16777243},   int64_t{1103789883434}},
                    torch::kInt64)
          .view({2, 6, 2});
  EXPECT_TRUE(torch::equal(output.cpu(), expected));
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
