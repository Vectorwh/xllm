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

#include <algorithm>
#include <cmath>
#include <numeric>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::kernel::mlu::bangc {
namespace {
using Stage = PagedAttentionStage;

// The 64-head prefill and decode kernels round intermediate attention scores
// and probabilities to BF16. Bound their L1 and L2 errors against FP64 using
// the standalone kernel's BF16 rounding baseline, including the rotary tail.
void check_bf16_pv(const torch::Tensor& actual, const torch::Tensor& expected) {
  const auto rounded = expected.to(torch::kBFloat16).to(torch::kFloat64);
  for (int32_t start : {0, 448}) {
    SCOPED_TRACE(::testing::Message() << "first channel=" << start);
    const auto ref = expected.slice(1, start);
    const auto baseline = rounded.slice(1, start) - ref;
    const auto error = actual.slice(1, start) - ref;
    for (int32_t order : {1, 2}) {
      const double limit =
          10.0 * std::max(baseline.norm(order).item<double>(),
                          1e-4 * ref.norm(order).item<double>());
      EXPECT_LE(error.norm(order).item<double>(), limit);
    }
  }
}

class PagedAttentionTest : public ::testing::Test {
 protected:
  const torch::Device device{torch::kPrivateUse1, 0};
  const torch::DeviceGuard guard{device};
  const torch::TensorOptions bf =
      torch::TensorOptions().dtype(torch::kBFloat16);
  PagedAttentionMetadata meta;
  torch::Tensor q, kv, swa, compressed, sink, output;
  int32_t ratio = 0;
  int32_t window = 128;

  void init(int32_t heads,
            int32_t query_rows,
            int32_t kv_rows,
            int32_t compression,
            int32_t requests = 3,
            int32_t columns = 1) {
    torch::manual_seed(42);
    ratio = compression;
    q = torch::randn({query_rows, heads, 512}, bf).to(device);
    kv = torch::randn({kv_rows, 512}, bf).to(device);
    swa = torch::randn({requests * columns * 128, 512}, bf).to(device);
    sink = torch::linspace(-3, 3, heads, torch::kFloat32).to(device);
    const auto ints =
        torch::TensorOptions().device(device).dtype(torch::kInt32);
    auto pages = torch::arange(requests * columns, torch::kInt32)
                     .flip({0})
                     .view({requests, columns});
    const auto table = [&](const torch::Tensor& ids) {
      auto backing = torch::full({requests, columns + 1}, -1, ints);
      auto view = backing.narrow(1, 0, columns);
      view.copy_(ids);
      return view;
    };
    meta.swa_table = table(pages);
    meta.swa_columns = torch::full({1}, columns, ints);
    meta.positions = torch::zeros({query_rows}, ints);
    meta.requests = torch::zeros_like(meta.positions);
    meta.valid_rows = torch::zeros({1}, ints);
    meta.prefix = torch::zeros({requests + 1}, ints);
    meta.begins =
        (torch::arange(requests, torch::kInt32) % 10 + 126).to(device);
    if (requests == 3) {
      meta.begins.copy_(torch::tensor({126, 3, 129}, torch::kInt32));
    }
    if (ratio > 0) {
      compressed =
          torch::randn({requests * columns * 128 / ratio, 512}, bf).to(device);
      meta.compressed_table = table(pages.flip({0}));
      meta.topk = torch::tensor({0, -1, 0}, ints).repeat({query_rows, 1});
    }
  }

  // Only the last three logical pages are resident. init() supplies eight
  // physical pages per request, including untouched guards.
  void map_table_tail(int32_t columns, int32_t stride) {
    const int32_t batch = meta.begins.numel();
    auto backing = torch::zeros({batch, stride}, meta.prefix.options());
    meta.swa_table = backing.narrow(1, 1, columns);
    meta.swa_table.fill_(-1);
    auto pages = torch::arange(batch, torch::kInt32).view({batch, 1}) * 8 +
                 torch::tensor({7, 2, 5}, torch::kInt32);
    meta.swa_table.slice(1, columns - 3).copy_(pages);
    meta.swa_columns.fill_(columns);
  }

  void layout(const std::vector<int32_t>& counts,
              const std::vector<int32_t>& queries) {
    std::vector<int32_t> prefix{0}, requests, positions;
    const auto begins = meta.begins.cpu();
    prefix.reserve(counts.size() + 1);
    requests.reserve(q.size(0));
    positions.reserve(q.size(0));
    for (int32_t count : counts) {
      prefix.push_back(prefix.back() + count);
    }
    for (int32_t row : queries) {
      const int32_t request = static_cast<int32_t>(
          std::upper_bound(prefix.begin(), prefix.end(), row) - prefix.begin() -
          1);
      requests.push_back(request);
      positions.push_back(begins[request].item<int32_t>() + row -
                          prefix[request]);
    }
    meta.valid_rows.fill_(queries.size());
    meta.prefix.copy_(torch::tensor(prefix, torch::kInt32));
    requests.resize(q.size(0), -1);
    positions.resize(q.size(0), -1);
    meta.requests.copy_(torch::tensor(requests, torch::kInt32));
    meta.positions.copy_(torch::tensor(positions, torch::kInt32));
  }

  void run(Stage stage) {
    output = fused_paged_sparsed_attention(
        stage, q, kv, swa, compressed, sink, meta, ratio, window);
  }

  void decode_layout(int32_t live) {
    std::vector<int32_t> counts(meta.begins.numel(), 0);
    std::fill_n(counts.begin(), live, 1);
    std::vector<int32_t> queries(live);
    std::iota(queries.begin(), queries.end(), 0);
    layout(counts, queries);
  }

  void check(Stage stage, const torch::Tensor& before) {
    auto expected_cache = before.clone();
    const auto query = q.cpu().to(torch::kFloat64);
    const auto current = kv.cpu();
    const auto result = output.cpu().to(torch::kFloat64);
    const auto sink_logits = sink.cpu().to(torch::kFloat64).unsqueeze(1);
    const auto prefix = meta.prefix.cpu();
    const auto table = meta.swa_table.cpu();
    const auto begins = meta.begins.cpu();
    const auto requests = meta.requests.cpu();
    const auto positions = meta.positions.cpu();
    const int32_t live = meta.valid_rows.item<int32_t>();
    const auto comp = ratio > 0 ? compressed.cpu() : torch::Tensor();
    const auto topk = ratio > 0 ? meta.topk.cpu() : torch::Tensor();
    const auto ctable =
        ratio > 0 ? meta.compressed_table.cpu() : torch::Tensor();
    const int32_t columns = meta.swa_columns.item<int32_t>();
    const auto slot = [&](int32_t request, int32_t pos) {
      return table[request][(pos / 128) % columns].item<int32_t>() * 128 +
             pos % 128;
    };
    for (int32_t row = 0; row < live; ++row) {
      const int32_t request = requests[row].item<int32_t>();
      const int32_t begin = begins[request].item<int32_t>();
      const int32_t offset = prefix[request].item<int32_t>();
      const int32_t position = positions[row].item<int32_t>();
      const int32_t first = std::max(
          0, stage == Stage::DRAFT ? begin - window : position - window + 1);
      const int32_t end =
          stage == Stage::DRAFT
              ? begin + prefix[request + 1].item<int32_t>() - offset
              : position + 1;
      std::vector<torch::Tensor> rows;
      rows.reserve(end - first + (ratio > 0 ? topk.size(1) : 0));
      for (int32_t t = first; t < end; ++t) {
        rows.push_back(t < begin ? before[slot(request, t)]
                                 : current[offset + t - begin]);
      }
      if (ratio > 0) {
        const int32_t page_size = 128 / ratio;
        for (int64_t j = 0; j < topk.size(1); ++j) {
          const int32_t idx = topk[row][j].item<int32_t>();
          if (idx >= 0 && idx < ctable.size(1) * page_size) {
            const int32_t page =
                ctable[request][idx / page_size].item<int32_t>();
            rows.emplace_back(comp[page * page_size + idx % page_size]);
          }
        }
      }
      const auto keys = torch::stack(rows).to(torch::kFloat64);
      const auto scores =
          torch::matmul(query[row], keys.t()) / std::sqrt(512.0);
      const auto weights = torch::cat({scores, sink_logits}, 1)
                               .softmax(1)
                               .slice(1, 0, keys.size(0));
      const auto expected = torch::matmul(weights, keys);
      SCOPED_TRACE(::testing::Message()
                   << "row=" << row << " heads=" << q.size(1));
      if (q.size(1) == 64 &&
          (stage == Stage::PREFILL || stage == Stage::DECODE)) {
        check_bf16_pv(result[row], expected);
      } else {
        ASSERT_TRUE(torch::allclose(result[row], expected, 0.012, 0.004))
            << "max_abs="
            << (result[row] - expected).abs().max().item<double>();
      }
    }
    EXPECT_TRUE(result.slice(0, live).eq(0).all().item<bool>());
    if (stage != Stage::DRAFT) {
      for (int32_t request = 0; request < begins.numel(); ++request) {
        for (int32_t row = prefix[request].item<int32_t>();
             row < prefix[request + 1].item<int32_t>();
             ++row) {
          const int32_t position = begins[request].item<int32_t>() + row -
                                   prefix[request].item<int32_t>();
          expected_cache[slot(request, position)].copy_(current[row]);
        }
      }
    }
    EXPECT_TRUE(torch::equal(swa.cpu(), expected_cache));
  }
};

TEST_F(PagedAttentionTest, WideDecodeReadsPreviousCommitWithStridedTables) {
  constexpr int32_t columns = 1025;
  constexpr int32_t position = (columns - 1) * 128;
  init(64, 2, 2, 0, 2, 8);
  map_table_tail(columns, 8192);
  meta.begins.copy_(
      torch::tensor({position, position + columns * 128}, torch::kInt32));
  for (int32_t step = 0; step < 2; ++step) {
    decode_layout(2);
    const auto before = swa.cpu();
    run(Stage::DECODE);
    check(Stage::DECODE, before);
    q.copy_(kv.unsqueeze(1) * 0.5);
    kv.neg_();
    meta.begins.add_(1);
  }
}

TEST_F(PagedAttentionTest, PackedStagesUseMappedPagesBeyond32K) {
  for (Stage stage : {Stage::PREFILL, Stage::VERIFY, Stage::DRAFT}) {
    SCOPED_TRACE(static_cast<int32_t>(stage));
    const int32_t count =
        stage == Stage::DRAFT ? 5 : (stage == Stage::VERIFY ? 6 : 8);
    init(64, count, count, stage == Stage::VERIFY ? 2 : 0, 1, 8);
    map_table_tail(257, 260);
    meta.begins.fill_(32768 + (stage == Stage::DRAFT ? 3 : -2));
    std::vector<int32_t> queries(count);
    std::iota(queries.begin(), queries.end(), 0);
    layout({count}, queries);
    const auto before = swa.cpu();
    run(stage);
    check(stage, before);
  }
}

TEST_F(PagedAttentionTest, LocalPrefillCommitsLastPageOfMillionTokenContext) {
  init(8, 2, 4, 0, 1, 8);
  map_table_tail(8192, 8195);
  meta.begins.fill_(1048576 - 4);
  layout({4}, {0, 3});
  auto before = swa.cpu();
  run(Stage::PREFILL);
  check(Stage::PREFILL, before);

  q = q.narrow(0, 0, 0);
  meta.requests = meta.requests.narrow(0, 0, 0);
  meta.positions = meta.positions.narrow(0, 0, 0);
  layout({4}, {});
  kv.neg_();
  before = swa.cpu();
  run(Stage::PREFILL);
  check(Stage::PREFILL, before);
}

TEST_F(PagedAttentionTest, LocalPrefillCommitsGlobalRowsAcrossRingBoundary) {
  init(16, 4, 7, 2);
  for (const std::vector<int32_t>& local :
       std::vector<std::vector<int32_t>>{{6, 1, 3}, {}}) {
    kv.neg_();
    layout({4, 0, 3}, local);
    const auto before = swa.cpu();
    run(Stage::PREFILL);
    check(Stage::PREFILL, before);
  }
}

TEST_F(PagedAttentionTest, DecodeReadsItsCommitAndDraftPreservesHistory) {
  init(8, 4, 4, 1);
  for (int32_t step = 0; step < 2; ++step) {
    layout({1, 1, 1}, {0, 1, 2});
    const auto before = swa.cpu();
    run(Stage::DECODE);
    check(Stage::DECODE, before);
    meta.begins.add_(1);
  }
  init(32, 6, 6, 0);
  layout({2, 0, 3}, {0, 1, 2, 3, 4});
  const auto before = swa.cpu();
  run(Stage::DRAFT);
  check(Stage::DRAFT, before);
}

TEST_F(PagedAttentionTest,
       PackedVerifyGraphRedistributesRowsAndRecoversFromEmpty) {
  for (int32_t heads : {8, 64}) {
    init(heads, 8, 8, heads == 64 ? 2 : 1, 3, 2);
    meta.topk =
        torch::tensor({63, 64, 63, -1}, meta.topk.options()).repeat({8, 1});
    layout({1, 6, 0}, {0, 1, 2, 3, 4, 5, 6});
    const auto original = swa.cpu();
    torch_mlu::MLUGraph graph;
    torch_mlu::synchronize();
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      run(Stage::VERIFY);
      graph.capture_end();
    }
    for (const auto& counts : std::vector<std::vector<int32_t>>{
             {1, 6, 0}, {5, 2, 0}, {0, 0, 0}, {2, 1, 4}}) {
      std::vector<int32_t> queries(counts[0] + counts[1] + counts[2]);
      std::iota(queries.begin(), queries.end(), 0);
      layout(counts, queries);
      meta.swa_columns.fill_(counts[0] == 5 ? 2 : 1);
      swa.copy_(original);
      graph.replay();
      torch_mlu::synchronize();
      check(Stage::VERIFY, original);
    }
  }
}

TEST_F(PagedAttentionTest, FullPrefillReadsHistoryBeforeCommittingAllKv) {
  for (int32_t compression : {0, 2}) {
    init(64, 36, 42, compression, 3, 2);
    if (compression == 0) {
      meta.swa_columns.fill_(1);
    } else {
      // Two full compressed tiles per query, with a subsequent query on the
      // same IPU.
      auto ids = torch::arange(256, torch::kInt32).remainder(128);
      ids = torch::cat({ids, torch::tensor({-1, 128}, torch::kInt32)});
      meta.topk = ids.repeat({q.size(0), 1}).to(device);
    }
    std::vector<int32_t> local(34);
    std::iota(local.begin(), local.end(), 0);
    local.back() = 41;
    for (const auto& queries : std::vector<std::vector<int32_t>>{local, {}}) {
      layout({32, 0, 10}, queries);
      const auto before = swa.cpu();
      kv.neg_();
      run(Stage::PREFILL);
      check(Stage::PREFILL, before);
    }
    q = q.narrow(0, 0, 0);
    meta.requests = meta.requests.narrow(0, 0, 0);
    meta.positions = meta.positions.narrow(0, 0, 0);
    if (compression > 0) {
      meta.topk = meta.topk.narrow(0, 0, 0);
    }
    const auto before = swa.cpu();
    kv.neg_();
    run(Stage::PREFILL);
    check(Stage::PREFILL, before);
  }
}

TEST_F(PagedAttentionTest, DecodeBucketsReadPreviousCommit) {
  for (int32_t compression : {0, 2}) {
    const std::vector<int32_t> capacities =
        compression == 0 ? std::vector<int32_t>{16, 17}
                         : std::vector<int32_t>{2, 4, 8, 16, 33};
    for (int32_t capacity : capacities) {
      SCOPED_TRACE(::testing::Message()
                   << "capacity=" << capacity << " ratio=" << compression);
      init(64, capacity, capacity, compression, capacity, 2);
      decode_layout(capacity);
      if (compression > 0) {
        meta.topk = torch::tensor({63, 64, 64, -1}, meta.topk.options())
                        .repeat({capacity, 1});
      }
      for (int32_t step = 0; step < 2; ++step) {
        const auto before = swa.cpu();
        run(Stage::DECODE);
        check(Stage::DECODE, before);
        meta.begins.add_(1);
        decode_layout(capacity);
        kv.neg_();
      }
    }
  }
}

TEST_F(PagedAttentionTest, DecodeGraphReplaysChangingLiveRows) {
  init(64, 33, 33, 1, 33, 2);
  decode_layout(33);
  torch_mlu::MLUGraph graph;
  torch_mlu::synchronize();
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    run(Stage::DECODE);
    graph.capture_end();
  }
  for (int32_t live : {33, 0, 2}) {
    decode_layout(live);
    const auto before = swa.cpu();
    graph.replay();
    torch_mlu::synchronize();
    check(Stage::DECODE, before);
  }
}

TEST_F(PagedAttentionTest, PackedVerifyCommitsBeforeReloadingNextTable) {
  init(64, 64, 64, 1, 64, 2);
  decode_layout(64);
  meta.topk = torch::full({64, 32}, -1, meta.topk.options());
  const auto before = swa.cpu();
  run(Stage::VERIFY);
  check(Stage::VERIFY, before);
}

TEST_F(PagedAttentionTest, PackedDraftPreservesCacheAcrossActiveLayouts) {
  init(64, 16, 16, 0, 4, 2);
  const auto before = swa.cpu();
  for (int32_t step = 0; step < 8; ++step) {
    const auto counts = step % 2 == 0 ? std::vector<int32_t>{3, 3, 3, 3}
                                      : std::vector<int32_t>{5, 0, 3, 0};
    std::vector<int32_t> queries(
        std::accumulate(counts.begin(), counts.end(), 0));
    std::iota(queries.begin(), queries.end(), 0);
    layout(counts, queries);
    kv.neg_();
    run(Stage::DRAFT);
    check(Stage::DRAFT, before);
  }
}
}  // namespace
}  // namespace xllm::kernel::mlu::bangc
