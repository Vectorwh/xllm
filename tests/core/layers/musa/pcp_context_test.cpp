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
#include "core/layers/musa/pcp_context.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <optional>
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#include <utility>

#include "core/framework/parallel_state/musa/musa_custom_all_reduce.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/layers/common/attention_metadata.h"
#include "core/util/env_var.h"

namespace xllm::layer::musa_pcp {
namespace {

AttentionMetadata make_global_metadata(int32_t q_rows, int32_t history) {
  AttentionMetadata m{};
  m.is_prefill = true;
  m.q_seq_lens_vec = {q_rows};
  m.kv_seq_lens_vec = {history + q_rows};
  const auto int32 = torch::TensorOptions().dtype(torch::kInt32);
  m.q_cu_seq_lens = torch::tensor({0, q_rows}, int32);
  m.kv_cu_seq_lens = torch::tensor({history, history + q_rows}, int32);
  m.slot_mapping = torch::arange(q_rows).to(torch::kInt32);
  m.has_initial_states =
      torch::zeros({1}, torch::TensorOptions().dtype(torch::kBool));
  return m;
}

}  // namespace

TEST(MusaPcpCostDeathTest, DefaultsRemainUnchanged) {
  EXPECT_EXIT(
      {
        unsetenv("XLLM_MUSA_PCP_COST_PAIR");
        unsetenv("XLLM_MUSA_PCP_COST_OTHER_KV_ROW");
        const auto& costs = util::get_musa_pcp_costs();
        std::_Exit(costs[0] == 9.4e-6 && costs[1] == 0.04 ? 0 : 1);
      },
      testing::ExitedWithCode(0),
      "");
}

TEST(MusaPcpCostDeathTest, OverridesChangeSplitAndStayCached) {
  EXPECT_EXIT(
      {
        setenv("XLLM_MUSA_PCP_COST_PAIR", "1e-3", /*overwrite=*/1);
        setenv("XLLM_MUSA_PCP_COST_OTHER_KV_ROW", "0", /*overwrite=*/1);
        const auto& costs = util::get_musa_pcp_costs();
        if (costs[0] != 1e-3 || costs[1] != 0) {
          std::_Exit(1);
        }
        setenv("XLLM_MUSA_PCP_COST_PAIR", "0", /*overwrite=*/1);
        constexpr int32_t kRows = 4096;
        AttentionMetadata metadata = make_global_metadata(kRows, /*history=*/0);
        ProcessGroup cp(
            /*rank=*/0, /*world_size=*/2, torch::Device(torch::kCPU));
        ParallelArgs args(/*rank=*/0, /*world_size=*/2, &cp);
        args.cp_group_ = &cp;
        std::optional<Context> context;
        build_context(metadata, args, kRows, &context);
        std::_Exit(context.has_value() && context->row_offsets[1] == 2560 &&
                           util::get_musa_pcp_costs()[0] == 1e-3
                       ? 0
                       : 1);
      },
      testing::ExitedWithCode(0),
      "");
}

TEST(MusaPcpCostDeathTest, RejectsInvalidCost) {
  EXPECT_DEATH(
      {
        setenv("XLLM_MUSA_PCP_COST_PAIR", "nan", /*overwrite=*/1);
        util::get_musa_pcp_costs();
      },
      "XLLM_MUSA_PCP_COST_PAIR must be a finite non-negative double");
}

TEST(MusaPcpCostDeathTest, RejectsMismatchedCpCostsBeforeCustomArInit) {
  EXPECT_DEATH(
      {
        unsetenv("XLLM_MUSA_PCP_COST_PAIR");
        unsetenv("XLLM_MUSA_PCP_COST_OTHER_KV_ROW");
        setenv("XLLM_MUSA_PCP_CUSTOM_AR", "false", /*overwrite=*/1);
        auto store = c10::make_intrusive<c10d::HashStore>();
        store->set("xllm_pcp/costs/1",
                   std::vector<uint8_t>(2 * sizeof(double), 0));
        MusaCustomAllReduce reducer(/*rank=*/0,
                                    /*world_size=*/2,
                                    "cp_group",
                                    torch::Device(torch::kCPU),
                                    store);
      },
      "MUSA PCP costs must match on every CP rank");
}

// build_context is pure metadata math (no collective): a plain CPU
// ProcessGroup with two ranks is enough to exercise the split and the
// per-rank metadata slicing.
TEST(MusaPcpContextTest, SplitAndLocalMetadataKeepCachedPrefixInMaxSeqLen) {
  constexpr int32_t kRows = 1024;
  constexpr int32_t kHistory = 300;
  AttentionMetadata m = make_global_metadata(kRows, kHistory);

  ProcessGroup cp0(0, /*world_size=*/2, torch::Device(torch::kCPU));
  ParallelArgs a0(0, /*world_size=*/2, &cp0);
  a0.cp_group_ = &cp0;
  std::optional<Context> ctx0;
  build_context(m, a0, kRows, &ctx0);

  ProcessGroup cp1(1, /*world_size=*/2, torch::Device(torch::kCPU));
  ParallelArgs a1(1, /*world_size=*/2, &cp1);
  a1.cp_group_ = &cp1;
  std::optional<Context> ctx1;
  build_context(m, a1, kRows, &ctx1);

  ASSERT_TRUE(ctx0.has_value());
  ASSERT_TRUE(ctx1.has_value());
  const Context& c0 = *ctx0;
  const Context& c1 = *ctx1;

  // Same split on both ranks, 256-aligned and strictly inside the prompt.
  ASSERT_EQ(c0.row_offsets, c1.row_offsets);
  ASSERT_EQ(c0.row_offsets.size(), 3u);
  EXPECT_EQ(c0.row_offsets.front(), 0);
  EXPECT_EQ(c0.row_offsets.back(), kRows);
  const int32_t first = c0.row_offsets[1];
  EXPECT_EQ(first % 256, 0);
  EXPECT_GT(first, 0);
  EXPECT_LT(first, kRows);
  // A plain ProcessGroup is not a personalised custom-AR group.
  EXPECT_FALSE(c0.use_custom_ar);
  EXPECT_TRUE(c0.replicate_final_state);

  // rank0 owns [0, first): stays a plain prefill, K/V pre-written, and the key
  // length bound must include the cached prefix (regression: was `end`).
  EXPECT_TRUE(c0.local_metadata.is_prefill);
  EXPECT_FALSE(c0.local_metadata.is_chunked_prefill);
  EXPECT_TRUE(c0.local_metadata.kv_cache_written);
  EXPECT_EQ(c0.local_metadata.max_query_len, first);
  EXPECT_EQ(c0.local_metadata.max_seq_len, kHistory + first);
  EXPECT_EQ(c0.local_metadata.total_kv_len, kHistory + first);
  ASSERT_EQ(c0.local_metadata.kv_seq_lens_vec.size(), 1u);
  EXPECT_EQ(c0.local_metadata.kv_seq_lens_vec[0], kHistory + first);
  EXPECT_EQ(c0.local_metadata.slot_mapping.numel(), first);

  // rank1 owns [first, kRows): continues the prefix and carries full history.
  EXPECT_FALSE(c1.local_metadata.is_prefill);
  EXPECT_TRUE(c1.local_metadata.is_chunked_prefill);
  EXPECT_EQ(c1.local_metadata.max_query_len, kRows - first);
  EXPECT_EQ(c1.local_metadata.max_seq_len, kHistory + kRows);
  ASSERT_EQ(c1.local_metadata.kv_seq_lens_vec.size(), 1u);
  EXPECT_EQ(c1.local_metadata.kv_seq_lens_vec[0], kHistory + kRows);
  EXPECT_EQ(c1.local_metadata.slot_mapping.numel(), kRows - first);
}

TEST(MusaPcpContextTest, NoContextWhenNotPcpPrefill) {
  constexpr int32_t kRows = 1024;
  AttentionMetadata m = make_global_metadata(kRows, 300);

  // No CP group.
  ParallelArgs no_cp(0, /*world_size=*/1, nullptr);
  std::optional<Context> no_cp_ctx;
  build_context(m, no_cp, kRows, &no_cp_ctx);
  EXPECT_FALSE(no_cp_ctx.has_value());

  ProcessGroup cp(0, /*world_size=*/2, torch::Device(torch::kCPU));
  ParallelArgs a(0, /*world_size=*/2, &cp);
  a.cp_group_ = &cp;

  // Decode (neither prefill nor chunked prefill).
  AttentionMetadata decode = m;
  decode.is_prefill = false;
  decode.is_chunked_prefill = false;
  std::optional<Context> decode_ctx;
  build_context(decode, a, kRows, &decode_ctx);
  EXPECT_FALSE(decode_ctx.has_value());

  // Spec verify.
  AttentionMetadata spec = m;
  spec.is_spec_verify = true;
  std::optional<Context> spec_ctx;
  build_context(spec, a, kRows, &spec_ctx);
  EXPECT_FALSE(spec_ctx.has_value());
}

// A 2-rank CPU stand-in for the CP group used by gather_rows: the local rank
// supplies its own (already padded) rows as the input, and the peer rank's
// already-padded rows are fixed at construction. allgather_base_sync then
// reproduces the real stack without a backend or device.
class StubGatherGroup final : public ProcessGroup {
 public:
  StubGatherGroup(int32_t rank, torch::Tensor peer_rows)
      : ProcessGroup(rank, /*world_size=*/2, torch::Device(torch::kCPU)),
        peer_rows_(std::move(peer_rows)) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    EXPECT_EQ(input.size(0), peer_rows_.size(0));
    return rank() == 0 ? torch::stack({input, peer_rows_}, /*dim=*/0)
                       : torch::stack({peer_rows_, input}, /*dim=*/0);
  }

 private:
  torch::Tensor peer_rows_;
};

torch::Tensor pad_rows_to(const torch::Tensor& rows, int64_t max_rows) {
  if (rows.size(0) == max_rows) {
    return rows;
  }
  torch::Tensor pad = torch::zeros({max_rows - rows.size(0)}, rows.options());
  return torch::cat({rows, pad}, /*dim=*/0);
}

// gather_rows pads each rank's rows up to max_rows before the allgather, then
// trims every rank back to rows_of(rank). With an asymmetric split (300 | 724)
// the short rank's padding must be dropped and both ends must reconstruct the
// same [0, num_rows) sequence -- the "both ends agree" invariant a prior
// mismatch bug violated.
TEST(MusaPcpContextTest, GatherRowsPadsAndTrimsAsymmetricSplit) {
  constexpr int32_t kFirst = 300;
  constexpr int32_t kRows = 1024;
  const int32_t max_rows = kRows - kFirst;
  const torch::Tensor local0 = torch::arange(kFirst);
  const torch::Tensor local1 = torch::arange(kFirst, kRows);
  const torch::Tensor expected = torch::arange(kRows);

  // Rank 0 owns [0, 300): shorter than max_rows, padded with junk then trimmed.
  Context ctx0;
  StubGatherGroup group0(/*rank=*/0, local1);
  ctx0.cp_group = &group0;
  ctx0.cp_rank = 0;
  ctx0.use_custom_ar = false;
  ctx0.row_offsets = {0, kFirst, kRows};
  ctx0.max_rows = max_rows;
  EXPECT_TRUE(torch::equal(gather_rows(local0, ctx0), expected));

  // Rank 1 owns [300, 1024): already max_rows long, trimmed to its own length.
  Context ctx1;
  StubGatherGroup group1(/*rank=*/1, pad_rows_to(local0, max_rows));
  ctx1.cp_group = &group1;
  ctx1.cp_rank = 1;
  ctx1.use_custom_ar = false;
  ctx1.row_offsets = {0, kFirst, kRows};
  ctx1.max_rows = max_rows;
  EXPECT_TRUE(torch::equal(gather_rows(local1, ctx1), expected));
}

// An even split fills max_rows on every rank (row_offsets.back() == cp_size *
// max_rows), so gather_rows takes the flatten-without-trim path; a half/half
// stack must round-trip unchanged.
TEST(MusaPcpContextTest, GatherRowsFlattensEvenSplitWithoutTrim) {
  constexpr int32_t kHalf = 512;
  constexpr int32_t kRows = 1024;
  const torch::Tensor local0 = torch::arange(kHalf);
  const torch::Tensor local1 = torch::arange(kHalf, kRows);

  Context ctx;
  StubGatherGroup group(/*rank=*/0, local1);
  ctx.cp_group = &group;
  ctx.cp_rank = 0;
  ctx.use_custom_ar = false;
  ctx.row_offsets = {0, kHalf, kRows};
  ctx.max_rows = kHalf;
  EXPECT_TRUE(torch::equal(gather_rows(local0, ctx), torch::arange(kRows)));
}

}  // namespace xllm::layer::musa_pcp
