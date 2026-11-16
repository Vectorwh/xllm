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

#include "framework/parallel_state/npu_cp_plan.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/parallel_state/process_group.h"
#include "runtime/forward_params.h"

namespace xllm {
namespace {

torch::Tensor int32_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values, torch::dtype(torch::kInt32));
}

torch::Tensor int64_tensor(const std::vector<int64_t>& values) {
  return torch::tensor(values, torch::dtype(torch::kInt64));
}

void expect_tensor_equal(const torch::Tensor& actual,
                         const torch::Tensor& expected) {
  ASSERT_TRUE(actual.defined());
  ASSERT_EQ(actual.scalar_type(), expected.scalar_type());
  ASSERT_EQ(actual.device(), expected.device());
  ASSERT_EQ(actual.sizes(), expected.sizes());
  EXPECT_TRUE(torch::equal(actual, expected));
}

ForwardInput make_forward_input(const CpPlanInput& input) {
  ForwardInput processed_input;
  processed_input.positions = input.position_ids;
  auto& params = processed_input.input_params;
  params.meta.num_sequences = static_cast<int32_t>(input.q_seq_lens.size());
  params.meta.batch_forward_type = BatchForwardType::PREFILL;
  params.attention.host.q_seq_lens = input.q_seq_lens;
  params.attention.host.kv_seq_lens = input.q_seq_lens;
  if (input.q_seq_lens_are_cumulative) {
    params.attention.host.q_seq_lens = {0};
    for (int32_t length : input.q_seq_lens) {
      params.attention.host.q_seq_lens.emplace_back(
          params.attention.host.q_seq_lens.back() + length);
    }
  }
  if (input.kv_seq_lens_are_cumulative) {
    params.attention.host.kv_seq_lens = {0};
    for (int32_t length : input.q_seq_lens) {
      params.attention.host.kv_seq_lens.emplace_back(
          params.attention.host.kv_seq_lens.back() + length);
    }
  }
  params.attention.host.kv_cache_tokens_nums = input.prefix_token_counts;
  params.attention.device.block_tables = input.block_tables;
  params.attention.device.new_cache_slots =
      torch::arange(input.position_ids.numel(), torch::kInt32);
  return processed_input;
}

NpuCpPlan prepare_plan(ForwardInput& processed_input,
                       const CpPlanConfig& config,
                       ProcessGroup* cp_group = nullptr,
                       bool has_prefix_slots = false) {
  CpPlanRuntimeConfig runtime_config;
  runtime_config.enabled = true;
  runtime_config.plan_config = config;
  runtime_config.cp_group = cp_group;
  runtime_config.has_prefix_slots = has_prefix_slots;
  NpuCpPlan plan;
  plan.prepare(processed_input, runtime_config);
  return plan;
}

class GatheredShardsProcessGroup final : public ProcessGroup {
 public:
  GatheredShardsProcessGroup(int32_t rank,
                             const std::vector<torch::Tensor>& shards)
      : ProcessGroup(rank,
                     static_cast<int32_t>(shards.size()),
                     torch::Device(torch::kCPU)),
        shards_(shards) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    expect_tensor_equal(input, shards_[rank()]);
    return torch::stack(shards_, /*dim=*/0);
  }

 private:
  const std::vector<torch::Tensor>& shards_;
};

CpPlanInput make_plan_input(const std::vector<int32_t>& q_seq_lens,
                            const std::vector<int32_t>& position_starts) {
  CHECK_EQ(q_seq_lens.size(), position_starts.size());
  CpPlanInput input;
  input.q_seq_lens = q_seq_lens;
  std::vector<int32_t> positions;
  positions.reserve(
      std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int64_t{0}));
  for (size_t i = 0; i < q_seq_lens.size(); ++i) {
    for (int32_t token = 0; token < q_seq_lens[i]; ++token) {
      positions.emplace_back(position_starts[i] + token);
    }
  }
  input.position_ids = int32_tensor(positions);
  input.prefix_token_counts.resize(q_seq_lens.size(), 0);
  return input;
}

CpPlanInput aligned_input() { return make_plan_input({8, 12}, {0, 0}); }

CpPlanConfig cp2_rank0_config() {
  CpPlanConfig config;
  config.cp_size = 2;
  config.cp_rank = 0;
  config.kv_split_size = 2;
  config.block_size = 128;
  config.attention_tp_size = 1;
  config.attention_tp_rank = 0;
  config.attention_cp_size = 2;
  config.attention_cp_group_size = 2;
  config.moe_ep_size = 1;
  config.expert_parallel_degree = 1;
  config.num_experts_per_token = 8;
  config.device = torch::kCPU;
  config.dtype = torch::kBFloat16;
  return config;
}

TEST(NpuCpPlanTest, BuildsAlignedGraphMetadata) {
  const NpuCpPlan plan = NpuCpPlan::build(aligned_input(), cp2_rank0_config());
  const CpInputShardMeta& shard_meta = plan.input_shard_meta();
  const CpOutputMergeMeta& merge_meta = plan.output_merge_meta();
  const CpAttentionMeta& attention = plan.attention_meta();
  const CpEpMeta& cp_ep = plan.cp_ep_meta();

  EXPECT_EQ(shard_meta.global_real_token_count, 20);
  EXPECT_EQ(merge_meta.global_padded_token_count, 20);
  EXPECT_EQ(shard_meta.local_real_token_count, 10);
  EXPECT_EQ(shard_meta.local_padded_token_count, 10);
  EXPECT_EQ(shard_meta.local_real_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(shard_meta.local_padded_seq_lens, std::vector<int32_t>({4, 6}));
  expect_tensor_equal(shard_meta.input_source_indices,
                      int64_tensor({0, 1, 6, 7, 8, 9, 10, 17, 18, 19}));
  expect_tensor_equal(shard_meta.input_destination_indices,
                      int64_tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  expect_tensor_equal(shard_meta.local_position_ids,
                      int32_tensor({0, 1, 6, 7, 0, 1, 2, 9, 10, 11}));
  expect_tensor_equal(merge_meta.output_restore_indices,
                      int64_tensor({0, 1,  10, 11, 12, 13, 2,  3, 4, 5,
                                    6, 14, 15, 16, 17, 18, 19, 7, 8, 9}));

  EXPECT_EQ(attention.host_q_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(attention.host_kv_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(attention.host_q_cu_seq_lens, std::vector<int32_t>({4, 10}));
  EXPECT_EQ(attention.q_max_seq_len, 6);
  EXPECT_EQ(attention.kv_max_seq_len, 6);
  expect_tensor_equal(attention.q_seq_lens, int32_tensor({4, 6}));
  expect_tensor_equal(attention.kv_seq_lens, int32_tensor({4, 6}));
  expect_tensor_equal(attention.q_cu_seq_lens, int32_tensor({4, 10}));
  expect_tensor_equal(attention.query_balance_indices,
                      int32_tensor({0, 1, 4, 5, 6, 2, 3, 7, 8, 9}));
  expect_tensor_equal(attention.attention_output_reorder_indices,
                      int32_tensor({0, 1, 5, 6, 2, 3, 4, 7, 8, 9}));
  expect_tensor_equal(attention.kv_reorder_indices,
                      int32_tensor({0, 1,  10, 11, 12, 13, 2,  3, 4, 5,
                                    6, 14, 15, 16, 17, 18, 19, 7, 8, 9}));
  expect_tensor_equal(attention.prev_kv_gather_indices,
                      int32_tensor({0, 1, 8, 9, 10}));
  expect_tensor_equal(attention.next_kv_gather_indices,
                      int32_tensor({0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
  expect_tensor_equal(attention.prev_query_cu_seq_lens, int32_tensor({2, 5}));
  expect_tensor_equal(attention.next_query_cu_seq_lens, int32_tensor({2, 5}));
  expect_tensor_equal(attention.prev_key_cu_seq_lens, int32_tensor({2, 5}));
  expect_tensor_equal(attention.next_key_cu_seq_lens, int32_tensor({8, 20}));

  expect_tensor_equal(cp_ep.attention_tp_padding_indices,
                      torch::arange(10, torch::kInt32));
  expect_tensor_equal(cp_ep.attention_tp_unpadding_indices,
                      torch::arange(20, torch::kInt32));
  expect_tensor_equal(cp_ep.ffn_padding_indices,
                      torch::arange(20, torch::kInt32));
  expect_tensor_equal(cp_ep.ffn_unpadding_indices,
                      torch::arange(10, torch::kInt32));
  expect_tensor_equal(cp_ep.lm_head_skip_padding_indices,
                      torch::arange(20, torch::kInt32));
  expect_tensor_equal(cp_ep.prenorm_gather_indices,
                      torch::arange(10, torch::kInt32));
  expect_tensor_equal(cp_ep.attention_padding_indices,
                      torch::arange(10, torch::kInt32));
  expect_tensor_equal(cp_ep.attention_unpadding_indices, int32_tensor({0}));
  expect_tensor_equal(cp_ep.dynamic_ep_indices, int32_tensor({0}));
  expect_tensor_equal(cp_ep.moe_indices, int32_tensor({0}));
  expect_tensor_equal(cp_ep.expert_array, int64_tensor({0}));
}

TEST(NpuCpPlanTest, ShardsModelInputAndAppliesAttentionMeta) {
  ForwardInput processed_input = make_forward_input(aligned_input());
  const NpuCpPlan plan = prepare_plan(processed_input, cp2_rank0_config());
  torch::Tensor hidden = torch::arange(80, torch::kFloat).view({20, 4});
  const torch::Tensor global_hidden = hidden.clone();
  torch::Tensor positions = aligned_input().position_ids;
  plan.shard_model_input(hidden, positions);

  expect_tensor_equal(
      hidden,
      global_hidden.index_select(
          /*dim=*/0, int64_tensor({0, 1, 6, 7, 8, 9, 10, 17, 18, 19})));
  expect_tensor_equal(positions,
                      int32_tensor({0, 1, 6, 7, 0, 1, 2, 9, 10, 11}));

  const ModelInputParams& params = processed_input.input_params;
  EXPECT_EQ(params.attention.host.q_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(params.attention.host.kv_seq_lens, std::vector<int32_t>({4, 6}));
  EXPECT_EQ(params.attention.host.q_cu_seq_lens, std::vector<int32_t>({4, 10}));
  expect_tensor_equal(params.attention.device.q_seq_lens, int32_tensor({4, 6}));
  expect_tensor_equal(params.attention.device.kv_seq_lens,
                      int32_tensor({4, 6}));
  expect_tensor_equal(params.attention.device.q_cu_seq_lens,
                      int32_tensor({4, 10}));
  EXPECT_EQ(params.meta.q_max_seq_len, 6);
  EXPECT_EQ(params.meta.kv_max_seq_len, 6);
}

TEST(NpuCpPlanTest, NonAlignedInputUsesVirtualPadding) {
  CpPlanInput input;
  input.q_seq_lens = {5, 7};
  input.position_ids = int32_tensor({0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6});
  input.prefix_token_counts = {0, 0};
  ForwardInput processed_input = make_forward_input(input);
  processed_input.kv_slot_layout = KvSlotLayout::LOGICAL_REAL;
  processed_input.input_params.attention.device.new_cache_slots =
      torch::cat({torch::arange(768, 773, torch::kInt32),
                  torch::arange(1024, 1031, torch::kInt32)});
  const NpuCpPlan plan = prepare_plan(processed_input, cp2_rank0_config());
  EXPECT_EQ(processed_input.kv_slot_layout,
            KvSlotLayout::NPU_CP_RECOVERED_PHYSICAL);
  expect_tensor_equal(
      processed_input.input_params.attention.device.new_cache_slots,
      int32_tensor({384,
                    385,
                    386,
                    387,
                    388,
                    -1,
                    -1,
                    -1,
                    512,
                    513,
                    514,
                    515,
                    516,
                    517,
                    518,
                    -1}));

  EXPECT_EQ(plan.input_shard_meta().global_real_token_count, 12);
  EXPECT_EQ(plan.output_merge_meta().global_padded_token_count, 16);
  EXPECT_EQ(plan.input_shard_meta().local_real_token_count, 5);
  EXPECT_EQ(plan.input_shard_meta().local_padded_token_count, 8);
  EXPECT_EQ(plan.input_shard_meta().local_real_seq_lens,
            std::vector<int32_t>({2, 3}));
  EXPECT_EQ(plan.input_shard_meta().local_padded_seq_lens,
            std::vector<int32_t>({4, 4}));

  torch::Tensor hidden = torch::arange(12, torch::kFloat).view({12, 1});
  torch::Tensor positions = input.position_ids;
  plan.shard_model_input(hidden, positions);
  expect_tensor_equal(
      hidden.flatten(),
      torch::tensor({0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 6.0f, 11.0f, 0.0f}));

  const CpAttentionMeta& attention = plan.attention_meta();
  EXPECT_EQ(attention.host_q_seq_lens, std::vector<int32_t>({4, 4}));
  EXPECT_EQ(attention.host_kv_seq_lens, std::vector<int32_t>({4, 4}));
  EXPECT_EQ(attention.host_q_cu_seq_lens, std::vector<int32_t>({4, 8}));
  EXPECT_EQ(attention.q_max_seq_len, 4);
  EXPECT_EQ(attention.kv_max_seq_len, 4);
  expect_tensor_equal(attention.q_seq_lens, int32_tensor({4, 4}));
  expect_tensor_equal(attention.kv_seq_lens, int32_tensor({4, 4}));
  expect_tensor_equal(attention.q_cu_seq_lens, int32_tensor({4, 8}));
  expect_tensor_equal(attention.query_balance_indices,
                      int32_tensor({0, 1, 4, 5, 2, 3, 6, 7}));
  expect_tensor_equal(attention.attention_output_reorder_indices,
                      int32_tensor({0, 1, 4, 5, 2, 3, 6, 7}));
  expect_tensor_equal(
      attention.kv_reorder_indices,
      int32_tensor({0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 12, 13, 14, 15, 6, 7}));
  expect_tensor_equal(attention.prev_kv_gather_indices,
                      int32_tensor({0, 1, 8, 9}));
  expect_tensor_equal(
      attention.next_kv_gather_indices,
      int32_tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
  expect_tensor_equal(attention.prev_query_cu_seq_lens, int32_tensor({2, 4}));
  expect_tensor_equal(attention.next_query_cu_seq_lens, int32_tensor({2, 4}));
  expect_tensor_equal(attention.prev_key_cu_seq_lens, int32_tensor({2, 4}));
  expect_tensor_equal(attention.next_key_cu_seq_lens, int32_tensor({8, 16}));
}

TEST(NpuCpPlanTest, InputShardAndOutputMergeRoundTripAcrossRanks) {
  struct TestCase {
    std::vector<int32_t> q_seq_lens;
    int32_t cp_size;
  };
  const std::vector<TestCase> cases = {
      {{8, 12}, 2}, {{5, 7, 1, 3}, 2}, {{16, 8, 24}, 4}, {{1}, 4}};

  for (const TestCase& test_case : cases) {
    const std::vector<int32_t> position_starts(test_case.q_seq_lens.size(), 0);
    const CpPlanInput input =
        make_plan_input(test_case.q_seq_lens, position_starts);
    torch::Tensor global_hidden =
        torch::arange(input.position_ids.numel(), torch::kInt32).view({-1, 1});
    std::vector<torch::Tensor> rank_shards;
    rank_shards.reserve(test_case.cp_size);
    for (int32_t cp_rank = 0; cp_rank < test_case.cp_size; ++cp_rank) {
      CpPlanConfig config = cp2_rank0_config();
      config.cp_size = test_case.cp_size;
      config.cp_rank = cp_rank;
      config.kv_split_size = test_case.cp_size;
      config.attention_cp_size = test_case.cp_size;
      config.attention_cp_group_size = test_case.cp_size;
      NpuCpPlan plan = NpuCpPlan::build(input, config);
      torch::Tensor local_hidden = global_hidden.clone();
      torch::Tensor local_positions = input.position_ids;
      plan.shard_model_input(local_hidden, local_positions);
      rank_shards.emplace_back(std::move(local_hidden));
    }

    for (int32_t cp_rank = 0; cp_rank < test_case.cp_size; ++cp_rank) {
      SCOPED_TRACE(cp_rank);
      CpPlanConfig config = cp2_rank0_config();
      config.cp_size = test_case.cp_size;
      config.cp_rank = cp_rank;
      config.kv_split_size = test_case.cp_size;
      config.attention_cp_size = test_case.cp_size;
      config.attention_cp_group_size = test_case.cp_size;
      GatheredShardsProcessGroup group(cp_rank, rank_shards);
      ForwardInput processed_input = make_forward_input(input);
      const NpuCpPlan plan = prepare_plan(processed_input, config, &group);
      expect_tensor_equal(plan.merge_model_output(rank_shards[cp_rank]),
                          global_hidden);
    }
  }
}

TEST(NpuCpPlanTest, OutputMergeRejectsInvalidProcessGroup) {
#if GTEST_HAS_DEATH_TEST
  NpuCpPlan plan = NpuCpPlan::build(aligned_input(), cp2_rank0_config());
  const torch::Tensor local_hidden = torch::zeros({10, 1}, torch::kFloat);
  // No process group bound -> merge must reject.
  EXPECT_DEATH(plan.merge_model_output(local_hidden), "process_group");

  ProcessGroup wrong_size_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  ForwardInput wrong_size_input = make_forward_input(aligned_input());
  plan = prepare_plan(wrong_size_input, cp2_rank0_config(), &wrong_size_group);
  EXPECT_DEATH(plan.merge_model_output(local_hidden), "size mismatch");

  ProcessGroup wrong_rank_group(
      /*rank=*/1, /*world_size=*/2, torch::Device(torch::kCPU));
  ForwardInput wrong_rank_input = make_forward_input(aligned_input());
  plan = prepare_plan(wrong_rank_input, cp2_rank0_config(), &wrong_rank_group);
  EXPECT_DEATH(plan.merge_model_output(local_hidden), "rank mismatch");
#endif
}

TEST(NpuCpPlanTest, PreparesPrefixAttentionMetadata) {
  CpPlanInput input;
  input.q_seq_lens = {8};
  input.position_ids = int32_tensor({256, 257, 258, 259, 260, 261, 262, 263});
  input.prefix_token_counts = {256};
  input.block_tables = int32_tensor({5, 6}).view({1, 2});
  input.has_prefix_slots = true;
  const NpuCpPlan plan = NpuCpPlan::build(input, cp2_rank0_config());
  const CpAttentionMeta& attention = plan.attention_meta();

  expect_tensor_equal(attention.prev_kv_gather_indices,
                      torch::arange(258, torch::kInt32));
  expect_tensor_equal(attention.next_kv_gather_indices,
                      torch::arange(264, torch::kInt32));
  expect_tensor_equal(attention.prev_key_cu_seq_lens, int32_tensor({258}));
  expect_tensor_equal(attention.next_key_cu_seq_lens, int32_tensor({264}));
  expect_tensor_equal(attention.prefix_cache_slots,
                      torch::arange(640, 768, torch::kInt32));

  ForwardInput processed_input = make_forward_input(input);
  prepare_plan(processed_input,
               cp2_rank0_config(),
               /*cp_group=*/nullptr,
               /*has_prefix_slots=*/true);
  expect_tensor_equal(
      processed_input.input_params.attention.device.in_prefix_slots,
      torch::arange(640, 768, torch::kInt32));
}

TEST(NpuCpPlanTest, MixedPrefixAttentionMetadataSkipsPaddingSlots) {
  CpPlanInput input;
  input.q_seq_lens = {8, 4};
  input.position_ids =
      int32_tensor({256, 257, 258, 259, 260, 261, 262, 263, 0, 1, 2, 3});
  input.prefix_token_counts = {256, 0};
  input.block_tables = int32_tensor({5, 6, 9, 10}).view({2, 2});
  input.has_prefix_slots = true;
  const NpuCpPlan plan = NpuCpPlan::build(input, cp2_rank0_config());
  const CpAttentionMeta& attention = plan.attention_meta();

  const torch::Tensor prefix_indices =
      torch::cat({torch::arange(0, 128, torch::kInt32),
                  torch::arange(129, 257, torch::kInt32)});
  expect_tensor_equal(attention.prev_kv_gather_indices,
                      torch::cat({prefix_indices,
                                  torch::arange(258, 260, torch::kInt32),
                                  int32_tensor({266})}));
  expect_tensor_equal(
      attention.next_kv_gather_indices,
      torch::cat({prefix_indices, torch::arange(258, 270, torch::kInt32)}));
  expect_tensor_equal(attention.prev_key_cu_seq_lens, int32_tensor({258, 259}));
  expect_tensor_equal(attention.next_key_cu_seq_lens, int32_tensor({264, 268}));
  expect_tensor_equal(
      attention.prefix_cache_slots,
      torch::cat({torch::arange(640, 768, torch::kInt32), int32_tensor({0})}));
}

TEST(NpuCpPlanTest, HandlesUnsplitAndEmptyPrefixCacheSlots) {
  CpPlanInput input;
  input.q_seq_lens = {8};
  input.position_ids = int32_tensor({256, 257, 258, 259, 260, 261, 262, 263});
  input.prefix_token_counts = {256};
  input.block_tables = int32_tensor({5, 6}).view({1, 2});
  input.has_prefix_slots = true;

  CpPlanConfig config = cp2_rank0_config();
  config.kv_split_size = 1;
  const NpuCpPlan plan = NpuCpPlan::build(input, config);
  expect_tensor_equal(plan.attention_meta().prefix_cache_slots,
                      torch::arange(640, 896, torch::kInt32));

  CpPlanInput empty_input = make_plan_input({}, {});
  empty_input.has_prefix_slots = true;
  empty_input.block_tables = torch::empty({0, 2}, torch::kInt32);
  const NpuCpPlan empty_plan =
      NpuCpPlan::build(empty_input, cp2_rank0_config());
  expect_tensor_equal(empty_plan.attention_meta().prefix_cache_slots,
                      int32_tensor({0}));
}

TEST(NpuCpPlanTest, BuildsDynamicEpMetadata) {
  CpPlanConfig config = cp2_rank0_config();
  config.attention_tp_size = 2;
  config.moe_ep_size = 4;
  config.expert_parallel_degree = 2;
  const NpuCpPlan plan = NpuCpPlan::build(aligned_input(), config);
  const CpEpMeta& cp_ep = plan.cp_ep_meta();

  expect_tensor_equal(cp_ep.attention_tp_unpadding_indices,
                      torch::arange(5, torch::kInt32));
  expect_tensor_equal(cp_ep.ffn_padding_indices,
                      torch::arange(5, torch::kInt32));
  expect_tensor_equal(cp_ep.dynamic_ep_indices,
                      torch::arange(40, torch::kInt32));
  expect_tensor_equal(cp_ep.moe_indices, torch::arange(1, 321, torch::kInt32));
  expect_tensor_equal(cp_ep.expert_array,
                      torch::ones({320, 1}, torch::dtype(torch::kBFloat16)));
}

TEST(NpuCpPlanTest, BuildsDynamicEpMetadataForAttentionTpSizeOne) {
  CpPlanConfig config = cp2_rank0_config();
  config.moe_ep_size = 4;
  config.expert_parallel_degree = 2;
  const NpuCpPlan plan = NpuCpPlan::build(aligned_input(), config);
  const int64_t local_tokens = plan.input_shard_meta().local_padded_token_count;
  const CpEpMeta& cp_ep = plan.cp_ep_meta();
  expect_tensor_equal(cp_ep.attention_tp_padding_indices,
                      torch::arange(local_tokens, torch::kInt32));
  expect_tensor_equal(cp_ep.attention_tp_unpadding_indices,
                      torch::arange(local_tokens, torch::kInt32));
  expect_tensor_equal(cp_ep.ffn_padding_indices,
                      torch::arange(local_tokens, torch::kInt32));
  expect_tensor_equal(cp_ep.ffn_unpadding_indices,
                      torch::arange(local_tokens, torch::kInt32));
  expect_tensor_equal(
      cp_ep.lm_head_skip_padding_indices,
      torch::arange(local_tokens * config.cp_size, torch::kInt32));
  expect_tensor_equal(cp_ep.prenorm_gather_indices,
                      torch::arange(local_tokens, torch::kInt32));
  expect_tensor_equal(cp_ep.attention_padding_indices,
                      cp_ep.attention_tp_padding_indices);
  expect_tensor_equal(cp_ep.attention_unpadding_indices, int32_tensor({0}));
  expect_tensor_equal(cp_ep.dynamic_ep_indices,
                      torch::arange(local_tokens * config.num_experts_per_token,
                                    torch::kInt32));
  expect_tensor_equal(cp_ep.moe_indices, torch::arange(1, 641, torch::kInt32));
  expect_tensor_equal(cp_ep.expert_array,
                      torch::ones({640, 1}, torch::dtype(torch::kBFloat16)));
}

TEST(NpuCpPlanTest, DynamicEpDegreeThreeRequiresPrefill) {
  CpPlanConfig config = cp2_rank0_config();
  config.attention_tp_size = 2;
  config.moe_ep_size = 4;
  config.expert_parallel_degree = 3;
  const CpPlanInput input = aligned_input();

  config.is_prefill = true;
  const NpuCpPlan prefill_plan = NpuCpPlan::build(input, config);
  expect_tensor_equal(prefill_plan.cp_ep_meta().dynamic_ep_indices,
                      torch::arange(40, torch::kInt32));
  config.attention_tp_rank = 1;
  const NpuCpPlan tp_rank1_prefill = NpuCpPlan::build(input, config);
  const CpEpMeta& rank1_meta = tp_rank1_prefill.cp_ep_meta();
  expect_tensor_equal(rank1_meta.attention_tp_unpadding_indices,
                      torch::arange(5, torch::kInt32));
  expect_tensor_equal(rank1_meta.ffn_padding_indices,
                      torch::arange(5, torch::kInt32));
  expect_tensor_equal(rank1_meta.prenorm_gather_indices,
                      torch::arange(5, 10, torch::kInt32));
  expect_tensor_equal(rank1_meta.dynamic_ep_indices,
                      torch::arange(40, torch::kInt32));
  expect_tensor_equal(rank1_meta.moe_indices,
                      torch::arange(1, 321, torch::kInt32));
  expect_tensor_equal(rank1_meta.expert_array,
                      torch::ones({320, 1}, torch::dtype(torch::kBFloat16)));

  config.attention_tp_rank = 0;
  config.is_prefill = false;
  const NpuCpPlan decode_plan = NpuCpPlan::build(input, config);
  expect_tensor_equal(decode_plan.cp_ep_meta().dynamic_ep_indices,
                      torch::zeros({1}, torch::kInt32));
  expect_tensor_equal(decode_plan.cp_ep_meta().moe_indices,
                      torch::zeros({1}, torch::kInt32));
}

TEST(NpuCpPlanTest, CacheSlotsPreserveUnsplitLogicalOffsets) {
  const CpPlanInput input = make_plan_input({8}, {128});
  const torch::Tensor global_slots =
      torch::arange(30 * 256 + 128, 30 * 256 + 136, torch::kInt32);
  CpPlanConfig config = cp2_rank0_config();
  config.kv_split_size = 1;
  const torch::Tensor expected = torch::arange(
      61 * config.block_size, 61 * config.block_size + 8, torch::kInt32);

  for (int32_t cp_rank = 0; cp_rank < config.cp_size; ++cp_rank) {
    config.cp_rank = cp_rank;
    const NpuCpPlan plan = NpuCpPlan::build(input, config);
    const torch::Tensor mapped = plan.prepare_cache_slots(global_slots);
    ASSERT_EQ(mapped.numel(), expected.numel());
    expect_tensor_equal(mapped, expected);
  }
}

TEST(NpuCpPlanTest, CacheSlotsPreserveNonAlignedCpFourSplitOrder) {
  const CpPlanInput input = make_plan_input({5, 7, 1, 3}, {0, 0, 0, 0});
  const torch::Tensor global_slots = int32_tensor({2686,
                                                   2687,
                                                   2688,
                                                   2689,
                                                   2690,
                                                   2940,
                                                   2941,
                                                   2942,
                                                   2943,
                                                   2944,
                                                   2945,
                                                   2946,
                                                   3327,
                                                   3455,
                                                   3456,
                                                   3457});
  CpPlanConfig config = cp2_rank0_config();
  config.cp_size = 4;
  config.attention_cp_size = 4;
  config.attention_cp_group_size = 4;
  config.kv_split_size = 2;

  const std::vector<std::vector<int32_t>> expected = {
      {1406, 1407, -1,   -1, -1, -1, -1, -1, 1532, 1533, 1534,
       1535, -1,   -1,   -1, -1, -1, -1, -1, -1,   -1,   -1,
       -1,   -1,   1791, -1, -1, -1, -1, -1, -1,   -1},
      {-1, -1,   1280, 1281, 1282, -1,   -1, -1, -1, -1, -1,
       -1, 1408, 1409, 1410, -1,   1663, -1, -1, -1, -1, -1,
       -1, -1,   -1,   1664, 1665, -1,   -1, -1, -1, -1}};
  for (int32_t cp_rank = 0; cp_rank < config.cp_size; ++cp_rank) {
    SCOPED_TRACE(cp_rank);
    config.cp_rank = cp_rank;
    config.kv_split_rank = cp_rank % config.kv_split_size;
    const NpuCpPlan plan = NpuCpPlan::build(input, config);
    expect_tensor_equal(plan.prepare_cache_slots(global_slots),
                        int32_tensor(expected[config.kv_split_rank]));
  }
}

TEST(NpuCpPlanTest, CacheSlotsPreserveShortCpFourUnsplitOrder) {
  const CpPlanInput input = make_plan_input({1}, {128});
  const torch::Tensor global_slots = int32_tensor({1297});
  CpPlanConfig config = cp2_rank0_config();
  config.cp_size = 4;
  config.attention_cp_size = 4;
  config.attention_cp_group_size = 4;
  config.kv_split_size = 1;
  const NpuCpPlan plan = NpuCpPlan::build(input, config);
  expect_tensor_equal(plan.prepare_cache_slots(global_slots),
                      int32_tensor({1297, -1, -1, -1, -1, -1, -1, -1}));
}

TEST(NpuCpPlanTest, CacheSlotsUseLogicalOffsetsAcrossSequences) {
  const CpPlanInput input = make_plan_input({200, 8}, {0, 0});
  const int32_t logical_block_size = 256;
  torch::Tensor global_slots =
      torch::cat({torch::arange(10 * logical_block_size,
                                10 * logical_block_size + 200,
                                torch::kInt32),
                  torch::arange(20 * logical_block_size,
                                20 * logical_block_size + 8,
                                torch::kInt32)});

  CpPlanConfig rank0_config = cp2_rank0_config();
  const NpuCpPlan rank0_plan = NpuCpPlan::build(input, rank0_config);
  expect_tensor_equal(
      rank0_plan.prepare_cache_slots(global_slots),
      torch::cat({torch::arange(10 * rank0_config.block_size,
                                11 * rank0_config.block_size,
                                torch::kInt32),
                  torch::full({72}, -1, torch::kInt32),
                  torch::arange(20 * rank0_config.block_size,
                                20 * rank0_config.block_size + 8,
                                torch::kInt32)}));

  CpPlanConfig rank1_config = rank0_config;
  rank1_config.cp_rank = 1;
  rank1_config.kv_split_rank = 1;
  const NpuCpPlan rank1_plan = NpuCpPlan::build(input, rank1_config);
  expect_tensor_equal(
      rank1_plan.prepare_cache_slots(global_slots),
      torch::cat({torch::full({128}, -1, torch::kInt32),
                  torch::arange(10 * rank1_config.block_size,
                                10 * rank1_config.block_size + 72,
                                torch::kInt32),
                  torch::full({8}, -1, torch::kInt32)}));
}

TEST(NpuCpPlanTest, CacheSlotsUseLogicalOffsetsForSingleSequenceChunk) {
  const CpPlanInput input = make_plan_input({8}, {128});
  const int32_t logical_block_size = 256;
  torch::Tensor global_slots = torch::arange(30 * logical_block_size + 128,
                                             30 * logical_block_size + 136,
                                             torch::kInt32);

  CpPlanConfig rank0_config = cp2_rank0_config();
  const NpuCpPlan rank0_plan = NpuCpPlan::build(input, rank0_config);
  expect_tensor_equal(rank0_plan.prepare_cache_slots(global_slots),
                      torch::full({8}, -1, torch::kInt32));

  CpPlanConfig rank1_config = rank0_config;
  rank1_config.cp_rank = 1;
  rank1_config.kv_split_rank = 1;
  const NpuCpPlan rank1_plan = NpuCpPlan::build(input, rank1_config);
  expect_tensor_equal(rank1_plan.prepare_cache_slots(global_slots),
                      torch::arange(30 * rank1_config.block_size,
                                    30 * rank1_config.block_size + 8,
                                    torch::kInt32));
}

TEST(NpuCpPlanTest, MtpTargetAndDraftShardGlobalInputsExactlyOnce) {
  const CpPlanInput mtp_input = make_plan_input({5, 7}, {0, 0});
  const NpuCpPlan target_plan = NpuCpPlan::build(mtp_input, cp2_rank0_config());
  const NpuCpPlan draft_plan = NpuCpPlan::build(mtp_input, cp2_rank0_config());
  const torch::Tensor global_slots =
      torch::cat({torch::arange(3 * 256, 3 * 256 + 5, torch::kInt32),
                  torch::arange(4 * 256, 4 * 256 + 7, torch::kInt32)});
  const torch::Tensor target_slots =
      target_plan.prepare_cache_slots(global_slots);
  const torch::Tensor draft_slots =
      draft_plan.prepare_cache_slots(global_slots);
  EXPECT_EQ(target_slots.numel(), target_plan.recovered_token_count());
  expect_tensor_equal(target_slots,
                      int32_tensor({384,
                                    385,
                                    386,
                                    387,
                                    388,
                                    -1,
                                    -1,
                                    -1,
                                    512,
                                    513,
                                    514,
                                    515,
                                    516,
                                    517,
                                    518,
                                    -1}));
  expect_tensor_equal(draft_slots, target_slots);

  torch::Tensor target_hidden = torch::arange(96, torch::kFloat).view({12, 8});
  torch::Tensor draft_hidden = target_hidden + 1000;
  torch::Tensor target_positions = mtp_input.position_ids;
  torch::Tensor draft_positions = mtp_input.position_ids;
  target_plan.shard_model_input(target_hidden, target_positions);
  draft_plan.shard_model_input(draft_hidden, draft_positions);
  EXPECT_EQ(target_hidden.size(0),
            target_plan.input_shard_meta().local_padded_token_count);
  EXPECT_EQ(draft_hidden.size(0),
            draft_plan.input_shard_meta().local_padded_token_count);
  const torch::Tensor& destination_indices =
      target_plan.input_shard_meta().input_destination_indices;
  expect_tensor_equal(
      draft_hidden.index_select(/*dim=*/0, destination_indices),
      target_hidden.index_select(/*dim=*/0, destination_indices) + 1000);
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(draft_plan.prepare_cache_slots(draft_slots),
               "global-real logical layout");
  EXPECT_DEATH(draft_plan.shard_model_input(draft_hidden, draft_positions),
               "exactly once");
#endif
}

TEST(NpuCpPlanTest, CumulativeHostLayoutIsPreserved) {
  CpPlanInput input = aligned_input();
  input.q_seq_lens_are_cumulative = true;
  input.kv_seq_lens_are_cumulative = true;
  ForwardInput processed_input = make_forward_input(input);
  const NpuCpPlan plan = prepare_plan(processed_input, cp2_rank0_config());
  EXPECT_EQ(plan.attention_meta().host_q_seq_lens,
            std::vector<int32_t>({0, 4, 10}));
  EXPECT_EQ(plan.attention_meta().host_kv_seq_lens,
            std::vector<int32_t>({0, 4, 10}));
  expect_tensor_equal(plan.attention_meta().q_seq_lens,
                      int32_tensor({0, 4, 10}));
  expect_tensor_equal(plan.attention_meta().kv_seq_lens,
                      int32_tensor({0, 4, 10}));
  expect_tensor_equal(plan.attention_meta().q_cu_seq_lens,
                      int32_tensor({4, 10}));
}

TEST(NpuCpPlanTest, EmptyPlanDropsWorkerFakeModelRow) {
  const NpuCpPlan plan =
      NpuCpPlan::build(make_plan_input({}, {}), cp2_rank0_config());
  torch::Tensor fake_hidden = torch::ones({1, 8}, torch::kFloat);
  torch::Tensor fake_position = int32_tensor({0});
  plan.shard_model_input(fake_hidden, fake_position);

  EXPECT_EQ(fake_hidden.sizes(), torch::IntArrayRef({0, 8}));
  EXPECT_EQ(fake_position.numel(), 0);
}

TEST(NpuCpPlanTest, DisabledPlanIsNoOp) {
  const NpuCpPlan plan;
  torch::Tensor hidden = torch::randn({4, 8});
  torch::Tensor positions = torch::arange(4, torch::kInt32);
  torch::Tensor slots = torch::arange(4, torch::kInt32);
  const torch::Tensor hidden_before = hidden.clone();
  const torch::Tensor positions_before = positions.clone();
  // shard_model_input is a no-op (leaves both tensors unchanged) when the
  // plan is disabled; merge_model_output / prepare_cache_slots return their
  // input tensor unchanged.
  plan.shard_model_input(hidden, positions);
  expect_tensor_equal(hidden, hidden_before);
  expect_tensor_equal(positions, positions_before);
  EXPECT_TRUE(plan.prepare_cache_slots(slots).is_same(slots));
  EXPECT_TRUE(plan.merge_model_output(hidden).is_same(hidden));
}

}  // namespace
}  // namespace xllm
