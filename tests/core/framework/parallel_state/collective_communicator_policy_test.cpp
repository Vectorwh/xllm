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

#include <gtest/gtest.h>

#include "core/framework/parallel_state/collective_communicator.h"
#include "core/framework/parallel_state/parallel_state.h"

namespace xllm {
namespace {

TEST(CollectiveCommunicatorPolicyTest,
     CpIndependentTpPreservesDpAndClearsContextSharding) {
  const torch::Device device(torch::kCPU);
  // Two independent DP batches, each with four CP ranks and two TP ranks.
  for (int32_t rank = 0; rank < 16; ++rank) {
    ProcessGroup target_tp(rank % 2, /*world_size=*/2, device);
    ProcessGroup tp_cp(rank % 8, /*world_size=*/8, device);
    ProcessGroup cp((rank % 8) / 2, /*world_size=*/4, device);
    ParallelArgs target(rank, /*world_size=*/16, /*process_group=*/nullptr);
    target.dp_size(2).cp_size(4).tp_size(2).kv_split_size(4);
    target.tp_group_ = &target_tp;
    target.tp_cp_group_ = &tp_cp;
    target.cp_group_ = &cp;
    target.dcp_group_ = &cp;
    target.dcp_topology_ = parallel_state::build_contiguous_dcp_topology(
        rank, /*world_size=*/16, /*dp_size=*/2, /*dcp_size=*/4);

    const ParallelArgs draft =
        parallel_state::get_cp_independent_tp_parallel_args(target);
    EXPECT_EQ(draft.rank(), rank);
    EXPECT_EQ(draft.world_size(), 16);
    EXPECT_EQ(draft.dp_size(), 2);
    EXPECT_EQ(draft.tp_size(), 8);
    EXPECT_EQ(draft.tp_group_, &tp_cp);
    EXPECT_EQ(draft.lm_head_group_, nullptr);
    EXPECT_EQ(draft.cp_size(), 1);
    EXPECT_EQ(draft.kv_split_size_effective(), 1);
    EXPECT_EQ(draft.cp_group_, nullptr);
    EXPECT_EQ(draft.dcp_group_, nullptr);
    EXPECT_FALSE(draft.dcp_topology_.has_value());
    EXPECT_EQ(target.cp_size(), 4);
    EXPECT_EQ(target.tp_group_, &target_tp);
    EXPECT_TRUE(target.dcp_topology_.has_value());
  }
}

TEST(CollectiveCommunicatorPolicyTest,
     PreservesLegacyKvSplitRankWithoutTopology) {
  for (int32_t global_rank = 0; global_rank < 8; ++global_rank) {
    ParallelArgs args(global_rank, /*world_size=*/8, /*process_group=*/nullptr);
    args.kv_split_size(2);
    EXPECT_FALSE(args.dcp_topology_.has_value());
    EXPECT_EQ(args.kv_split_rank(), global_rank / 4);
  }
}

TEST(CollectiveCommunicatorPolicyTest,
     PreservesNativeDcpGroupRankWithoutTopology) {
  ProcessGroup dcp_group(
      /*rank=*/1, /*world_size=*/2, torch::Device(torch::kCPU));
  ParallelArgs args(/*rank=*/1, /*world_size=*/8, /*process_group=*/nullptr);
  args.kv_split_size(2);
  args.dcp_group_ = &dcp_group;
  EXPECT_FALSE(args.dcp_topology_.has_value());
  EXPECT_EQ(args.kv_split_rank(), 1);
}

TEST(CollectiveCommunicatorPolicyTest, MoETpGroupIdentityFollowsReusePolicy) {
  // Bare ProcessGroup instances carry no backend communicator: construction
  // and destruction are safe while the pg_ backend handle stays null.
  ProcessGroup tp_group(
      /*rank=*/0, /*world_size=*/4, torch::Device(torch::kCPU));
  ProcessGroup world_group(
      /*rank=*/0, /*world_size=*/4, torch::Device(torch::kCPU));
  ASSERT_NE(&tp_group, &world_group);

  // DP1/EP1 with an equivalent TP rank set: dense and MoE collectives share
  // the very same TP communicator object.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/1,
                                /*tp_size=*/4,
                                /*moe_tp_size=*/4),
            &tp_group);

  // DP>1 or a non-equivalent MoE TP size keeps the world process group.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/2,
                                /*tp_size=*/2,
                                /*moe_tp_size=*/4),
            &world_group);
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/1,
                                /*tp_size=*/2,
                                /*moe_tp_size=*/4),
            &world_group);
  // dp_size > 1 with moe_tp_size == tp_size is the width
  // resolve_moe_tp_size() returns for an opted-in caller whatever dp_size says,
  // so it is what a wrongly-set `moe_tp_scoped_to_cp_cohort` flag would produce
  // here. The selection must still refuse the cohort's TP communicator: the
  // caller-side dp_size == 1 gate (worker_server.cpp) keeps the published flag
  // and the chosen group consistent, and this pins the half that makes a wrong
  // flag harmless instead of world-mixing.
  EXPECT_EQ(select_moe_tp_group(&tp_group,
                                &world_group,
                                /*dp_size=*/2,
                                /*tp_size=*/4,
                                /*moe_tp_size=*/4),
            &world_group);
}

}  // namespace
}  // namespace xllm
