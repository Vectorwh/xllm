/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/parallel_state/mapping_npu.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/framework/parallel_state/context_parallel_topology.h"
#include "xllm_atb_layers/core/include/atb_speed/base/external_comm_manager.h"

namespace xllm {
namespace {

class ScopedEnvVar final {
 public:
  explicit ScopedEnvVar(const std::string& name) : name_(name) {
    const char* old_value = std::getenv(name_.c_str());
    if (old_value != nullptr) {
      old_value_ = old_value;
    }
  }

  ~ScopedEnvVar() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_.value().c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  void unset() { unsetenv(name_.c_str()); }

  void set(const std::string& value) {
    setenv(name_.c_str(), value.c_str(), 1);
  }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

MappingNPU::Options get_mapping_options() {
  MappingNPU::Options options;
  options.dp_size(2)
      .tp_size(8)
      .moe_tp_size(2)
      .moe_ep_size(8)
      .pp_size(1)
      .sp_size(1);
  return options;
}

}  // namespace

TEST(TestMappingNPU, ToJson) {
  ScopedEnvVar hccl_buffsize("HCCL_BUFFSIZE");
  hccl_buffsize.unset();

  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 6, options);
  nlohmann::json data = mapping.to_json();
  nlohmann::json attn_dp = data["attnDp"];
  int32_t attn_dp_group_id = attn_dp["groupId"];
  EXPECT_EQ(attn_dp_group_id, 6);
  nlohmann::json attn_tp = data["attnTp"];
  int32_t attn_tp_group_id = attn_tp["groupId"];
  EXPECT_EQ(attn_tp_group_id, 0);
  nlohmann::json mlp_tp = data["mlpTp"];
  int32_t mlp_tp_buffer_size = mlp_tp["bufferSize"];
  EXPECT_EQ(mlp_tp_buffer_size, 512);
}

TEST(TestMappingNPU, ToJsonUsesHcclBuffsizeEnv) {
  ScopedEnvVar hccl_buffsize("HCCL_BUFFSIZE");
  hccl_buffsize.set("128");

  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 6, options);
  nlohmann::json data = mapping.to_json();
  nlohmann::json mlp_tp = data["mlpTp"];
  int32_t mlp_tp_buffer_size = mlp_tp["bufferSize"];
  EXPECT_EQ(mlp_tp_buffer_size, 128);
}

TEST(TestMappingNPU, ContextParallelGroupsStayWithinDataParallelReplicas) {
  constexpr int32_t kWorldSize = 16;
  const std::vector<std::tuple<int32_t, int32_t, int32_t>> layouts = {
      {2, 8, 2}, {2, 4, 4}, {2, 4, 2}, {1, 8, 2}, {8, 1, 1}};
  for (const auto& [dp_size, cp_size, kv_split_size] : layouts) {
    MappingNPU::Options options;
    options.dp_size(dp_size)
        .cp_size(cp_size)
        .tp_size(kWorldSize / dp_size / cp_size)
        .moe_tp_size(1)
        .moe_ep_size(kWorldSize)
        .sp_size(1)
        .kv_split_size(kv_split_size);
    for (int32_t rank = 0; rank < kWorldSize; ++rank) {
      SCOPED_TRACE(::testing::Message() << "dp=" << dp_size << " cp=" << cp_size
                                        << " rank=" << rank);
      const parallel_state::ContextParallelTopology topology(
          rank, kWorldSize, dp_size, cp_size, kv_split_size);
      MappingNPU mapping("", kWorldSize, rank, options);
      const auto data = mapping.to_json();
      EXPECT_EQ(data["attnCp"]["rankIds"], topology.pcp_group_ranks());
      EXPECT_EQ(data["attnCp"]["rank"], topology.pcp_rank());
      EXPECT_EQ(data["kvSplit"]["rankIds"], topology.dcp_group_ranks());
      EXPECT_EQ(data["kvSplit"]["rank"], topology.dcp_rank());
    }
  }
}

TEST(TestMappingNPU, BufferOffsetIsExplicitAndRankIndependent) {
  ScopedEnvVar hccl_buffsize("HCCL_BUFFSIZE");
  hccl_buffsize.set("128");
  MappingNPU::Options options = get_mapping_options();
  MappingNPU first("", /*world_size=*/16, /*rank=*/0, options);
  MappingNPU second("", /*world_size=*/16, /*rank=*/1, options);
  EXPECT_EQ(first.to_json()["attnTp"]["bufferSize"], 128);
  EXPECT_EQ(second.to_json()["attnTp"]["bufferSize"], 128);
  options.buffer_offset(1);
  MappingNPU draft("", /*world_size=*/16, /*rank=*/1, options);
  EXPECT_EQ(draft.to_json()["attnTp"]["bufferSize"], 129);
  EXPECT_EQ(first.to_json()["attnTp"]["bufferSize"], 128);
  options.buffer_offset(2);
  MappingNPU second_model("", /*world_size=*/16, /*rank=*/0, options);
  EXPECT_EQ(second_model.to_json()["attnTp"]["bufferSize"], 130);
  options.buffer_offset(3);
  MappingNPU second_model_draft("", /*world_size=*/16, /*rank=*/1, options);
  EXPECT_EQ(second_model_draft.to_json()["attnTp"]["bufferSize"], 131);
}

TEST(TestMappingNPU, ModelOffsetsSeparateActualAtbCommunicationCache) {
  ScopedEnvVar hccl_buffsize("HCCL_BUFFSIZE");
  hccl_buffsize.set("128");
  atb_speed::ExternalCommManager manager;
  manager.SetLcclCommDomainRange(/*lowerBound=*/0, /*upperBound=*/65536);
  manager.Init(/*worldSize=*/2,
               /*subCommRankId=*/0,
               /*backend=*/"lccl",
               /*rankTableFile=*/"",
               /*streamId=*/0);
  const auto domain = [&manager](int32_t buffer_offset) {
    MappingNPU::Options options = get_mapping_options();
    options.buffer_offset(buffer_offset);
    MappingNPU mapping("", /*world_size=*/16, /*rank=*/0, options);
    const uint32_t buffer_size = mapping.to_json()["attnTp"]["bufferSize"];
    return manager.GetCommDomain(/*groupId=*/0,
                                 /*rankIds=*/{0, 1},
                                 /*subCommRankId=*/0,
                                 /*backend=*/"lccl",
                                 buffer_size,
                                 /*streamId=*/0);
  };
  const std::string target = domain(0);
  const std::string draft = domain(1);
  const std::string second_target = domain(2);
  const std::string second_draft = domain(3);
  EXPECT_EQ(domain(0), target);
  EXPECT_EQ(domain(2), second_target);
  EXPECT_NE(target, draft);
  EXPECT_NE(target, second_target);
  EXPECT_NE(draft, second_target);
  EXPECT_NE(second_target, second_draft);
}

}  // namespace xllm
