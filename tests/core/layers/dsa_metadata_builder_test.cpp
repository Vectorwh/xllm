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

#include "core/layers/common/dsa_metadata_builder.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/framework/model/model_input_params.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/dsa_metadata.h"

namespace xllm::layer {
namespace {

TEST(DSAMetadataBuilderTest,
     ModelBindingsSeparateLogicalTablesFromPhysicalGroups) {
  for (const auto& ratios : std::vector<std::vector<int32_t>>{
           {4, 128}, {128, 4}, {1, 2}, {2, 1}, {1}, {2}}) {
    for (const bool graph : {false, true}) {
      SCOPED_TRACE(graph);
      const int32_t span = ratios.front() > 2 ? 2048 : 128;
      ModelInputParams params;
      params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
      params.meta.num_sequences = 1;
      params.meta.actual_num_sequences = 1;
      params.meta.q_max_seq_len = 128;
      params.meta.kv_max_seq_len = span + 128;
      params.enable_graph = graph;
      params.attention.host.q_seq_lens = {128};
      params.attention.host.kv_seq_lens = {span + 128};
      params.attention.host.q_cu_seq_lens = {128};
      params.attention.device.q_seq_lens =
          torch::tensor({0, 128}, torch::kInt32);
      params.attention.device.kv_seq_lens =
          torch::tensor({0, span + 128}, torch::kInt32);
      params.attention.device.q_cu_seq_lens =
          torch::tensor({128}, torch::kInt32);
      params.attention.device.block_tables =
          torch::zeros({1, 32}, torch::kInt32);
      params.attention.device.new_cache_slots =
          torch::arange(128, torch::kInt32);
      const auto compressed =
          torch::tensor({7, 2, 9}, torch::kInt32).view({1, 3});
      params.multi_block_tables = {
          torch::arange(32, torch::kInt32).view({1, 32}), compressed};
      // Physical TOKEN groups precede SWA, while logical tables remain SWA/KV.
      std::vector<DSAGroupInfo> groups;
      groups.reserve(ratios.size() + 1);
      std::vector<std::vector<DSACacheInfo>> caches;
      caches.reserve(ratios.size());
      for (const int32_t ratio : ratios) {
        const int32_t gid = static_cast<int32_t>(groups.size());
        groups.emplace_back(
            DSAGroupInfo{DSACacheType::TOKEN, ratio, span / ratio});
        caches.emplace_back(std::vector<DSACacheInfo>{
            {gid, DSACacheType::TOKEN, ratio, span / ratio}});
      }
      groups.emplace_back(DSAGroupInfo{DSACacheType::SLIDING_WINDOW, 1, 128});
      EXPECT_EQ(dsa_block_table_count(groups), 2);
      const auto positions = torch::arange(span, span + 128, torch::kInt32);
      const auto metadata =
          DSAMetadataBuilder::build(params, positions, {}, caches, groups);
      ASSERT_NE(metadata.dsa_metadata, nullptr);
      for (size_t layer_id = 0; layer_id < ratios.size(); ++layer_id) {
        const auto& dsa = *metadata.dsa_metadata;
        ASSERT_FALSE(dsa.block_tables[layer_id][0].numel() == 0);
        ASSERT_GT(dsa.slot_mappings[layer_id][0].numel(), 0);
        EXPECT_EQ(dsa.slot_mappings[layer_id][0][0].item<int64_t>(),
                  2 * (span / ratios[layer_id]));
      }
    }
  }
}

TEST(DSAMetadataBuilderTest, DerivesLogicalTableBindingsFromCacheType) {
  const DSAGroupInfo swa(DSACacheType::SLIDING_WINDOW, 1, 128);
  const DSAGroupInfo token(DSACacheType::TOKEN, 4, 512);
  EXPECT_EQ(static_cast<int32_t>(swa.type), 0);
  EXPECT_EQ(static_cast<int32_t>(token.type), 1);
  EXPECT_EQ(dsa_block_table_count({swa, token}), 2);
}

TEST(DSAMetadataBuilderTest, GraphPaddingDoesNotRequireLogicalTableRows) {
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 16;
  params.meta.actual_num_sequences = 12;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 128;
  params.enable_graph = true;
  params.attention.host.q_seq_lens.assign(16, 0);
  params.attention.host.kv_seq_lens.assign(16, 0);
  std::fill_n(params.attention.host.q_seq_lens.begin(), 12, 1);
  std::fill_n(params.attention.host.kv_seq_lens.begin(), 12, 128);
  params.attention.device.q_seq_lens =
      torch::tensor(params.attention.host.q_seq_lens, torch::kInt32);
  params.attention.device.kv_seq_lens =
      torch::tensor(params.attention.host.kv_seq_lens, torch::kInt32);
  params.attention.device.block_tables = torch::zeros({16, 1}, torch::kInt32);
  params.attention.device.new_cache_slots = torch::zeros({16}, torch::kInt32);
  params.multi_block_tables = {torch::full({12, 1}, 3, torch::kInt32),
                               torch::full({12, 1}, 7, torch::kInt32)};
  const std::vector<DSAGroupInfo> groups = {
      {DSACacheType::SLIDING_WINDOW, 1, 128}, {DSACacheType::TOKEN, 4, 512}};
  const std::vector<std::vector<DSACacheInfo>> caches = {
      {{0, DSACacheType::SLIDING_WINDOW, 1, 128},
       {1, DSACacheType::TOKEN, 4, 512}}};
  const auto metadata = DSAMetadataBuilder::build(
      params, torch::zeros({16}, torch::kInt32), {}, caches, groups);
  ASSERT_NE(metadata.dsa_metadata, nullptr);
  const auto& dsa = *metadata.dsa_metadata;
  for (size_t cache = 0; cache < 2; ++cache) {
    EXPECT_EQ(dsa.block_tables[0][cache].size(0), 16);
    EXPECT_TRUE(
        dsa.block_tables[0][cache].slice(0, 12).eq(-1).all().item<bool>());
    EXPECT_TRUE(
        dsa.slot_mappings[0][cache].slice(0, 12).eq(-1).all().item<bool>());
  }
  EXPECT_EQ(dsa.slot_mappings[0][0][0].item<int32_t>(), 3 * 128 + 127);
  EXPECT_EQ(dsa.slot_mappings[0][1][0].item<int32_t>(), 7 * 512 + 31);
}

}  // namespace
}  // namespace xllm::layer
