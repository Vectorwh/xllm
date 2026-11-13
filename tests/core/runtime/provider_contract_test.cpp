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

#include "core/runtime/provider_contract.h"

#include <gtest/gtest.h>

#include "core/common/types.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

class DescriptorEngine final : public Engine {
 public:
  DescriptorEngine() {
    args_.dtype("bfloat16").n_heads(8).n_kv_heads(8).head_dim(64);
    tokenizer_args_.chat_template("{{ messages }}");
  }
  ForwardOutput step(std::vector<Batch>&) override { return {}; }
  void update_last_step_result(std::vector<Batch>&) override {}
  std::vector<int64_t> get_active_activation_memory() const override {
    return {};
  }
};

TEST(ProviderContractTest, RegistersActualCacheGroupsForFlatAndGroupedPools) {
  auto& model = ModelConfig::get_instance();
  auto& parallel = ParallelConfig::get_instance();
  auto& disagg = DisaggPDConfig::get_instance();
  const ModelConfig saved_model = model;
  const ParallelConfig saved_parallel = parallel;
  const DisaggPDConfig saved_disagg = disagg;
  const ScopeGuard restore_config([&] {
    model = saved_model;
    parallel = saved_parallel;
    disagg = saved_disagg;
  });
  model.backend("llm").model_id("test-model");
  parallel.tp_size(1).dp_size(1).cp_size(1).ep_size(1).kv_split_size(1);
  disagg.enable_disagg_pd(true).kv_cache_transfer_mode("PUSH");

  const DescriptorEngine engine;
  InstanceInfo instance;
  instance.name = "127.0.0.1:7442";
  instance.incarnation_id = "test-incarnation";
  instance.type = "PREFILL";
  instance.cluster_ids = {1};

  const std::vector<std::vector<uint32_t>> ratios = {
      {}, {0, 4, 128}, {0, 2, 1}};
  const std::vector<std::vector<std::string>> expected = {
      {"group:0"},
      {"group:1", "group:2", "group:3"},
      {"group:1", "group:6", "group:7"}};
  for (size_t index = 0; index < ratios.size(); ++index) {
    SCOPED_TRACE(index);
    BlockManagerPool::Options options;
    options.num_blocks(128)
        .block_size(128)
        .enable_prefix_cache(false)
        .sliding_window_size(128)
        .swa_blocks_per_seq(2)
        .swa_num_blocks(8)
        .c4_num_blocks(4)
        .c128_num_blocks(4)
        .max_seqs_per_batch(1)
        .max_tokens_per_batch(128);
    if (!ratios[index].empty()) {
      options.manager_types({1, 0, 0}).compress_ratios(ratios[index]);
      options.model_type(index == 1 ? "deepseek_v4" : "deepseek_v41");
    }
    const BlockManagerPool pool(options);
    std::string error;
    const auto config = collect_native_provider_descriptor_config(
        instance, engine, pool, &error);
    EXPECT_TRUE(config.has_value()) << error;
    if (!config.has_value()) {
      continue;
    }
    EXPECT_EQ(config->cache_groups, expected[index]);
    const auto descriptor = build_native_provider_descriptor(*config, &error);
    EXPECT_TRUE(descriptor.has_value()) << error;
  }
}

}  // namespace
}  // namespace xllm
