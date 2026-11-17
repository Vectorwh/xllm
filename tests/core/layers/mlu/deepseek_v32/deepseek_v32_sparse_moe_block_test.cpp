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

#include "layers/mlu/deepseek_v32/deepseek_v32_sparse_moe_block.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v32/post_attn_layout.h"
#include "layers/mlu/fused_moe.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/model_stream_registry.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
namespace {
using Weights = std::unordered_map<std::string, torch::Tensor>;

class MergeOutputProcessGroup final : public test::MockProcessGroup {
 public:
  explicit MergeOutputProcessGroup(const torch::Device& device)
      : test::MockProcessGroup(device, /*rank=*/0, /*world_size=*/2) {}

  c10::intrusive_ptr<c10d::Work> allgather_base_async(
      const torch::Tensor& input,
      torch::Tensor& output) override {
    inputs_.emplace_back(input.clone());
    return test::MockProcessGroup::allgather_base_async(input, output);
  }

  const std::vector<torch::Tensor>& inputs() const { return inputs_; }

 private:
  std::vector<torch::Tensor> inputs_;
};

Weights create_moe_bf16_weights(const ModelArgs& model_args,
                                const torch::TensorOptions& options,
                                int64_t n_shared_experts);

Weights create_moe_bf16_weights(const ModelArgs& model_args,
                                const torch::TensorOptions& options,
                                int64_t n_shared_experts) {
  std::unordered_map<std::string, torch::Tensor> weight_dict;
  const int64_t num_experts = model_args.n_routed_experts();
  const int64_t hidden_size = model_args.hidden_size();
  const int64_t intermediate_size = model_args.moe_intermediate_size();

  for (int64_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    const std::string expert_prefix =
        "experts." + std::to_string(expert_id) + ".";
    const std::string seed_prefix =
        "deepseek_v2_sparse_moe_block.expert_" + std::to_string(expert_id);
    weight_dict[expert_prefix + "gate_proj.weight"] =
        test::seeded_tensor(seed_prefix + ".gate_proj",
                            {intermediate_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict[expert_prefix + "up_proj.weight"] =
        test::seeded_tensor(seed_prefix + ".up_proj",
                            {intermediate_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict[expert_prefix + "down_proj.weight"] =
        test::seeded_tensor(seed_prefix + ".down_proj",
                            {hidden_size, intermediate_size},
                            torch::kBFloat16,
                            options.device());
  }

  weight_dict["gate.weight"] =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.gate",
                          {num_experts, hidden_size},
                          torch::kBFloat16,
                          options.device());

  if (n_shared_experts > 0) {
    const int64_t shared_size = intermediate_size * n_shared_experts;
    weight_dict["shared_experts.gate_proj.weight"] =
        test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.gate_proj",
                            {shared_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict["shared_experts.up_proj.weight"] =
        test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.up_proj",
                            {shared_size, hidden_size},
                            torch::kBFloat16,
                            options.device());
    weight_dict["shared_experts.down_proj.weight"] =
        test::seeded_tensor("deepseek_v2_sparse_moe_block.shared.down_proj",
                            {hidden_size, shared_size},
                            torch::kBFloat16,
                            options.device());
  }

  return weight_dict;
}

}  // namespace

namespace {
void moe_set_pg_ctx(ParallelArgs& parallel_args_,
                    torch::TensorOptions& options_,
                    std::unique_ptr<test::MockProcessGroup>& global_pg_,
                    std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                    std::unique_ptr<test::MockProcessGroup>& single_rank_pg_);
void moe_set_tp_ctx(ParallelArgs& parallel_args_,
                    torch::TensorOptions& options_,
                    std::unique_ptr<test::MockProcessGroup>& global_pg_,
                    std::unique_ptr<test::MockProcessGroup>& dp_pg_,
                    std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                    std::unique_ptr<test::MockProcessGroup>& single_rank_pg_,
                    int64_t world_size,
                    int64_t ep_size);
void moe_set_tp_dp_ctx(ParallelArgs& parallel_args_,
                       torch::TensorOptions& options_,
                       std::unique_ptr<test::MockProcessGroup>& global_pg_,
                       std::unique_ptr<test::MockProcessGroup>& dp_pg_,
                       std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                       std::unique_ptr<test::MockProcessGroup>& single_rank_pg_,
                       int64_t world_size,
                       int64_t dp_size,
                       int64_t tp_size,
                       int64_t ep_size);
torch::Tensor moe_mat(torch::TensorOptions& options_,
                      int64_t rows,
                      const std::vector<float>& vals);
torch::TensorOptions moe_fp32_opts(torch::TensorOptions& options_);
std::unordered_map<std::string, torch::Tensor> moe_create_fp_weights(
    ModelArgs& model_args_,
    torch::TensorOptions& options_,
    int64_t n_shared_experts);
DeepseekV32SparseMoEBlock moe_create_block(
    ModelArgs& model_args_,
    QuantArgs& quant_args_,
    ParallelArgs& parallel_args_,
    torch::TensorOptions& options_,
    std::shared_ptr<ModelStreamRegistry>& stream_registry_);
FusedMoE moe_create_raw_moe(
    ModelArgs& model_args_,
    QuantArgs& quant_args_,
    ParallelArgs& parallel_args_,
    torch::TensorOptions& options_,
    std::shared_ptr<ModelStreamRegistry>& stream_registry_);
torch::Tensor moe_run_comm(std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                           torch::Tensor x,
                           ProcessGroup* pg);
torch::Tensor moe_run_reduce(std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                             torch::Tensor x,
                             ProcessGroup* pg);
void moe_sync_dev(torch::TensorOptions& options_);
v32_cp::DeepseekV32CPContext moe_make_cp_ctx(
    std::unique_ptr<test::MockProcessGroup>& tp_pg_,
    std::vector<int32_t> tokens_per_rank = {2, 2});
void moe_set_pg_ctx(ParallelArgs& parallel_args_,
                    torch::TensorOptions& options_,
                    std::unique_ptr<test::MockProcessGroup>& global_pg_,
                    std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                    std::unique_ptr<test::MockProcessGroup>& single_rank_pg_) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  tp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/1);

  parallel_args_ = ParallelArgs(/*rank=*/0, /*world_size=*/2, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  parallel_args_.single_rank_group_ = single_rank_pg_.get();
  parallel_args_.cp_group_ = tp_pg_.get();
  parallel_args_.ep_size_ = 1;
}
void moe_set_tp_ctx(ParallelArgs& parallel_args_,
                    torch::TensorOptions& options_,
                    std::unique_ptr<test::MockProcessGroup>& global_pg_,
                    std::unique_ptr<test::MockProcessGroup>& dp_pg_,
                    std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                    std::unique_ptr<test::MockProcessGroup>& single_rank_pg_,
                    int64_t world_size,
                    int64_t ep_size) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, world_size);
  tp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, world_size);
  single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/1);
  dp_pg_.reset();

  parallel_args_ =
      ParallelArgs(/*rank=*/0, world_size, /*dp_size=*/1, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  parallel_args_.single_rank_group_ = single_rank_pg_.get();
  parallel_args_.cp_group_ = tp_pg_.get();
  parallel_args_.dp_local_process_group_ = nullptr;
  parallel_args_.ep_size_ = ep_size;
  parallel_args_.moe_ep_group_ = global_pg_.get();
  parallel_args_.moe_tp_group_ = tp_pg_.get();
}
void moe_set_tp_dp_ctx(ParallelArgs& parallel_args_,
                       torch::TensorOptions& options_,
                       std::unique_ptr<test::MockProcessGroup>& global_pg_,
                       std::unique_ptr<test::MockProcessGroup>& dp_pg_,
                       std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                       std::unique_ptr<test::MockProcessGroup>& single_rank_pg_,
                       int64_t world_size,
                       int64_t dp_size,
                       int64_t tp_size,
                       int64_t ep_size) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, world_size);
  dp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, dp_size);
  tp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, tp_size);
  single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/1);

  parallel_args_ =
      ParallelArgs(/*rank=*/0, world_size, dp_size, global_pg_.get());
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  parallel_args_.single_rank_group_ = single_rank_pg_.get();
  parallel_args_.cp_group_ = tp_pg_.get();
  parallel_args_.dp_local_process_group_ = dp_pg_.get();
  parallel_args_.ep_size_ = ep_size;
  parallel_args_.moe_ep_group_ = global_pg_.get();
  parallel_args_.moe_tp_group_ = tp_pg_.get();
}
torch::Tensor moe_mat(torch::TensorOptions& options_,
                      int64_t rows,
                      const std::vector<float>& vals) {
  return torch::tensor(vals, moe_fp32_opts(options_)).reshape({rows, 2});
}
torch::TensorOptions moe_fp32_opts(torch::TensorOptions& options_) {
  return torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(options_.device());
}
std::unordered_map<std::string, torch::Tensor> moe_create_fp_weights(
    ModelArgs& model_args_,
    torch::TensorOptions& options_,
    int64_t n_shared_experts) {
  return create_moe_bf16_weights(model_args_, options_, n_shared_experts);
}
DeepseekV32SparseMoEBlock moe_create_block(
    ModelArgs& model_args_,
    QuantArgs& quant_args_,
    ParallelArgs& parallel_args_,
    torch::TensorOptions& options_,
    std::shared_ptr<ModelStreamRegistry>& stream_registry_) {
  return DeepseekV32SparseMoEBlock(
      model_args_,
      quant_args_,
      parallel_args_,
      options_,
      stream_registry_->get(ExecutionStreamRole::COMMUNICATION),
      stream_registry_->get(ExecutionStreamRole::AUXILIARY_COMPUTE));
}
FusedMoE moe_create_raw_moe(
    ModelArgs& model_args_,
    QuantArgs& quant_args_,
    ParallelArgs& parallel_args_,
    torch::TensorOptions& options_,
    std::shared_ptr<ModelStreamRegistry>& stream_registry_) {
  const FusedMoEArgs moe_args{.is_gated = true,
                              .enable_result_reduction = false};
  return FusedMoE(
      model_args_,
      moe_args,
      quant_args_,
      parallel_args_,
      options_,
      stream_registry_->get(ExecutionStreamRole::COMMUNICATION),
      stream_registry_->get(ExecutionStreamRole::AUXILIARY_COMPUTE));
}
torch::Tensor moe_run_comm(std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                           torch::Tensor x,
                           ProcessGroup* pg) {
  return x + (pg == tp_pg_.get() ? 10.0f : 20.0f);
}
torch::Tensor moe_run_reduce(std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                             torch::Tensor x,
                             ProcessGroup* pg) {
  return x + (pg == tp_pg_.get() ? 100.0f : 200.0f);
}
void moe_sync_dev(torch::TensorOptions& options_) {
  xllm::Device(options_.device()).synchronize_default_stream();
}
v32_cp::DeepseekV32CPContext moe_make_cp_ctx(
    std::unique_ptr<test::MockProcessGroup>& tp_pg_,
    std::vector<int32_t> tokens_per_rank) {
  v32_cp::DeepseekV32CPContext cp_ctx;
  cp_ctx.rank = 0;
  cp_ctx.process_group = tp_pg_.get();
  cp_ctx.comm_plan.tokens_per_rank = std::move(tokens_per_rank);
  cp_ctx.comm_plan.token_num_offset = 0;
  cp_ctx.comm_plan.ffn_can_rs =
      v32_cp::can_ffn_rs(cp_ctx.comm_plan.tokens_per_rank);
  return cp_ctx;
}
}  // namespace

namespace {

class DeepseekV2SparseMoEBlockTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    stream_registry_ = std::make_shared<ModelStreamRegistry>(options_.device());
    model_args_ = test::create_default_model_args();
    model_args_.hidden_size() = 256;
    model_args_.moe_intermediate_size() = 256;
    model_args_.n_routed_experts() = 4;
    model_args_.num_experts_per_tok() = 2;
    model_args_.n_group() = 1;
    model_args_.topk_group() = 2;
    model_args_.n_shared_experts() = 1;
    model_args_.routed_scaling_factor() = 1.0f;
    model_args_.norm_topk_prob() = true;
    model_args_.hidden_act() = "silu";
    model_args_.scoring_func() = "softmax";
    model_args_.topk_method() = "greedy";
    moe_set_pg_ctx(
        parallel_args_, options_, global_pg_, tp_pg_, single_rank_pg_);
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::shared_ptr<ModelStreamRegistry> stream_registry_;
  std::unique_ptr<test::MockProcessGroup> global_pg_;
  std::unique_ptr<test::MockProcessGroup> dp_pg_;
  std::unique_ptr<test::MockProcessGroup> tp_pg_;
  std::unique_ptr<test::MockProcessGroup> single_rank_pg_;
};

}  // namespace

namespace {

class ScopedExpertParallelDegree final {
 public:
  explicit ScopedExpertParallelDegree(int32_t value)
      : config_(xllm::EPLBConfig::get_instance()),
        original_(config_.expert_parallel_degree()) {
    config_.expert_parallel_degree(value);
  }

  ~ScopedExpertParallelDegree() { config_.expert_parallel_degree(original_); }

 private:
  xllm::EPLBConfig& config_;
  int32_t original_;
};

}  // namespace

TEST_F(DeepseekV2SparseMoEBlockTest, PrepInDpGatherBuildsLocalSkip) {
  moe_set_tp_dp_ctx(parallel_args_,
                    options_,
                    global_pg_,
                    dp_pg_,
                    tp_pg_,
                    single_rank_pg_,
                    /*world_size=*/4,
                    /*dp_size=*/2,
                    /*tp_size=*/2,
                    /*ep_size=*/4);
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  auto attn_out = moe_mat(
      options_, /*rows=*/4, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  auto residual =
      moe_mat(options_,
              /*rows=*/4,
              {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f});
  auto full_tokens =
      moe_mat(options_,
              /*rows=*/4,
              {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f, 77.0f, 88.0f});
  tp_pg_->set_allgather_outputs(
      {full_tokens.slice(0, 0, 2), full_tokens.slice(0, 2, 4)});

  auto prep = block->prep_in(attn_out,
                             residual,
                             input_params,
                             block->plan_exec(input_params),
                             PostAttnLayout::TP_SHARD);

  EXPECT_TRUE(prep.need_dp_gather);
  EXPECT_FALSE(prep.need_tp_pad);
  EXPECT_FALSE(prep.pad_info.active);
  test::verify_tensor_close(prep.ffn_in, full_tokens.slice(0, 0, 2));
  test::verify_tensor_close(
      prep.skip_local,
      moe_mat(
          options_, /*rows=*/3, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest,
       PlanExecUsesAll2AllOnlyWhenEveryDpRankDecodes) {
  ScopedExpertParallelDegree expert_parallel_degree(/*value=*/2);
  moe_set_tp_dp_ctx(parallel_args_,
                    options_,
                    global_pg_,
                    dp_pg_,
                    tp_pg_,
                    single_rank_pg_,
                    /*world_size=*/4,
                    /*dp_size=*/2,
                    /*tp_size=*/2,
                    /*ep_size=*/4);
  parallel_args_.tp_size(2);
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);

  ModelInputParams decode_params;
  decode_params.parallel.dp_global_token_nums = {1, 1};
  decode_params.parallel.dp_is_decode = {1, 1};
  const auto decode_cfg = block->plan_exec(decode_params);
  EXPECT_TRUE(decode_cfg.enable_all2all);
  EXPECT_FALSE(decode_cfg.need_dp_gather);

  ModelInputParams mixed_params;
  mixed_params.parallel.dp_global_token_nums = {2, 1};
  mixed_params.parallel.dp_is_decode = {0, 1};
  const auto mixed_cfg = block->plan_exec(mixed_params);
  EXPECT_FALSE(mixed_cfg.enable_all2all);
  EXPECT_TRUE(mixed_cfg.need_dp_gather);
}

TEST_F(DeepseekV2SparseMoEBlockTest, GatherInDpGatherRebuildsGlobalTokens) {
  moe_set_tp_dp_ctx(parallel_args_,
                    options_,
                    global_pg_,
                    dp_pg_,
                    tp_pg_,
                    single_rank_pg_,
                    /*world_size=*/4,
                    /*dp_size=*/2,
                    /*tp_size=*/2,
                    /*ep_size=*/4);
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  DeepseekV32SparseMoEBlockImpl::PrepOut prep;
  prep.ffn_in = moe_mat(options_, /*rows=*/2, {11.0f, 22.0f, 33.0f, 44.0f});
  prep.need_dp_gather = true;
  auto dp0_tp1 = moe_mat(options_, /*rows=*/2, {55.0f, 66.0f, 0.0f, 0.0f});
  auto dp1_tp0 = moe_mat(options_, /*rows=*/2, {101.0f, 202.0f, 0.0f, 0.0f});
  auto dp1_tp1 = torch::zeros_like(dp1_tp0);
  global_pg_->set_allgather_outputs({prep.ffn_in, dp0_tp1, dp1_tp0, dp1_tp1});

  auto gathered = block->gather_in(prep, input_params);

  test::verify_tensor_close(
      gathered,
      moe_mat(options_,
              /*rows=*/4,
              {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f, 101.0f, 202.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest, PrepInAll2AllPadsTpShardInput) {
  moe_set_tp_ctx(parallel_args_,
                 options_,
                 global_pg_,
                 dp_pg_,
                 tp_pg_,
                 single_rank_pg_,
                 /*world_size=*/2,
                 /*ep_size=*/2);
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {1, 1};
  input_params.parallel.dp_is_decode = {1, 1};
  auto attn_out =
      moe_mat(options_, /*rows=*/3, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto residual =
      moe_mat(options_, /*rows=*/3, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});

  auto prep =
      block->prep_in(attn_out,
                     residual,
                     input_params,
                     DeepseekV32SparseMoEBlockImpl::ExecCfg{
                         .enable_all2all = true, .need_dp_gather = false},
                     PostAttnLayout::TP_SHARD);

  EXPECT_FALSE(prep.need_dp_gather);
  EXPECT_TRUE(prep.need_tp_pad);
  EXPECT_TRUE(prep.pad_info.active);
  EXPECT_EQ(prep.pad_info.original_tokens, 3);
  EXPECT_EQ(prep.pad_info.padded_tokens, 4);
  test::verify_tensor_close(
      prep.ffn_in, moe_mat(options_, /*rows=*/2, {11.0f, 22.0f, 33.0f, 44.0f}));
  test::verify_tensor_close(prep.skip_local, prep.ffn_in);
}

TEST_F(DeepseekV2SparseMoEBlockTest, MergeOutTpPadGathersAndUnpads) {
  moe_set_tp_ctx(parallel_args_,
                 options_,
                 global_pg_,
                 dp_pg_,
                 tp_pg_,
                 single_rank_pg_,
                 /*world_size=*/2,
                 /*ep_size=*/2);
  auto merge_pg = std::make_unique<MergeOutputProcessGroup>(options_.device());
  parallel_args_.tp_group_ = merge_pg.get();
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);

  ModelInputParams input_params;
  DeepseekV32SparseMoEBlockImpl::PrepOut prep;
  prep.need_tp_pad = true;
  prep.pad_info = {.original_tokens = 3, .padded_tokens = 4, .active = true};
  auto shard0 = moe_mat(options_, /*rows=*/2, {1.0f, 2.0f, 3.0f, 4.0f});
  auto shard1 = moe_mat(options_, /*rows=*/2, {5.0f, 6.0f, 0.0f, 0.0f});
  prep.skip_local = moe_mat(options_, /*rows=*/2, {10.0f, 20.0f, 30.0f, 40.0f});
  auto peer_skip = moe_mat(options_, /*rows=*/2, {50.0f, 60.0f, 90.0f, 90.0f});
  merge_pg->set_allgather_outputs(
      {shard0 + prep.skip_local, shard1 + peer_skip});

  auto merged = block->merge_out(shard0, prep, input_params);

  ASSERT_EQ(merge_pg->inputs().size(), 1U);
  test::verify_tensor_close(merge_pg->inputs().front(),
                            shard0 + prep.skip_local);
  test::verify_tensor_close(
      merged,
      moe_mat(
          options_, /*rows=*/3, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest, MergeOutDpGatherSlicesLocalTokens) {
  moe_set_tp_dp_ctx(parallel_args_,
                    options_,
                    global_pg_,
                    dp_pg_,
                    tp_pg_,
                    single_rank_pg_,
                    /*world_size=*/4,
                    /*dp_size=*/2,
                    /*tp_size=*/2,
                    /*ep_size=*/4);
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);

  ModelInputParams input_params;
  input_params.parallel.dp_global_token_nums = {3, 1};
  DeepseekV32SparseMoEBlockImpl::PrepOut prep;
  prep.skip_local =
      moe_mat(options_, /*rows=*/3, {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f});
  prep.need_dp_gather = true;
  auto ffn_out =
      moe_mat(options_,
              /*rows=*/4,
              {101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f});

  auto merged = block->merge_out(ffn_out, prep, input_params);

  test::verify_tensor_close(
      merged,
      moe_mat(options_,
              /*rows=*/3,
              {112.0f, 124.0f, 136.0f, 148.0f, 160.0f, 172.0f}));
}

TEST_F(DeepseekV2SparseMoEBlockTest,
       Glm5NextClampedExpertsRepeatWithCachedKernel) {
  model_args_.model_type() = "glm5_next";
  model_args_.swiglu_limit() = 10.0f;
  moe_set_tp_ctx(parallel_args_,
                 options_,
                 global_pg_,
                 dp_pg_,
                 tp_pg_,
                 single_rank_pg_,
                 /*world_size=*/1,
                 /*ep_size=*/1);
  auto moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/1));
  moe->load_state_dict(state_dict);
  const torch::Tensor input =
      test::seeded_tensor("glm5_next_clamped_experts.input",
                          {4, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  const torch::Tensor original = input.clone();
  const torch::Tensor first =
      moe->forward_experts(input, /*enable_all2all_communication=*/false);
  const torch::Tensor second =
      moe->forward_experts(input, /*enable_all2all_communication=*/false);
  moe_sync_dev(options_);
  EXPECT_EQ(first.sizes(), input.sizes());
  EXPECT_TRUE(torch::isfinite(first.cpu()).all().item<bool>());
  EXPECT_TRUE(torch::equal(first.cpu(), second.cpu()));
  EXPECT_TRUE(torch::equal(input.cpu(), original.cpu()));
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardReducePathCombinesSharedAndRouted) {
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  auto raw_moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states = test::seeded_tensor("deepseek_v2_sparse_moe_block.input",
                                           {4, model_args_.hidden_size()},
                                           torch::kBFloat16,
                                           options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV32SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [](ProcessGroup*) { return false; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return moe_run_comm(tp_pg_, std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return moe_run_reduce(tp_pg_, std::move(x), pg);
              }),
      });

  moe_sync_dev(options_);

  auto expected = moe_run_reduce(tp_pg_, routed, tp_pg_.get()) + shared;
  EXPECT_FALSE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 0);
  EXPECT_EQ(reduce_calls, 1);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardReduceOverlapUsesAsyncReduceFns) {
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  auto raw_moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.async_reduce",
                          {4, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());

  int comm_calls = 0;
  int reduce_calls = 0;
  int launch_reduce_calls = 0;
  int finish_reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV32SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [](ProcessGroup*) { return false; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return moe_run_comm(tp_pg_, std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return moe_run_reduce(tp_pg_, std::move(x), pg);
              }),
          .launch_reduce = std::function<parallel_state::ReduceAsyncCtx(
              torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++launch_reduce_calls;
                return parallel_state::ReduceAsyncCtx{
                    .tensor = moe_run_reduce(tp_pg_, std::move(x), pg),
                };
              }),
          .finish_reduce =
              std::function<torch::Tensor(parallel_state::ReduceAsyncCtx)>(
                  [&](parallel_state::ReduceAsyncCtx ctx) {
                    ++finish_reduce_calls;
                    return std::move(ctx.tensor);
                  }),
      });

  moe_sync_dev(options_);

  auto expected = moe_run_reduce(tp_pg_, routed, tp_pg_.get()) + shared;
  EXPECT_FALSE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 0);
  EXPECT_EQ(reduce_calls, 0);
  EXPECT_EQ(launch_reduce_calls, 1);
  EXPECT_EQ(finish_reduce_calls, 1);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardKeepLocalUsesCommPath) {
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  auto raw_moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states = test::seeded_tensor("deepseek_v2_sparse_moe_block.comm",
                                           {4, model_args_.hidden_size()},
                                           torch::kBFloat16,
                                           options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV32SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [](ProcessGroup*) { return true; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return moe_run_comm(tp_pg_, std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return moe_run_reduce(tp_pg_, std::move(x), pg);
              }),
      });

  moe_sync_dev(options_);

  auto expected = moe_run_comm(tp_pg_, routed, tp_pg_.get()) +
                  moe_run_comm(tp_pg_, shared, single_rank_pg_.get());
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 2);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardIgnoresSharedLocalGate) {
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  auto raw_moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.shared_fallback",
                          {4, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());

  ProcessGroup* routed_pg = tp_pg_.get();
  ProcessGroup* shared_pg = single_rank_pg_.get();
  ASSERT_NE(routed_pg, shared_pg);
  ASSERT_NE(shared_pg, nullptr);

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV32SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* pg) { return pg == routed_pg; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return moe_run_comm(tp_pg_, std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return moe_run_reduce(tp_pg_, std::move(x), pg);
              }),
      });

  moe_sync_dev(options_);

  auto expected = moe_run_comm(tp_pg_, routed, routed_pg) +
                  moe_run_comm(tp_pg_, shared, shared_pg);
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 2);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardWithoutSharedIgnoresNullSharedPg) {
  model_args_.n_shared_experts() = 0;
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  auto raw_moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/0));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.no_shared",
                          {4, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto routed = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  ASSERT_FALSE(raw_moe->forward_shared(hidden_states).defined());

  ProcessGroup* routed_pg = tp_pg_.get();

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward(
      hidden_states,
      /*enable_moe_all2all=*/false,
      DeepseekV32SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* pg) { return pg == routed_pg; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return moe_run_comm(tp_pg_, std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return moe_run_reduce(tp_pg_, std::move(x), pg);
              }),
      });

  moe_sync_dev(options_);

  auto expected = moe_run_comm(tp_pg_, routed, routed_pg);
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 1);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardCPAddsLocalSharedAfterRoutedComm) {
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  auto raw_moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto local = test::seeded_tensor("deepseek_v2_sparse_moe_block.cp_local",
                                   {2, model_args_.hidden_size()},
                                   torch::kBFloat16,
                                   options_.device());
  auto remote = test::seeded_tensor("deepseek_v2_sparse_moe_block.cp_remote",
                                    {2, model_args_.hidden_size()},
                                    torch::kBFloat16,
                                    options_.device());
  tp_pg_->set_allgather_outputs({local, remote});
  auto cp_ctx = moe_make_cp_ctx(tp_pg_);

  auto gathered = torch::cat({local, remote}, 0);
  auto routed = raw_moe->forward_experts(
      gathered, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(local);
  ProcessGroup* routed_pg = tp_pg_.get();
  const int64_t local_token_num = local.size(0);

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward_cp(
      local,
      cp_ctx,
      DeepseekV32SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* pg) { return pg == routed_pg; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                if (pg == routed_pg) {
                  x = x.slice(0, 0, local_token_num);
                }
                return moe_run_comm(tp_pg_, std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return moe_run_reduce(tp_pg_, std::move(x), pg);
              }),
      });

  moe_sync_dev(options_);

  auto expected =
      moe_run_comm(tp_pg_, routed.slice(0, 0, local_token_num), routed_pg) +
      shared;
  EXPECT_TRUE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 1);
  EXPECT_EQ(reduce_calls, 0);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

TEST_F(DeepseekV2SparseMoEBlockTest, ForwardCPFallsBackWhenLocalKeepOff) {
  auto block = moe_create_block(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  auto raw_moe = moe_create_raw_moe(
      model_args_, quant_args_, parallel_args_, options_, stream_registry_);
  StateDict state_dict(
      moe_create_fp_weights(model_args_, options_, /*n_shared_experts=*/1));
  block->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto local =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.cp_fallback_local",
                          {2, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  auto remote =
      test::seeded_tensor("deepseek_v2_sparse_moe_block.cp_fallback_remote",
                          {2, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  tp_pg_->set_allgather_outputs({local, remote});
  auto cp_ctx = moe_make_cp_ctx(tp_pg_);

  auto gathered = torch::cat({local, remote}, 0);
  auto routed = raw_moe->forward_experts(
      gathered, /*enable_all2all_communication=*/false);
  auto shared = raw_moe->forward_shared(gathered);
  ProcessGroup* routed_pg = tp_pg_.get();

  int comm_calls = 0;
  int reduce_calls = 0;
  auto result = block->forward_cp(
      local,
      cp_ctx,
      DeepseekV32SparseMoEBlockImpl::CommFns{
          .can_keep_local = std::function<bool(ProcessGroup*)>(
              [&](ProcessGroup* /*pg*/) { return false; }),
          .comm = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++comm_calls;
                return moe_run_comm(tp_pg_, std::move(x), pg);
              }),
          .reduce = std::function<torch::Tensor(torch::Tensor, ProcessGroup*)>(
              [&](torch::Tensor x, ProcessGroup* pg) {
                ++reduce_calls;
                return moe_run_reduce(tp_pg_, std::move(x), pg);
              }),
      });

  moe_sync_dev(options_);

  auto expected = moe_run_reduce(tp_pg_, routed, routed_pg) + shared;
  EXPECT_FALSE(result.keep_local_output);
  EXPECT_EQ(comm_calls, 0);
  EXPECT_EQ(reduce_calls, 1);
  test::verify_tensor_close(result.output, expected, 1e-3, 1e-4);
}

}  // namespace layer
}  // namespace xllm
