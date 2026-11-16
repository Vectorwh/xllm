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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_MLU)
#include "layers/mlu/attention.h"
#include "layers/mlu/deepseek_v32/deepseek_v32_decoder_layer.h"
#elif defined(USE_CUDA)
#include "layers/cuda/attention.h"
#endif
#include <unordered_map>

#include "layers/common/attention_metadata_builder.h"
#include "layers/common/dsa_topk_share_plan.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

namespace {
using Weights = std::unordered_map<std::string, torch::Tensor>;

ModelArgs create_decoder_model_args();
Weights create_decoder_weights(const ModelArgs& model_args,
                               const torch::TensorOptions& options,
                               int32_t layer_id,
                               int64_t hidden_size,
                               int64_t intermediate_size,
                               int64_t moe_intermediate_size,
                               int32_t num_routed_experts);
ModelInputParams create_decoder_prefill_params(
    int64_t batch_size,
    int64_t seq_len,
    const torch::TensorOptions& options);
KVCache create_mla_cache(const ModelArgs& model_args,
                         const torch::TensorOptions& options,
                         int64_t block_num,
                         int64_t block_size);
KVCache create_indexed_kv_cache(torch::Tensor key_cache,
                                torch::Tensor index_cache);

ModelArgs create_decoder_model_args() {
  ModelArgs args = ::xllm::layer::test::create_default_model_args();
  // Use a small but structurally valid DeepSeek-style config so the tests
  // exercise Dense/MoE/MLA wiring without allocating near-production tensors.
  args.model_type() = "deepseek_v3";
  args.dtype() = "";
  args.vocab_size() = 4096;
  args.hidden_size() = 512;
  args.n_layers() = 8;
  args.n_heads() = 8;
  args.n_kv_heads() = 8;
  args.intermediate_size() = 1024;
  args.hidden_act() = "silu";
  args.rms_norm_eps() = 1e-6f;
  args.max_position_embeddings() = 4096;
  args.eos_token_id() = 1;
  args.bos_token_id() = 0;

  // Decoder layer specific routing between MoE and Dense
  args.first_k_dense_replace() = 1;
  args.moe_layer_freq() = 1;

  // MoE-related params
  args.n_routed_experts() = 8;
  args.n_shared_experts() = 1;
  args.num_experts_per_tok() = 2;
  args.moe_intermediate_size() = 128;
  args.routed_scaling_factor() = 2.5f;
  args.norm_topk_prob() = true;
  args.n_group() = 2;
  args.topk_group() = 1;
  args.scoring_func() = "sigmoid";
  args.topk_method() = "noaux_tc";

  // Q/K/V dims used by DeepseekV32Attention
  args.qk_nope_head_dim() = 64;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 64;
  args.head_dim() = 128;  // qk_nope_head_dim + qk_rope_head_dim
  args.rotary_dim() = args.qk_rope_head_dim();

  // Rope scaling related
  args.rope_scaling_rope_type() = "deepseek_yarn";
  // The following values may be model/export dependent; set common defaults.
  args.rope_scaling_beta_fast() = 32;
  args.rope_scaling_beta_slow() = 1;
  args.rope_scaling_factor() = 40;
  args.rope_extrapolation_factor() = 1.0f;
  args.rope_scaling_mscale() = 1.0f;
  args.rope_scaling_mscale_all_dim() = 1.0f;
  args.rope_scaling_original_max_position_embeddings() = 1024;
  args.rope_scaling_attn_factor() = 1.0f;

  // Sliding window
  args.use_sliding_window() = false;
  args.sliding_window() = 512;
  args.max_window_layers() = 8;

  // LORA ranks for DeepSeek-V3
  args.q_lora_rank() = 128;
  args.kv_lora_rank() = 64;  // qk_rope_head_dim + kv_lora_rank = 128

  // extra parameters for DeepSeek-V3.2-Exp
  args.index_head_dim() = 128;
  args.index_n_heads() = 0;
  args.index_topk() = 0;
  args.enable_mla() = true;

  return args;
}

Weights create_decoder_weights(const ModelArgs& model_args,
                               const torch::TensorOptions& options,
                               int32_t layer_id,
                               int64_t hidden_size,
                               int64_t intermediate_size,
                               int64_t moe_intermediate_size,
                               int32_t num_routed_experts) {
  Weights weights;
  const auto cpu_float = torch::TensorOptions().dtype(torch::kFloat32);
  auto constant = [&](const std::string& name,
                      torch::IntArrayRef shape,
                      float value) {
    weights[name] = torch::full(shape, value, cpu_float).to(options.device());
  };
  auto quant = [&](const std::string& name,
                   int64_t out,
                   int64_t in,
                   int8_t value,
                   float scale,
                   float smooth) {
    weights[name + ".qweight"] =
        torch::full({out, in}, value, torch::kInt8).to(options.device());
    constant(name + ".per_channel_scale", {out}, scale);
    constant(name + ".smooth", {in}, smooth);
  };
  auto ffn =
      [&](const std::string& prefix, int64_t width, int8_t up, int8_t down) {
        quant(prefix + "gate_proj", width, hidden_size, up, 0.1f, 0.05f);
        quant(prefix + "up_proj", width, hidden_size, up, 0.1f, 0.05f);
        quant(prefix + "down_proj", hidden_size, width, down, 0.1f, 0.05f);
      };
  constant("input_layernorm.weight", {hidden_size}, 1.0f);
  constant("post_attention_layernorm.weight", {hidden_size}, 1.0f);
  if (layer_id < model_args.first_k_dense_replace()) {
    ffn("mlp.", intermediate_size, 5, 3);
  } else {
    const int64_t width = moe_intermediate_size > 0
                              ? moe_intermediate_size
                              : model_args.moe_intermediate_size();
    const int32_t experts = num_routed_experts > 0
                                ? num_routed_experts
                                : model_args.n_routed_experts();
    weights["mlp.gate.weight"] =
        torch::full({experts, hidden_size},
                    0.8f,
                    options.dtype(torch::kBFloat16).device(torch::kCPU))
            .to(options.device());
    weights["mlp.gate.e_score_correction_bias"] =
        torch::full({experts}, 0.1f, options.device(torch::kCPU))
            .to(options.device());
    if (model_args.n_shared_experts() > 0) {
      ffn("mlp.shared_experts.", width, 0, 0);
    }
    for (int32_t expert = 0; expert < experts; ++expert) {
      ffn("mlp.experts." + std::to_string(expert) + ".", width, 0, 0);
    }
  }
  const int64_t heads = model_args.n_heads();
  const int64_t qrank = model_args.q_lora_rank();
  const int64_t latent = model_args.kv_lora_rank();
  const int64_t rope = model_args.qk_rope_head_dim();
  const int64_t nope = model_args.qk_nope_head_dim();
  const int64_t value = model_args.v_head_dim();
  quant("self_attn.o_proj", hidden_size, heads * value, 1, 0.03f, 0.03f);
  quant("self_attn.q_b_proj", heads * (nope + rope), qrank, 1, 0.03f, 0.03f);
  constant(
      "self_attn.kv_b_proj.weight", {heads * (nope + value), latent}, 0.02f);
  constant("self_attn.kv_a_proj_with_mqa.weight",
           {latent + rope, hidden_size},
           0.02f);
  constant("self_attn.q_a_proj.weight", {qrank, hidden_size}, 0.02f);
  constant("self_attn.kv_a_layernorm.weight", {latent}, 1.0f);
  constant("self_attn.q_a_layernorm.weight", {qrank}, 1.0f);
  if (model_args.index_n_heads() > 0) {
    const int64_t dim = model_args.index_head_dim();
    const int64_t count = model_args.index_n_heads();
    constant("self_attn.indexer.k_norm.bias", {dim}, 0.0f);
    constant("self_attn.indexer.k_norm.weight", {dim}, 1.0f);
    constant(
        "self_attn.indexer.weights_proj.weight", {count, hidden_size}, 0.02f);
    constant("self_attn.indexer.wk.weight", {dim, hidden_size}, 0.02f);
    constant("self_attn.indexer.wq_b.weight", {count * dim, qrank}, 0.02f);
  }
  return weights;
}

ModelInputParams create_decoder_prefill_params(
    int64_t batch_size,
    int64_t seq_len,
    const torch::TensorOptions& options) {
  ModelInputParams input_params;
  input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input_params.meta.num_sequences = batch_size;
  input_params.meta.q_max_seq_len = seq_len;
  input_params.meta.kv_max_seq_len = batch_size * seq_len;
  input_params.attention.device.q_seq_lens = torch::arange(
      0,
      (batch_size + 1) * seq_len,
      seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options.device()));
  input_params.attention.device.kv_seq_lens = torch::arange(
      0,
      (batch_size + 1) * seq_len,
      seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options.device()));
  input_params.attention.host.q_seq_lens.resize(batch_size, seq_len);
  input_params.attention.host.kv_seq_lens.resize(batch_size, seq_len);
  input_params.attention.device.new_cache_slots = torch::arange(
      1,
      batch_size * seq_len + 1,
      torch::TensorOptions().dtype(torch::kInt32).device(options.device()));
  input_params.attention.device.block_tables = torch::zeros(
      {batch_size, batch_size * seq_len},
      torch::TensorOptions().dtype(torch::kInt32).device(options.device()));

  for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const int64_t start = batch_idx * seq_len + 1;
    input_params.attention.device.block_tables[batch_idx].index_put_(
        {torch::indexing::Slice(0, seq_len)},
        torch::arange(start,
                      start + seq_len,
                      torch::TensorOptions()
                          .dtype(torch::kInt32)
                          .device(options.device())));
  }

  return input_params.to(options.device());
}

KVCache create_mla_cache(const ModelArgs& model_args,
                         const torch::TensorOptions& options,
                         int64_t block_num,
                         int64_t block_size) {
  auto k_cache =
      torch::zeros({block_num,
                    1,
                    block_size,
                    model_args.qk_rope_head_dim() + model_args.kv_lora_rank()},
                   options);
  auto index_cache = torch::zeros(
      {block_num, 1, block_size, model_args.index_head_dim()}, options);
  return create_indexed_kv_cache(k_cache, index_cache);
}

KVCache create_indexed_kv_cache(torch::Tensor key_cache,
                                torch::Tensor index_cache) {
  return KVCache(IndexedKVCacheTensors{
      KVCacheTensors{std::move(key_cache), torch::Tensor()},
      std::move(index_cache)});
}

struct LayerCase {
  const char* name;
  int32_t layer_id;
};

class ExpertDegreeGuard {
 public:
  explicit ExpertDegreeGuard(int32_t value)
      : saved_(EPLBConfig::get_instance().expert_parallel_degree()) {
    EPLBConfig::get_instance().expert_parallel_degree(value);
  }

  ~ExpertDegreeGuard() {
    EPLBConfig::get_instance().expert_parallel_degree(saved_);
  }

 private:
  int32_t saved_;
};

}  // namespace
namespace {
using DecoderHolder = torch::nn::ModuleHolder<DeepseekV32DecoderLayerImpl>;
std::unordered_set<std::string> dec_get_child_module_names(
    const torch::nn::Module& module);
std::unordered_map<std::string, torch::Tensor> dec_create_default_test_weights(
    ModelArgs& model_args_,
    torch::TensorOptions& options_,
    int32_t layer_id,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t moe_intermediate_size = -1,
    int32_t num_routed_experts = -1);
std::unordered_map<std::string, torch::Tensor> dec_create_test_weights(
    ModelArgs& model_args_,
    torch::TensorOptions& options_,
    int32_t layer_id,
    int64_t custom_hidden_size = -1,
    int64_t custom_intermediate_size = -1,
    int64_t custom_moe_intermediate_size = -1,
    int32_t custom_num_routed_experts = -1);
DecoderHolder dec_make_decoder(ModelArgs& model_args_,
                               ModelContext& context_,
                               int32_t layer_id);
DecoderHolder dec_make_loaded_decoder(ModelArgs& model_args_,
                                      torch::TensorOptions& options_,
                                      ModelContext& context_,
                                      int32_t layer_id);

void dec_refresh_ctx(ModelArgs& model_args_,
                     QuantArgs& quant_args_,
                     ParallelArgs& parallel_args_,
                     torch::TensorOptions& options_,
                     ModelContext& context_);

void dec_set_mixed_dp_ctx(ModelArgs& model_args_,
                          QuantArgs& quant_args_,
                          ParallelArgs& parallel_args_,
                          torch::TensorOptions& options_,
                          std::unique_ptr<test::MockProcessGroup>& global_pg_,
                          std::unique_ptr<test::MockProcessGroup>& dp_pg_,
                          std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                          ModelContext& context_);

ModelInputParams dec_build_prefill_params(torch::TensorOptions& options_,
                                          int64_t batch_size,
                                          int64_t seq_len);
KVCache dec_build_indexed_cache(torch::Tensor key_cache,
                                torch::Tensor index_cache);
KVCache dec_build_cache(ModelArgs& model_args_,
                        torch::TensorOptions& options_,
                        int64_t block_num,
                        int64_t block_size);
void dec_sync_dev(torch::TensorOptions& options_);
std::unordered_set<std::string> dec_get_child_module_names(
    const torch::nn::Module& module) {
  std::unordered_set<std::string> names;
  for (const auto& named_child : module.named_children()) {
    names.insert(named_child.key());
  }
  return names;
}
std::unordered_map<std::string, torch::Tensor> dec_create_default_test_weights(
    ModelArgs& model_args_,
    torch::TensorOptions& options_,
    int32_t layer_id,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t moe_intermediate_size,
    int32_t num_routed_experts) {
  return create_decoder_weights(model_args_,
                                options_,
                                layer_id,
                                hidden_size,
                                intermediate_size,
                                moe_intermediate_size,
                                num_routed_experts);
}
std::unordered_map<std::string, torch::Tensor> dec_create_test_weights(
    ModelArgs& model_args_,
    torch::TensorOptions& options_,
    int32_t layer_id,
    int64_t custom_hidden_size,
    int64_t custom_intermediate_size,
    int64_t custom_moe_intermediate_size,
    int32_t custom_num_routed_experts) {
  int64_t test_hidden_size =
      (custom_hidden_size > 0) ? custom_hidden_size : model_args_.hidden_size();
  int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                       ? custom_intermediate_size
                                       : model_args_.intermediate_size();

  return dec_create_default_test_weights(model_args_,
                                         options_,
                                         layer_id,
                                         test_hidden_size,
                                         test_intermediate_size,
                                         custom_moe_intermediate_size,
                                         custom_num_routed_experts);
}
DecoderHolder dec_make_decoder(ModelArgs& model_args_,
                               ModelContext& context_,
                               int32_t layer_id) {
  const DsaTopkSharePlan topk_share_plan(model_args_);
  return DecoderHolder(
      DeepseekV32DecoderLayerImpl(context_, layer_id, topk_share_plan));
}
DecoderHolder dec_make_loaded_decoder(ModelArgs& model_args_,
                                      torch::TensorOptions& options_,
                                      ModelContext& context_,
                                      int32_t layer_id) {
  auto decoder = dec_make_decoder(model_args_, context_, layer_id);
  auto weight_dict = dec_create_test_weights(model_args_, options_, layer_id);
  StateDict state_dict(weight_dict);
  decoder->load_state_dict(state_dict);
  return decoder;
}

void dec_refresh_ctx(ModelArgs& model_args_,
                     QuantArgs& quant_args_,
                     ParallelArgs& parallel_args_,
                     torch::TensorOptions& options_,
                     ModelContext& context_) {
  context_ = ModelContext(parallel_args_, model_args_, quant_args_, options_);
}

void dec_set_mixed_dp_ctx(ModelArgs& model_args_,
                          QuantArgs& quant_args_,
                          ParallelArgs& parallel_args_,
                          torch::TensorOptions& options_,
                          std::unique_ptr<test::MockProcessGroup>& global_pg_,
                          std::unique_ptr<test::MockProcessGroup>& dp_pg_,
                          std::unique_ptr<test::MockProcessGroup>& tp_pg_,
                          ModelContext& context_) {
  global_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  dp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/2);
  tp_pg_ = std::make_unique<test::MockProcessGroup>(
      options_.device(), /*rank=*/0, /*world_size=*/1);

  parallel_args_ = ParallelArgs(
      /*rank=*/0, /*world_size=*/2, /*dp_size=*/2, global_pg_.get());
  parallel_args_.ep_size_ = 2;
  parallel_args_.process_group_ = global_pg_.get();
  parallel_args_.dp_local_process_group_ = dp_pg_.get();
  parallel_args_.tp_group_ = tp_pg_.get();
  parallel_args_.single_rank_group_ = tp_pg_.get();
  parallel_args_.cp_group_ = tp_pg_.get();
  parallel_args_.moe_ep_group_ = global_pg_.get();
  parallel_args_.moe_tp_group_ = tp_pg_.get();
  dec_refresh_ctx(model_args_, quant_args_, parallel_args_, options_, context_);
}

ModelInputParams dec_build_prefill_params(torch::TensorOptions& options_,
                                          int64_t batch_size,
                                          int64_t seq_len) {
  return create_decoder_prefill_params(batch_size, seq_len, options_);
}
KVCache dec_build_indexed_cache(torch::Tensor key_cache,
                                torch::Tensor index_cache) {
  return KVCache(IndexedKVCacheTensors{
      KVCacheTensors{key_cache, torch::Tensor()}, index_cache});
}
KVCache dec_build_cache(ModelArgs& model_args_,
                        torch::TensorOptions& options_,
                        int64_t block_num,
                        int64_t block_size) {
  return create_mla_cache(model_args_, options_, block_num, block_size);
}
void dec_sync_dev(torch::TensorOptions& options_) {
  xllm::Device(options_.device()).synchronize_default_stream();
}
}  // namespace

// namespace

namespace {
class DeepseekV2DecoderLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_block_size_ = KVCacheConfig::get_instance().block_size();
    quant_args_ = test::create_default_quant_args();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
    model_args_ = create_decoder_model_args();

    context_ = ModelContext(parallel_args_, model_args_, quant_args_, options_);
  }
  void TearDown() override {
    KVCacheConfig::get_instance().block_size(saved_block_size_);
  }

  int64_t saved_block_size_ = 0;
  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::unique_ptr<test::MockProcessGroup> global_pg_;
  std::unique_ptr<test::MockProcessGroup> dp_pg_;
  std::unique_ptr<test::MockProcessGroup> tp_pg_;
  std::unique_ptr<test::MockProcessGroup> single_rank_pg_;
  ModelContext context_{};
};
}  // namespace

TEST_F(DeepseekV2DecoderLayerTest,
       Constructor_WhenCreated_ThenRegistersExpectedSubmodules) {
  for (const auto& param :
       std::vector<LayerCase>{LayerCase{"Dense", 0}, LayerCase{"Moe", 5}}) {
    const auto decoder =
        dec_make_decoder(model_args_, context_, param.layer_id);

    const auto child_names = dec_get_child_module_names(*decoder);
    EXPECT_TRUE(child_names.count("self_attn")) << "self_attn missing";
    EXPECT_TRUE(child_names.count("input_layernorm"))
        << "input_layernorm missing";
    EXPECT_TRUE(child_names.count("post_attention_layernorm"))
        << "post_attention_layernorm missing";
    EXPECT_TRUE(child_names.count("mlp")) << "mlp missing";
  }
}

TEST_F(DeepseekV2DecoderLayerTest,
       LoadStateDict_WhenWeightsMatch_ThenSucceeds) {
  for (const auto& param :
       std::vector<LayerCase>{LayerCase{"Dense", 0}, LayerCase{"Moe", 5}}) {
    auto decoder = dec_make_decoder(model_args_, context_, param.layer_id);
    auto weight_dict =
        dec_create_test_weights(model_args_, options_, param.layer_id);
    StateDict state_dict(weight_dict);

    EXPECT_NO_THROW(decoder->load_state_dict(state_dict));
  }
}

TEST_F(DeepseekV2DecoderLayerTest, ForwardMixedDpMoEReturnsLocalSlice) {
  const int32_t layer_id =
      std::max<int32_t>(5, model_args_.first_k_dense_replace());
  ExpertDegreeGuard guard(/*value=*/0);
  dec_set_mixed_dp_ctx(model_args_,
                       quant_args_,
                       parallel_args_,
                       options_,
                       global_pg_,
                       dp_pg_,
                       tp_pg_,
                       context_);

  auto decoder =
      dec_make_loaded_decoder(model_args_, options_, context_, layer_id);

  auto hidden_states =
      xllm::layer::test::seeded_tensor("mixed_dp_moe.hidden_states",
                                       {2, model_args_.hidden_size()},
                                       torch::kBFloat16,
                                       options_.device());
  auto positions = torch::arange(
      0,
      2,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  ModelInputParams input_params;
  input_params.meta.batch_forward_type = BatchForwardType::PREFILL;
  input_params.meta.num_sequences = 2;
  input_params.meta.q_max_seq_len = 1;
  input_params.meta.kv_max_seq_len = 2;
  input_params.attention.device.q_seq_lens = torch::tensor(
      {0, 1, 2},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.attention.device.kv_seq_lens = torch::tensor(
      {0, 1, 2},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.attention.host.q_seq_lens = {1, 1};
  input_params.attention.host.kv_seq_lens = {1, 1};
  input_params.attention.device.new_cache_slots = torch::arange(
      1,
      3,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.attention.device.block_tables = torch::tensor(
      {{1, 0}, {2, 0}},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
  input_params.parallel.dp_global_token_nums = {2, 1};
  input_params.parallel.dp_is_decode = {0, 0};
  input_params = input_params.to(options_.device());

  auto attn_metadata =
      AttentionMetadataBuilder::build(input_params, /*enable_mla=*/true);
  auto k_cache = torch::zeros(
      {2048, 1, 1, model_args_.qk_rope_head_dim() + model_args_.kv_lora_rank()},
      options_);
  auto index_cache =
      torch::zeros({2048, 1, 1, model_args_.index_head_dim()}, options_);
  KVCache kv_cache =
      dec_build_indexed_cache(std::move(k_cache), std::move(index_cache));

  std::optional<torch::Tensor> residual = std::nullopt;
  auto output = decoder->forward(hidden_states,
                                 residual,
                                 positions,
                                 attn_metadata,
                                 kv_cache,
                                 input_params);

  dec_sync_dev(options_);

  ASSERT_EQ(output.dim(), 2);
  EXPECT_EQ(output.size(0), 2);
  EXPECT_EQ(output.size(1), model_args_.hidden_size());
  EXPECT_FALSE(residual.has_value());
}

namespace {

// Ground-truth indexer_types for GLM5.2-W4A8 (index_topk_freq=4,
// index_skip_topk_offset=3, 78 layers): 21 Full + 57 Shared. A Full layer owns
// indexer weights (layers 0/1 plus every 4th layer from 2); the rest are
// Shared and must reuse the previous Full layer's top-k.
bool is_full_indexer_layer_glm52(int32_t layer_id) {
  if (layer_id < 2) {
    return true;
  }
  return (layer_id - 2) % 4 == 0;
}

std::string build_fs_pattern_glm52(int32_t num_layers) {
  std::string pattern;
  pattern.reserve(num_layers);
  for (int32_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    pattern.push_back(is_full_indexer_layer_glm52(layer_id) ? 'F' : 'S');
  }
  return pattern;
}

}  // namespace

namespace {
constexpr int32_t kNumLayers = 78;
void dec_configure_glm5_indexer(ModelArgs& model_args_);
void dec_use_cpu_constructor_context(
    ModelArgs& model_args_,
    QuantArgs& quant_args_,
    ParallelArgs& parallel_args_,
    torch::TensorOptions& options_,
    std::unique_ptr<xllm::ProcessGroup>& mock_process_group_,
    ModelContext& context_);
void dec_configure_glm5_indexer(ModelArgs& model_args_) {
  model_args_.model_type() = "glm_moe_dsa";
  model_args_.n_layers() = kNumLayers;
  model_args_.index_n_heads() = 4;
  model_args_.index_head_dim() = 128;
  model_args_.index_topk() = 64;
  // Keep every layer dense so construction only exercises attention/indexer.
  model_args_.first_k_dense_replace() = kNumLayers;
}
void dec_use_cpu_constructor_context(
    ModelArgs& model_args_,
    QuantArgs& quant_args_,
    ParallelArgs& parallel_args_,
    torch::TensorOptions& options_,
    std::unique_ptr<xllm::ProcessGroup>& mock_process_group_,
    ModelContext& context_) {
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch::kCPU)
                 .requires_grad(false);
  mock_process_group_ =
      std::make_unique<test::MockProcessGroup>(torch::Device(torch::kCPU));
  parallel_args_ = ParallelArgs(/*rank=*/0,
                                /*world_size=*/1,
                                /*dp_size=*/1,
                                mock_process_group_.get());
  parallel_args_.tp_group_ = mock_process_group_.get();
  parallel_args_.single_rank_group_ = mock_process_group_.get();
  parallel_args_.cp_group_ = mock_process_group_.get();
  dec_refresh_ctx(model_args_, quant_args_, parallel_args_, options_, context_);
}
}  // namespace

TEST_F(DeepseekV2DecoderLayerTest,
       DefaultConstructorWhenShareConfiguredKeepsOwnIndexer) {
  dec_configure_glm5_indexer(model_args_);
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  dec_use_cpu_constructor_context(model_args_,
                                  quant_args_,
                                  parallel_args_,
                                  options_,
                                  mock_process_group_,
                                  context_);

  DecoderHolder decoder(DeepseekV32DecoderLayerImpl(context_, /*layer_id=*/3));

  const auto parameters = decoder->named_parameters(/*recurse=*/true);
  EXPECT_TRUE(parameters.contains("self_attn.indexer.wq_b.weight"));
  EXPECT_TRUE(parameters.contains("self_attn.indexer.wk.weight"));
  EXPECT_TRUE(parameters.contains("self_attn.indexer.weights_proj.weight"));
  EXPECT_TRUE(parameters.contains("self_attn.indexer.k_norm.weight"));
}

TEST_F(DeepseekV2DecoderLayerTest,
       FreqConfig_WhenGlm52_ThenPlanMatchesIndexerTypes) {
  dec_configure_glm5_indexer(model_args_);
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_topk_pattern() = "";
  dec_refresh_ctx(model_args_, quant_args_, parallel_args_, options_, context_);

  for (int32_t layer_id : {0, 2, 3, 5, 6, 7, kNumLayers - 1}) {
    auto decoder = dec_make_decoder(model_args_, context_, layer_id);
    EXPECT_EQ(decoder->named_modules().contains("self_attn.indexer"),
              is_full_indexer_layer_glm52(layer_id))
        << "layer " << layer_id;
  }
}

TEST_F(DeepseekV2DecoderLayerTest,
       SharedLayerOwnsNoIndexerParametersOrConstants) {
  dec_configure_glm5_indexer(model_args_);
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  dec_refresh_ctx(model_args_, quant_args_, parallel_args_, options_, context_);

  auto full_decoder = dec_make_decoder(model_args_, context_, /*layer_id=*/2);
  auto shared_decoder = dec_make_decoder(model_args_, context_, /*layer_id=*/3);

  EXPECT_TRUE(full_decoder->named_modules().contains("self_attn.indexer"));
  EXPECT_TRUE(
      full_decoder->named_modules().contains("self_attn.indexer_rotary_emb"));
  EXPECT_FALSE(shared_decoder->named_modules().contains("self_attn.indexer"));
  EXPECT_FALSE(
      shared_decoder->named_modules().contains("self_attn.indexer_rotary_emb"));

  const auto full_parameters = full_decoder->named_parameters(/*recurse=*/true);
  const auto shared_parameters =
      shared_decoder->named_parameters(/*recurse=*/true);
  EXPECT_TRUE(full_parameters.contains("self_attn.indexer.wq_b.weight"));
  EXPECT_TRUE(full_parameters.contains("self_attn.indexer.wk.weight"));
  EXPECT_TRUE(
      full_parameters.contains("self_attn.indexer.weights_proj.weight"));
  EXPECT_TRUE(full_parameters.contains("self_attn.indexer.k_norm.weight"));
  EXPECT_FALSE(shared_parameters.contains("self_attn.indexer.wq_b.weight"));
  EXPECT_FALSE(shared_parameters.contains("self_attn.indexer.wk.weight"));
  EXPECT_FALSE(
      shared_parameters.contains("self_attn.indexer.weights_proj.weight"));
  EXPECT_FALSE(shared_parameters.contains("self_attn.indexer.k_norm.weight"));
}

TEST_F(DeepseekV2DecoderLayerTest,
       Pattern_WhenEquivalentFsString_ThenPlanMatchesFreqConfig) {
  dec_configure_glm5_indexer(model_args_);
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = build_fs_pattern_glm52(kNumLayers);
  dec_refresh_ctx(model_args_, quant_args_, parallel_args_, options_, context_);

  for (int32_t layer_id : {0, 2, 3, 5, 6, 7, kNumLayers - 1}) {
    auto decoder = dec_make_decoder(model_args_, context_, layer_id);
    const bool expect_full = is_full_indexer_layer_glm52(layer_id);
    EXPECT_EQ(decoder->named_modules().contains("self_attn.indexer"),
              expect_full)
        << "layer " << layer_id;
  }
}

TEST_F(DeepseekV2DecoderLayerTest,
       DefaultIndexer_WhenGlm51_ThenEveryLayerLoadsOwnIndexerWeights) {
  dec_configure_glm5_indexer(model_args_);
  // GLM5.1 has an indexer on every layer but no cross-layer share fields.
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = "";
  dec_refresh_ctx(model_args_, quant_args_, parallel_args_, options_, context_);

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_EQ(topk_share_plan.num_indexer_layers(), kNumLayers);

  for (int32_t layer_id : {0, 3, 6, 7}) {
    auto decoder =
        dec_make_loaded_decoder(model_args_, options_, context_, layer_id);
    EXPECT_TRUE(decoder->named_modules().contains("self_attn.indexer"))
        << "layer " << layer_id;
    EXPECT_TRUE(
        decoder->named_modules().contains("self_attn.indexer_rotary_emb"))
        << "layer " << layer_id;

    const auto parameters = decoder->named_parameters(/*recurse=*/true);
    EXPECT_TRUE(parameters.contains("self_attn.indexer.wq_b.weight"))
        << "layer " << layer_id;
    EXPECT_TRUE(parameters.contains("self_attn.indexer.wk.weight"))
        << "layer " << layer_id;
    EXPECT_TRUE(parameters.contains("self_attn.indexer.weights_proj.weight"))
        << "layer " << layer_id;
    EXPECT_TRUE(parameters.contains("self_attn.indexer.k_norm.weight"))
        << "layer " << layer_id;
  }
}

TEST_F(DeepseekV2DecoderLayerTest,
       MtpForwardComputesThenReusesTopkAcrossSteps) {
  dec_configure_glm5_indexer(model_args_);
  model_args_.model_type() = "glm_moe_dsa_mtp";
  model_args_.mtp_mlp_type() = "dense";
  model_args_.index_n_heads() = 32;
  model_args_.kv_lora_rank() = 256;
  model_args_.v_head_dim() = 128;
  model_args_.first_k_dense_replace() = kNumLayers + 1;
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_share_for_mtp_iteration() = true;
  dec_refresh_ctx(model_args_, quant_args_, parallel_args_, options_, context_);

  DecoderHolder decoder = dec_make_loaded_decoder(
      model_args_, options_, context_, /*layer_id=*/kNumLayers);
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kSeqLen = 1;
  constexpr int64_t kBlockSize = 16;
  KVCacheConfig::get_instance().block_size(kBlockSize);
  ModelInputParams input_params =
      dec_build_prefill_params(options_, kBatchSize, kSeqLen);
  input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  AttentionMetadata attn_metadata =
      AttentionMetadataBuilder::build(input_params, /*enable_mla=*/true);
  const torch::Tensor initial_hidden =
      test::seeded_tensor("deepseek_v2_decoder.mtp_topk_hidden",
                          {kBatchSize, model_args_.hidden_size()},
                          torch::kBFloat16,
                          options_.device());
  torch::Tensor positions = torch::zeros(
      {kBatchSize},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  KVCache first_cache = dec_build_cache(model_args_,
                                        options_,
                                        /*block_num=*/kBatchSize + 1,
                                        /*block_size=*/kBlockSize);
  torch::Tensor first_hidden = initial_hidden.clone();
  std::optional<torch::Tensor> first_residual = std::nullopt;
  std::optional<DsaTopkState> first_topk_output = std::nullopt;
  torch::Tensor first_output = decoder->forward_mtp(first_hidden,
                                                    first_residual,
                                                    positions,
                                                    attn_metadata,
                                                    first_cache,
                                                    input_params,
                                                    /*input_ids=*/std::nullopt,
                                                    /*topk_input=*/std::nullopt,
                                                    first_topk_output);
  dec_sync_dev(options_);

  ASSERT_TRUE(first_topk_output.has_value());
  const DsaTopkState& first_topk = first_topk_output.value();
  EXPECT_EQ(first_topk.block_tables().sizes(),
            torch::IntArrayRef({kBatchSize, model_args_.index_topk()}));
  EXPECT_EQ(first_topk.context_lens().sizes(),
            torch::IntArrayRef({kBatchSize}));
  EXPECT_TRUE(first_topk.block_tables().is_contiguous());
  EXPECT_TRUE(first_topk.context_lens().is_contiguous());

  KVCache second_cache = dec_build_cache(model_args_,
                                         options_,
                                         /*block_num=*/kBatchSize + 1,
                                         /*block_size=*/kBlockSize);
  torch::Tensor second_hidden = initial_hidden.clone();
  std::optional<torch::Tensor> second_residual = std::nullopt;
  std::optional<DsaTopkState> second_topk_output = std::nullopt;
  torch::Tensor second_output = decoder->forward_mtp(second_hidden,
                                                     second_residual,
                                                     positions,
                                                     attn_metadata,
                                                     second_cache,
                                                     input_params,
                                                     /*input_ids=*/std::nullopt,
                                                     first_topk_output,
                                                     second_topk_output);
  dec_sync_dev(options_);

  ASSERT_TRUE(second_topk_output.has_value());
  const DsaTopkState& second_topk = second_topk_output.value();
  EXPECT_EQ(second_topk.block_tables().data_ptr(),
            first_topk.block_tables().data_ptr());
  EXPECT_EQ(second_topk.context_lens().data_ptr(),
            first_topk.context_lens().data_ptr());
  test::verify_tensor_close(
      second_output, first_output, /*rtol=*/1e-3, /*atol=*/1e-2);
}

TEST_F(DeepseekV2DecoderLayerTest, PatternWithOnlyFullLayersHasNoReuse) {
  dec_configure_glm5_indexer(model_args_);
  model_args_.n_layers() = 4;
  model_args_.index_topk_freq() = 1;
  model_args_.index_skip_topk_offset() = 0;
  model_args_.index_topk_pattern() = "FFFF";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_FALSE(topk_share_plan.decision_for(/*layer_id=*/3).reuse_topk);
  EXPECT_FALSE(topk_share_plan.decision_for(/*layer_id=*/3).output_topk);
}

TEST_F(DeepseekV2DecoderLayerTest,
       ShortFrequencyPlanWithNoSharedLayerHasNoReuse) {
  dec_configure_glm5_indexer(model_args_);
  model_args_.n_layers() = 3;
  model_args_.index_topk_freq() = 4;
  model_args_.index_skip_topk_offset() = 3;
  model_args_.index_topk_pattern() = "";

  const DsaTopkSharePlan topk_share_plan(model_args_);
  EXPECT_FALSE(topk_share_plan.has_reuse());
  EXPECT_FALSE(topk_share_plan.decision_for(/*layer_id=*/2).output_topk);
}

}  // namespace layer
}  // namespace xllm
