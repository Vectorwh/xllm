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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v32/deepseek_v32_attention.h"
#include "layers/mlu/tests_utils.h"
#include "platform/platform.h"

namespace xllm::layer {
namespace {

ModelArgs create_prefill_model_args(const std::string& rope_type) {
  ModelArgs args;
  args.hidden_size() = 2048;
  args.n_heads() = 4;
  args.q_lora_rank() = 1536;
  args.kv_lora_rank() = 512;
  args.qk_nope_head_dim() = 128;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 128;
  args.rms_norm_eps() = 1e-6;
  args.max_position_embeddings() = 4096;
  args.rope_theta() = 10000;
  args.rope_scaling_rope_type() = rope_type;
  args.rope_scaling_original_max_position_embeddings() = 4096;
  args.rope_scaling_factor() = 40;
  args.rope_extrapolation_factor() = 1;
  args.rope_scaling_attn_factor() = 1;
  args.rope_scaling_beta_fast() = 32;
  args.rope_scaling_beta_slow() = 1;
  args.rope_scaling_mscale() = 1;
  args.rope_scaling_mscale_all_dim() = 1;
  args.enable_mla() = true;
  args.index_n_heads() = 0;
  return args;
}

StateDict create_prefill_state_dict(const torch::TensorOptions& options) {
  const torch::Device device = options.device();
  std::unordered_map<std::string, torch::Tensor> weights;
  auto add_weight = [&](const std::string& name, torch::IntArrayRef shape) {
    weights.emplace(
        name,
        test::seeded_tensor(name, shape, torch::kBFloat16, device) * 0.02f);
  };
  add_weight("q_a_proj.weight", {1536, 2048});
  add_weight("q_b_proj.weight", {4 * 192, 1536});
  add_weight("kv_a_proj_with_mqa.weight", {576, 2048});
  add_weight("kv_b_proj.weight", {4 * 256, 512});
  add_weight("o_proj.weight", {2048, 4 * 128});
  weights.emplace("q_a_layernorm.weight", torch::ones({1536}, options));
  weights.emplace("kv_a_layernorm.weight", torch::ones({512}, options));
  return StateDict(std::move(weights), "");
}

class UnreducedAttentionProcessGroup final : public test::MockProcessGroup {
 public:
  UnreducedAttentionProcessGroup(const torch::Device& device, int64_t rank)
      : test::MockProcessGroup(device, rank, /*world_size=*/2) {}

  void allreduce(torch::Tensor& /*input*/,
                 c10d::ReduceOp /*op*/ = c10d::ReduceOp::SUM) override {
    ADD_FAILURE() << "Attention must leave TP_SHARD reduction to the caller";
  }
};

TEST(MluMlaPrefillDispatchTest,
     PreservesLatentCacheAndMatchesAbsorbedAttention) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto int_options = options.dtype(torch::kInt32);
  for (const std::string rope_type : {"default", "deepseek_yarn"}) {
    auto args = create_prefill_model_args(rope_type);
    std::unique_ptr<ProcessGroup> process_group;
    auto parallel_args = test::create_default_parallel_args(process_group);
    OptimizationConfig optimization;
    auto attention = DeepseekV32Attention(
        args, QuantArgs(), parallel_args, options, optimization);

    auto state_dict = create_prefill_state_dict(options);
    attention->load_state_dict(state_dict);
    attention->verify_loaded_weights();

    for (int64_t tokens : {int64_t{512}, int64_t{1024}}) {
      auto hidden = test::seeded_tensor(
          "mla_dispatch/hidden", {tokens, 2048}, torch::kBFloat16, device);
      auto positions = torch::arange(tokens, int_options);
      auto reference_cache_tensor =
          torch::zeros({tokens / 16, 1, 16, 576}, options);
      auto actual_cache_tensor = torch::zeros_like(reference_cache_tensor);
      KVCache reference_cache(
          KVCacheTensors{reference_cache_tensor, torch::Tensor()});
      KVCache actual_cache(
          KVCacheTensors{actual_cache_tensor, torch::Tensor()});
      AttentionMetadata metadata;
      metadata.is_prefill = true;
      metadata.is_chunked_prefill = false;
      metadata.is_dummy = false;
      metadata.max_query_len = tokens;
      metadata.max_seq_len = tokens;
      metadata.total_kv_len = tokens;
      metadata.q_cu_seq_lens = torch::tensor({int64_t{0}, tokens}, int_options);
      metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
      metadata.kv_seq_lens = torch::tensor({tokens}, int_options);
      metadata.slot_mapping = torch::arange(tokens, int_options);
      // The graph-compatible path remains absorbed and supplies the reference.
      metadata.enable_cuda_graph = true;
      auto reference =
          attention->forward(positions, hidden, metadata, reference_cache);
      metadata.enable_cuda_graph = false;
      auto actual =
          attention->forward(positions, hidden, metadata, actual_cache);
      EXPECT_TRUE(torch::equal(actual_cache_tensor, reference_cache_tensor));
      EXPECT_TRUE(torch::allclose(actual.output, reference.output, 0.03, 0.02))
          << "rope=" << rope_type << " tokens=" << tokens;
      auto error = actual.output.to(torch::kFloat32) -
                   reference.output.to(torch::kFloat32);
      const float relative_error =
          (error.norm() /
           reference.output.to(torch::kFloat32).norm().clamp_min(1e-6))
              .item<float>();
      EXPECT_LT(relative_error, 0.015f)
          << "rope=" << rope_type << " tokens=" << tokens;
    }

    // Compare cached suffixes with a complete causal forward. Unequal prefix
    // lengths, partial blocks and reversed physical pages exercise gathering
    // and bottom-right masking independently of the absorbed cached path.
    for (int64_t prefix_offset : {int64_t{0}, int64_t{15}}) {
      constexpr int64_t kSequenceTokens = 2048;
      constexpr int64_t kTotalTokens = 2 * kSequenceTokens;
      std::vector<int32_t> slots;
      std::vector<int64_t> suffix_indices;
      std::vector<int32_t> suffix_lengths;
      slots.reserve(kTotalTokens);
      suffix_indices.reserve(kTotalTokens);
      suffix_lengths.reserve(2);
      for (int64_t sequence = 0; sequence < 2; ++sequence) {
        const int64_t prefix = (sequence == 0 ? 1024 : 512) - prefix_offset;
        suffix_lengths.emplace_back(kSequenceTokens - prefix);
        for (int64_t token = 0; token < kSequenceTokens; ++token) {
          const int64_t logical = sequence * kSequenceTokens + token;
          slots.emplace_back(static_cast<int32_t>(
              (kTotalTokens / 16 - 1 - logical / 16) * 16 + token % 16));
        }
        for (int64_t token = prefix; token < kSequenceTokens; ++token) {
          suffix_indices.emplace_back(sequence * kSequenceTokens + token);
        }
      }
      auto full_hidden = test::seeded_tensor(
          "mla_cached/hidden", {kTotalTokens, 2048}, torch::kBFloat16, device);
      auto full_positions =
          torch::arange(kSequenceTokens, int_options).repeat({2});
      auto reference_tensor =
          torch::zeros({kTotalTokens / 16, 1, 16, 576}, options);
      KVCache reference_cache(
          KVCacheTensors{reference_tensor, torch::Tensor()});
      AttentionMetadata metadata;
      metadata.is_prefill = true;
      metadata.is_chunked_prefill = false;
      metadata.is_dummy = false;
      metadata.enable_cuda_graph = true;
      metadata.max_query_len = kSequenceTokens;
      metadata.max_seq_len = kSequenceTokens;
      metadata.total_kv_len = kTotalTokens;
      metadata.q_cu_seq_lens = torch::tensor({0, 2048, 4096}, int_options);
      metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
      metadata.kv_seq_lens = torch::tensor({2048, 2048}, int_options);
      metadata.slot_mapping = torch::tensor(slots, int_options);
      metadata.block_table =
          torch::arange(kTotalTokens / 16, int_options).flip({0}).view({2, -1});
      auto reference = attention->forward(
          full_positions, full_hidden, metadata, reference_cache);

      auto suffix_index =
          torch::tensor(suffix_indices, options.dtype(torch::kInt64));
      auto suffix_slots = metadata.slot_mapping.index_select(0, suffix_index);
      auto actual_tensor = reference_tensor.clone();
      actual_tensor.view({kTotalTokens, 576})
          .index_fill_(0, suffix_slots.to(torch::kInt64), 0);
      KVCache actual_cache(KVCacheTensors{actual_tensor, torch::Tensor()});
      metadata.is_prefill = false;
      metadata.is_chunked_prefill = true;
      metadata.enable_cuda_graph = false;
      metadata.max_query_len = suffix_lengths[1];
      metadata.q_cu_seq_lens = torch::tensor(
          {0, suffix_lengths[0], suffix_lengths[0] + suffix_lengths[1]},
          int_options);
      metadata.slot_mapping = suffix_slots;
      auto actual =
          attention->forward(full_positions.index_select(0, suffix_index),
                             full_hidden.index_select(0, suffix_index),
                             metadata,
                             actual_cache);
      auto expected = reference.output.index_select(0, suffix_index);
      EXPECT_TRUE(torch::equal(actual_tensor, reference_tensor));
      EXPECT_TRUE(torch::allclose(actual.output, expected, 0.03, 0.02))
          << "rope=" << rope_type << " prefix_offset=" << prefix_offset;
      const float relative_error =
          ((actual.output.to(torch::kFloat32) - expected.to(torch::kFloat32))
               .norm() /
           expected.to(torch::kFloat32).norm().clamp_min(1e-6))
              .item<float>();
      EXPECT_LT(relative_error, 0.015f)
          << "rope=" << rope_type << " prefix_offset=" << prefix_offset;
    }
  }
}

TEST(MluMlaPrefillTpTest, MatchesSingleRankAbsorbedAttention) {
  const torch::Device device(Platform::type_torch(), 0);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const auto int_options = options.dtype(torch::kInt32);
  auto args = create_prefill_model_args("default");
  auto state_dict = create_prefill_state_dict(options);
  std::unique_ptr<ProcessGroup> reference_group;
  auto reference_parallel = test::create_default_parallel_args(reference_group);
  OptimizationConfig optimization;
  auto reference_attention = DeepseekV32Attention(
      args, QuantArgs(), reference_parallel, options, optimization);
  reference_attention->load_state_dict(state_dict);
  reference_attention->verify_loaded_weights();

  constexpr int64_t kTokens = 1024;
  auto hidden = test::seeded_tensor(
      "mla_tp/hidden", {kTokens, args.hidden_size()}, torch::kBFloat16, device);
  auto positions = torch::arange(kTokens, int_options);
  auto reference_tensor = torch::zeros({kTokens / 16, 1, 16, 576}, options);
  KVCache reference_cache(KVCacheTensors{reference_tensor, torch::Tensor()});
  AttentionMetadata metadata;
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  metadata.max_query_len = kTokens;
  metadata.max_seq_len = kTokens;
  metadata.total_kv_len = kTokens;
  metadata.q_cu_seq_lens = torch::tensor({int64_t{0}, kTokens}, int_options);
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.kv_seq_lens = torch::tensor({kTokens}, int_options);
  metadata.slot_mapping = torch::arange(kTokens, int_options);
  // Use absorbed attention for an independent single-rank reference.
  metadata.enable_cuda_graph = true;
  auto reference = reference_attention->forward(
      positions, hidden, metadata, reference_cache);
  metadata.enable_cuda_graph = false;

  auto combined =
      torch::zeros_like(reference.output, options.dtype(torch::kFloat32));
  // Run both logical ranks on one device. Loading the same full checkpoint
  // exercises the actual column/row-parallel weight slicing for each rank.
  for (int32_t rank = 0; rank < 2; ++rank) {
    SCOPED_TRACE(rank);
    UnreducedAttentionProcessGroup process_group(device, rank);
    ParallelArgs parallel_args(rank, /*world_size=*/2, &process_group);
    parallel_args.tp_size() = 2;
    parallel_args.tp_group_ = &process_group;
    parallel_args.single_rank_group_ = reference_group.get();
    parallel_args.cp_group_ = reference_group.get();
    auto attention = DeepseekV32Attention(
        args, QuantArgs(), parallel_args, options, optimization);
    attention->load_state_dict(state_dict);
    attention->verify_loaded_weights();
    auto cache_tensor = torch::zeros_like(reference_tensor);
    KVCache cache(KVCacheTensors{cache_tensor, torch::Tensor()});
    auto actual = attention->forward(positions, hidden, metadata, cache);
    EXPECT_EQ(actual.layout, PostAttnLayout::TP_SHARD);
    ASSERT_EQ(actual.output.sizes(), reference.output.sizes());
    EXPECT_TRUE(torch::equal(cache_tensor, reference_tensor));
    // o_proj already maps local heads to the full hidden dimension, so TP
    // outputs are partial sums rather than tensors to concatenate by head.
    combined.add_(actual.output.to(torch::kFloat32));
  }
  auto expected = reference.output.to(torch::kFloat32);
  EXPECT_TRUE(torch::allclose(combined, expected, 0.03, 0.02));
  const float relative_error =
      ((combined - expected).norm() / expected.norm().clamp_min(1e-6))
          .item<float>();
  EXPECT_LT(relative_error, 0.015f);
}

}  // namespace
}  // namespace xllm::layer
