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

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

#include "core/framework/config/kv_cache_config.h"
#include "core/kernels/mlu/dcp/dcp_precision_test_utils.h"
#include "core/layers/mlu/deepseek_v32/dcp_batch_metadata.h"
#include "core/util/linalg.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/kv_shard_batch_metadata.h"
#include "layers/mlu/attention.h"
#include "layers/mlu/deepseek_v32/dcp_decode_context.h"
#include "layers/mlu/deepseek_v32/deepseek_v32_indexer.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "triton_jit/include/spec.h"

namespace xllm {
namespace layer {
namespace {
using Weights = std::unordered_map<std::string, torch::Tensor>;

Weights create_random_indexer_weights(int64_t dim,
                                      int64_t index_n_heads,
                                      int64_t index_head_dim,
                                      int64_t q_lora_rank,
                                      const torch::TensorOptions& options);
void populate_indexer_metadata(AttentionMetadata& metadata,
                               int64_t batch_size,
                               int64_t max_query_len,
                               int64_t max_seq_len,
                               bool is_prefill,
                               int64_t max_num_blocks_per_seq,
                               const torch::TensorOptions& int_options);
void populate_chunked_metadata(AttentionMetadata& metadata,
                               int64_t batch_size,
                               int64_t history_len,
                               int64_t current_len,
                               int64_t block_size,
                               bool use_noncontiguous_blocks,
                               const torch::TensorOptions& int_options);

Weights create_random_indexer_weights(int64_t dim,
                                      int64_t index_n_heads,
                                      int64_t index_head_dim,
                                      int64_t q_lora_rank,
                                      const torch::TensorOptions& options) {
  auto create_random_tensor = [&](const std::vector<int64_t>& shape,
                                  float min_val,
                                  float max_val,
                                  std::optional<torch::ScalarType> dtype =
                                      std::nullopt) {
    auto opts = dtype.has_value() ? options.dtype(dtype.value()) : options;
    return test::seeded_tensor(
               "indexer.input", shape, opts.dtype().toScalarType(), torch::kCPU)
                   .to(opts.device()) *
               (max_val - min_val) +
           min_val;
  };

  std::unordered_map<std::string, torch::Tensor> weight_dict;
  weight_dict["wq_b.weight"] = create_random_tensor(
      {index_n_heads * index_head_dim, q_lora_rank}, -0.1f, 0.1f);
  weight_dict["wk.weight"] =
      create_random_tensor({index_head_dim, dim}, -0.1f, 0.1f);
  weight_dict["weights_proj.weight"] =
      create_random_tensor({index_n_heads, dim}, -0.1f, 0.1f);
  weight_dict["k_norm.weight"] =
      create_random_tensor({index_head_dim}, -0.5f, 0.5f, torch::kFloat32);
  weight_dict["k_norm.bias"] =
      create_random_tensor({index_head_dim}, -0.5f, 0.5f, torch::kFloat32);

  return weight_dict;
}

void populate_indexer_metadata(AttentionMetadata& metadata,
                               int64_t batch_size,
                               int64_t max_query_len,
                               int64_t max_seq_len,
                               bool is_prefill,
                               int64_t max_num_blocks_per_seq,
                               const torch::TensorOptions& int_options) {
  // q_cu_seq_lens
  metadata.q_cu_seq_lens = torch::arange(
      0, (batch_size + 1) * max_query_len, max_query_len, int_options);

  // kv_cu_seq_lens
  metadata.kv_cu_seq_lens = torch::arange(
      0, (batch_size + 1) * max_query_len, max_query_len, int_options);

  metadata.kv_seq_lens = torch::full({batch_size}, max_query_len, int_options);

  const int64_t block_size = KVCacheConfig::get_instance().block_size();
  const int64_t blocks_per_request =
      (max_query_len + block_size - 1) / block_size;
  std::vector<int32_t> pages, slots;
  pages.reserve(batch_size * blocks_per_request);
  slots.reserve(batch_size * max_query_len);
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    const int64_t first = 1 + batch * blocks_per_request;
    for (int64_t page = 0; page < blocks_per_request; ++page) {
      pages.emplace_back(first + page);
    }
    for (int64_t token = 0; token < max_query_len; ++token) {
      slots.emplace_back(first * block_size + token);
    }
  }
  metadata.block_table = torch::tensor(pages, int_options)
                             .reshape({batch_size, blocks_per_request});
  metadata.slot_mapping = torch::tensor(slots, int_options);
  metadata.max_query_len = max_query_len;
  metadata.max_seq_len = max_seq_len;
  metadata.total_kv_len = batch_size * max_query_len;
  metadata.compute_dtype = "bfloat16";
  metadata.is_prefill = is_prefill;
  metadata.is_chunked_prefill = false;
}

void populate_chunked_metadata(AttentionMetadata& metadata,
                               int64_t batch_size,
                               int64_t history_len,
                               int64_t current_len,
                               int64_t block_size,
                               bool use_noncontiguous_blocks,
                               const torch::TensorOptions& int_options) {
  int64_t total_len = history_len + current_len;
  int64_t blocks_per_seq = (total_len + block_size - 1) / block_size;

  metadata.q_cu_seq_lens = torch::arange(
      0, (batch_size + 1) * current_len, current_len, int_options);

  metadata.kv_cu_seq_lens =
      torch::arange(0, (batch_size + 1) * total_len, total_len, int_options);

  metadata.kv_seq_lens = torch::full({batch_size}, total_len, int_options);

  metadata.block_table =
      torch::zeros({batch_size, blocks_per_seq}, int_options);

  std::vector<int32_t> slot_mapping;
  slot_mapping.reserve(batch_size * current_len);

  for (int64_t b = 0; b < batch_size; ++b) {
    std::vector<int32_t> block_ids;
    block_ids.reserve(blocks_per_seq);
    for (int64_t logical_block = 0; logical_block < blocks_per_seq;
         ++logical_block) {
      int64_t contiguous_block = b * blocks_per_seq + logical_block;
      int64_t physical_block = use_noncontiguous_blocks
                                   ? contiguous_block * 2 + 1
                                   : contiguous_block;
      block_ids.emplace_back(static_cast<int32_t>(physical_block));
    }
    metadata.block_table[b].copy_(torch::tensor(block_ids, int_options));

    for (int64_t position = history_len; position < total_len; ++position) {
      int64_t logical_block = position / block_size;
      int64_t block_offset = position % block_size;
      int64_t slot =
          static_cast<int64_t>(block_ids[logical_block]) * block_size +
          block_offset;
      slot_mapping.emplace_back(static_cast<int32_t>(slot));
    }
  }
  metadata.slot_mapping = torch::tensor(slot_mapping, int_options);

  metadata.max_query_len = current_len;
  metadata.max_seq_len = total_len;
  metadata.total_kv_len = batch_size * total_len;
  metadata.compute_dtype = "bfloat16";
  // Chunked prefill uses absolute positions with discrete RoPE.
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = true;
}

}  // namespace

namespace {
struct TestConfig {
  int64_t dim = 7168;
  int64_t index_n_heads = 64;
  int64_t index_head_dim = 128;
  int64_t qk_rope_head_dim = 64;
  int64_t index_topk = 2048;
  int64_t q_lora_rank = 1536;
  int64_t max_position_embeddings = 8192;
  int64_t rope_theta = 10000;
  bool rope_interleaved = true;
  int64_t head_kv = 1;
  int64_t block_size = 1;
  int64_t block_num = 10240;
};
struct TestInputs {
  torch::Tensor x;
  torch::Tensor q_norm;
  torch::Tensor positions;
  torch::Tensor k_cache;
  torch::Tensor expected_k;
  torch::Tensor initial_cache;
  torch::Tensor initial_scale;
  std::optional<torch::Tensor> k_cache_scale;
  std::unordered_map<std::string, torch::Tensor> weights;
  AttentionMetadata metadata;
};
torch::Tensor idx_create_random_tensor(
    torch::TensorOptions& options_,
    const std::vector<int64_t>& shape,
    float min_val = -1.0f,
    float max_val = 1.0f,
    std::optional<torch::ScalarType> dtype = std::nullopt);
std::unordered_map<std::string, torch::Tensor> idx_create_random_weights(
    torch::TensorOptions& options_,
    int64_t dim,
    int64_t index_n_heads,
    int64_t index_head_dim,
    int64_t q_lora_rank);
void idx_populate_attention_metadata(torch::TensorOptions& int_option_,
                                     AttentionMetadata& metadata,
                                     int64_t batch_size,
                                     int64_t max_query_len,
                                     int64_t max_seq_len,
                                     bool is_prefill,
                                     int64_t max_num_blocks_per_seq);
void idx_populate_chunked_attention_metadata(torch::TensorOptions& int_option_,
                                             AttentionMetadata& metadata,
                                             int64_t batch_size,
                                             int64_t history_len,
                                             int64_t current_len,
                                             int64_t block_size,
                                             bool use_noncontiguous_blocks);
TestInputs idx_create_inputs(TestConfig& test_config_,
                             torch::TensorOptions& options_,
                             torch::TensorOptions& int_option_,
                             std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
                             int64_t batch_size,
                             int64_t max_query_len,
                             bool is_prefill,
                             bool chunked_prefill = false,
                             int64_t history_len = 0,
                             bool use_default_rope = false,
                             bool quantized_cache = false,
                             int64_t cache_block_size = 1,
                             bool use_noncontiguous_blocks = false);
TestInputs idx_create_quantized_inputs(
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    torch::TensorOptions& int_option_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    int64_t batch_size,
    int64_t max_query_len,
    bool is_prefill,
    bool chunked_prefill = false,
    int64_t history_len = 0,
    int64_t cache_block_size = 1,
    bool use_noncontiguous_blocks = false);
void idx_fill_quantized_cache(torch::TensorOptions& options_,
                              TestInputs& inputs);
DeepseekV32Indexer idx_create_indexer(
    ParallelArgs& parallel_args_,
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    TestInputs& inputs,
    bool enable_fused_qk);
void idx_expect_dcp_local_prefill_candidates(
    ParallelArgs& parallel_args_,
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    TestInputs& inputs);
std::tuple<torch::Tensor, torch::Tensor> idx_run_indexer(
    ParallelArgs& parallel_args_,
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    TestInputs& inputs,
    bool is_prefill,
    bool enable_fused_qk);
void idx_expect_select_output(TestConfig& test_config_,
                              const torch::Tensor& block_tables,
                              const torch::Tensor& context_lens,
                              int64_t num_tokens);
void idx_expect_quantized_cache_updated(const TestInputs& inputs);
v32_cp::DeepseekV32CPContext idx_make_single_rank_cp_context(
    torch::TensorOptions& options_,
    torch::TensorOptions& int_option_,
    const TestInputs& inputs,
    int64_t token_num);
torch::Tensor idx_create_random_tensor(torch::TensorOptions& options_,
                                       const std::vector<int64_t>& shape,
                                       float min_val,
                                       float max_val,
                                       std::optional<torch::ScalarType> dtype) {
  auto opts = dtype.has_value() ? options_.dtype(dtype.value()) : options_;
  return test::seeded_tensor(
             "indexer.input", shape, opts.dtype().toScalarType(), torch::kCPU)
                 .to(opts.device()) *
             (max_val - min_val) +
         min_val;
}
std::unordered_map<std::string, torch::Tensor> idx_create_random_weights(
    torch::TensorOptions& options_,
    int64_t dim,
    int64_t index_n_heads,
    int64_t index_head_dim,
    int64_t q_lora_rank) {
  return create_random_indexer_weights(
      dim, index_n_heads, index_head_dim, q_lora_rank, options_);
}
void idx_populate_attention_metadata(torch::TensorOptions& int_option_,
                                     AttentionMetadata& metadata,
                                     int64_t batch_size,
                                     int64_t max_query_len,
                                     int64_t max_seq_len,
                                     bool is_prefill,
                                     int64_t max_num_blocks_per_seq) {
  populate_indexer_metadata(metadata,
                            batch_size,
                            max_query_len,
                            max_seq_len,
                            is_prefill,
                            max_num_blocks_per_seq,
                            int_option_);
}
void idx_populate_chunked_attention_metadata(torch::TensorOptions& int_option_,
                                             AttentionMetadata& metadata,
                                             int64_t batch_size,
                                             int64_t history_len,
                                             int64_t current_len,
                                             int64_t block_size,
                                             bool use_noncontiguous_blocks) {
  populate_chunked_metadata(metadata,
                            batch_size,
                            history_len,
                            current_len,
                            block_size,
                            use_noncontiguous_blocks,
                            int_option_);
}
TestInputs idx_create_inputs(TestConfig& test_config_,
                             torch::TensorOptions& options_,
                             torch::TensorOptions& int_option_,
                             std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
                             int64_t batch_size,
                             int64_t max_query_len,
                             bool is_prefill,
                             bool chunked_prefill,
                             int64_t history_len,
                             bool use_default_rope,
                             bool quantized_cache,
                             int64_t cache_block_size,
                             bool use_noncontiguous_blocks) {
  // Preserve per-test dimensions; each fixture starts with default config.
  test_config_.block_size = cache_block_size;
  KVCacheConfig::get_instance().block_size(cache_block_size);
  if (use_default_rope) {
    rotary_emb_ = std::make_shared<RotaryEmbeddingImpl>(
        test_config_.qk_rope_head_dim,
        test_config_.max_position_embeddings,
        test_config_.rope_theta,
        test_config_.rope_interleaved,
        options_);
  } else {
    rotary_emb_ = std::make_shared<DeepseekScalingRotaryEmbeddingImpl>(
        test_config_.qk_rope_head_dim,
        test_config_.qk_rope_head_dim,
        test_config_.max_position_embeddings,
        test_config_.max_position_embeddings,
        test_config_.rope_theta,
        test_config_.rope_interleaved,
        /*scaling_factor=*/2.5,
        /*extrapolation_factor=*/1.,
        /*attn_factor=*/40,
        /*beta_fast=*/32,
        /*beta_slow=*/1,
        /*mscale=*/1.,
        /*mscale_all_dim=*/1.,
        options_);
  }

  TestInputs inputs;
  int64_t num_tokens = batch_size * max_query_len;

  inputs.x = idx_create_random_tensor(
      options_, {num_tokens, test_config_.dim}, -1.0f, 1.0f);
  inputs.q_norm = idx_create_random_tensor(
      options_, {num_tokens, test_config_.q_lora_rank}, -1.0f, 1.0f);

  inputs.positions = torch::arange(num_tokens, torch::kInt32)
                         .remainder(max_query_len)
                         .to(int_option_.device());

  const std::vector<int64_t> cache_shape = {test_config_.block_num,
                                            test_config_.head_kv,
                                            test_config_.block_size,
                                            test_config_.index_head_dim};
  if (quantized_cache) {
    inputs.k_cache = torch::zeros(cache_shape, options_.dtype(torch::kChar));
    inputs.k_cache_scale = torch::full({test_config_.block_num,
                                        test_config_.head_kv,
                                        test_config_.block_size},
                                       0.125f,
                                       torch::kFloat32)
                               .to(options_.device());
  } else {
    inputs.k_cache =
        idx_create_random_tensor(options_, cache_shape, -0.5f, 0.5f);
  }

  inputs.weights = idx_create_random_weights(options_,
                                             test_config_.dim,
                                             test_config_.index_n_heads,
                                             test_config_.index_head_dim,
                                             test_config_.q_lora_rank);

  if (chunked_prefill) {
    idx_populate_chunked_attention_metadata(int_option_,
                                            inputs.metadata,
                                            batch_size,
                                            history_len,
                                            max_query_len,
                                            cache_block_size,
                                            use_noncontiguous_blocks);
  } else {
    idx_populate_attention_metadata(int_option_,
                                    inputs.metadata,
                                    batch_size,
                                    max_query_len,
                                    test_config_.max_position_embeddings,
                                    is_prefill,
                                    num_tokens);
  }
  if (chunked_prefill) {
    inputs.positions.add_(history_len);
  }
  return inputs;
}
TestInputs idx_create_quantized_inputs(
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    torch::TensorOptions& int_option_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    int64_t batch_size,
    int64_t max_query_len,
    bool is_prefill,
    bool chunked_prefill,
    int64_t history_len,
    int64_t cache_block_size,
    bool use_noncontiguous_blocks) {
  return idx_create_inputs(test_config_,
                           options_,
                           int_option_,
                           rotary_emb_,
                           batch_size,
                           max_query_len,
                           is_prefill,
                           chunked_prefill,
                           history_len,
                           /*use_default_rope=*/false,
                           /*quantized_cache=*/true,
                           cache_block_size,
                           use_noncontiguous_blocks);
}
void idx_fill_quantized_cache(torch::TensorOptions& options_,
                              TestInputs& inputs) {
  CHECK(inputs.k_cache_scale.has_value());
  const torch::Tensor cache_values = test::seeded_tensor("indexer.resident",
                                                         inputs.k_cache.sizes(),
                                                         torch::kInt32,
                                                         torch::kCPU)
                                         .remainder(128)
                                         .sub(64)
                                         .to(torch::kChar)
                                         .to(options_.device());
  inputs.k_cache.copy_(cache_values);
  inputs.k_cache_scale->copy_(test::seeded_tensor("indexer.scale",
                                                  inputs.k_cache_scale->sizes(),
                                                  torch::kFloat32,
                                                  torch::kCPU)
                                  .mul(0.1f)
                                  .add(0.01f)
                                  .to(options_.device()));
}

// The fused decode op retains projection and LayerNorm precision until RoPE
// writes BF16 values for the Hadamard transform. The eager path also rounds
// after projection and LayerNorm, so it is not a pointwise fused-op oracle.
torch::Tensor idx_reference_fused_k(
    const TestInputs& inputs,
    const std::shared_ptr<RotaryEmbeddingBase>& rotary_emb) {
  const torch::Tensor x = inputs.x.cpu().to(torch::kFloat64);
  const torch::Tensor wk =
      inputs.weights.at("wk.weight").cpu().to(torch::kFloat64);
  torch::Tensor key = torch::matmul(x, wk.t());
  key = key - key.mean(/*dim=*/-1, /*keepdim=*/true);
  key = key *
        torch::rsqrt(key.square().mean(/*dim=*/-1, /*keepdim=*/true) + 1e-6);
  key = key * inputs.weights.at("k_norm.weight").cpu().to(torch::kFloat64) +
        inputs.weights.at("k_norm.bias").cpu().to(torch::kFloat64);

  const torch::Tensor positions = inputs.positions.cpu().to(torch::kInt64);
  const torch::Tensor cos = rotary_emb->get_cos_cache()
                                .cpu()
                                .index_select(0, positions)
                                .to(torch::kFloat64);
  const torch::Tensor sin = rotary_emb->get_sin_cache()
                                .cpu()
                                .index_select(0, positions)
                                .to(torch::kFloat64);
  const int64_t rotary_dim = cos.size(-1);
  const torch::Tensor front = key.slice(/*dim=*/1, /*start=*/0, rotary_dim);
  CHECK(rotary_emb->get_interleaved());
  const torch::Tensor pairs = front.reshape({key.size(0), rotary_dim / 2, 2});
  const torch::Tensor swapped =
      torch::stack({-pairs.select(2, 1), pairs.select(2, 0)}, /*dim=*/2)
          .reshape_as(front);
  front.copy_(front * cos + swapped * sin);
  key = key.to(inputs.x.scalar_type()).to(torch::kFloat64);

  const torch::Tensor hadamard =
      util::create_hadamard_matrix(key.size(-1),
                                   torch::kFloat32,
                                   torch::Device(torch::kCPU),
                                   /*normalize=*/true)
          .to(inputs.x.scalar_type())
          .to(torch::kFloat64);
  return torch::matmul(key, hadamard.t());
}

DeepseekV32Indexer idx_create_indexer(
    ParallelArgs& parallel_args_,
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    TestInputs& inputs,
    bool enable_fused_qk) {
  StateDict state_dict(inputs.weights);
  QuantArgs quant_args;
  DeepseekV32Indexer indexer =
      DeepseekV32Indexer(DeepseekV32IndexerImpl(test_config_.dim,
                                                test_config_.index_n_heads,
                                                test_config_.index_head_dim,
                                                test_config_.qk_rope_head_dim,
                                                test_config_.index_topk,
                                                test_config_.q_lora_rank,
                                                enable_fused_qk,
                                                rotary_emb_,
                                                quant_args,
                                                parallel_args_,
                                                options_));
  indexer->load_state_dict(state_dict);
  if (inputs.k_cache.defined() &&
      inputs.k_cache.scalar_type() == torch::kChar) {
    inputs.initial_cache = inputs.k_cache.cpu().clone();
    inputs.initial_scale = inputs.k_cache_scale->cpu().clone();
    if (inputs.metadata.is_prefill || inputs.metadata.is_chunked_prefill) {
      inputs.expected_k =
          indexer
              ->cp_pre(
                  inputs.x, inputs.q_norm, inputs.positions, inputs.metadata)
              .k_local.cpu()
              .to(torch::kFloat64);
    } else {
      inputs.expected_k = idx_reference_fused_k(inputs, rotary_emb_);
    }
  }
  return indexer;
}
void idx_set_dcp_metadata(TestInputs& inputs, int32_t shards) {
  const int64_t stripe = inputs.k_cache.size(2) * shards;
  const auto q_offsets = inputs.metadata.q_cu_seq_lens.cpu().to(torch::kInt64);
  const auto kv_lengths = inputs.metadata.kv_seq_lens.cpu().to(torch::kInt64);
  const int64_t columns =
      (kv_lengths.max().item<int64_t>() + stripe - 1) / stripe;
  std::vector<int32_t> pages, slots;
  pages.reserve(kv_lengths.numel() * columns);
  slots.reserve(inputs.positions.numel());
  for (int64_t batch = 0; batch < kv_lengths.numel(); ++batch) {
    const int64_t first = 1 + batch * columns;
    for (int64_t page = 0; page < columns; ++page) {
      pages.emplace_back(first + page);
    }
    const int64_t q_length =
        q_offsets[batch + 1].item<int64_t>() - q_offsets[batch].item<int64_t>();
    const int64_t begin = kv_lengths[batch].item<int64_t>() - q_length;
    for (int64_t position = begin; position < begin + q_length; ++position) {
      slots.emplace_back(first * stripe + position);
    }
  }
  inputs.metadata.block_table =
      torch::tensor(pages, inputs.metadata.block_table.options())
          .reshape({kv_lengths.numel(), columns});
  inputs.metadata.slot_mapping =
      torch::tensor(slots, inputs.metadata.slot_mapping.options());
}
void idx_expect_dcp_local_prefill_candidates(
    ParallelArgs& parallel_args_,
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    TestInputs& inputs) {
  parallel_args_.world_size() = 2;
  parallel_args_.kv_split_size() = 2;
  constexpr int32_t kDcpRank = 0;
  const KVShardLayout layout(
      static_cast<int32_t>(KVCacheConfig::get_instance().block_size()),
      /*dcp_size=*/2,
      kDcpRank);
  idx_set_dcp_metadata(inputs, /*shards=*/2);
  const KVShardIndexerMetadata selector_metadata =
      build_kv_shard_indexer_metadata(inputs.metadata);
  AttentionMetadata prefill_metadata = inputs.metadata;
  prefill_metadata.slot_mapping =
      localize_kv_shard_slots(inputs.metadata.slot_mapping, layout);
  DeepseekV32Indexer indexer = idx_create_indexer(parallel_args_,
                                                  test_config_,
                                                  options_,
                                                  rotary_emb_,
                                                  inputs,
                                                  /*enable_fused_qk=*/true);
  const DcpIndexerLocalCandidates candidates =
      indexer->forward_dcp_local_prefill(inputs.x,
                                         inputs.q_norm,
                                         inputs.positions,
                                         inputs.k_cache,
                                         prefill_metadata,
                                         selector_metadata);

  const int64_t token_count = inputs.x.size(0);
  EXPECT_EQ(candidates.scores.sizes(),
            (torch::IntArrayRef{token_count, test_config_.index_topk}));
  EXPECT_EQ(candidates.global_slots.sizes(), candidates.scores.sizes());

  torch::Tensor columns =
      torch::arange(test_config_.index_topk, options_.dtype(torch::kInt32));
  torch::Tensor invalid_columns =
      columns.unsqueeze(0) >= localize_kv_shard_context_lens(
                                  selector_metadata.query_positions + 1, layout)
                                  .unsqueeze(1);
  EXPECT_TRUE(candidates.global_slots.masked_select(invalid_columns)
                  .eq(KVShardLayout::kInvalidSlot)
                  .all()
                  .item<bool>());
  EXPECT_TRUE(candidates.scores.masked_select(invalid_columns)
                  .isneginf()
                  .all()
                  .item<bool>());
  const auto table = inputs.metadata.block_table.cpu().to(torch::kInt64);
  const auto positions = selector_metadata.query_positions.cpu();
  const int64_t block_size = inputs.k_cache.size(2);
  for (int64_t row = 0; row < token_count; ++row) {
    std::vector<int32_t> expected;
    const int64_t visible = positions[row].item<int32_t>() + 1;
    expected.reserve(visible);
    for (int64_t token = 0; token < visible; ++token) {
      const int64_t slot =
          table[0][token / (block_size * 2)].item<int64_t>() * block_size * 2 +
          token % (block_size * 2);
      if (slot / block_size % 2 == kDcpRank) {
        expected.emplace_back(slot);
      }
    }
    const auto actual =
        candidates.global_slots[row].cpu().narrow(0, 0, expected.size());
    EXPECT_TRUE(torch::equal(
        std::get<0>(actual.sort()),
        std::get<0>(torch::tensor(expected, torch::kInt32).sort())));
  }
}
std::tuple<torch::Tensor, torch::Tensor> idx_run_indexer(
    ParallelArgs& parallel_args_,
    TestConfig& test_config_,
    torch::TensorOptions& options_,
    std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
    TestInputs& inputs,
    bool is_prefill,
    bool enable_fused_qk) {
  DeepseekV32Indexer indexer = idx_create_indexer(parallel_args_,
                                                  test_config_,
                                                  options_,
                                                  rotary_emb_,
                                                  inputs,
                                                  enable_fused_qk);
  return indexer->forward(inputs.x,
                          inputs.q_norm,
                          inputs.positions,
                          inputs.k_cache,
                          inputs.metadata,
                          is_prefill,
                          inputs.k_cache_scale);
}
void idx_expect_select_output(TestConfig& test_config_,
                              const torch::Tensor& block_tables,
                              const torch::Tensor& context_lens,
                              int64_t num_tokens) {
  EXPECT_EQ(block_tables.scalar_type(), torch::kInt32);
  EXPECT_EQ(block_tables.sizes(),
            (torch::IntArrayRef{num_tokens, test_config_.index_topk}));
  EXPECT_EQ(context_lens.scalar_type(), torch::kInt32);
  EXPECT_EQ(context_lens.sizes(), (torch::IntArrayRef{num_tokens}));
}
void idx_expect_all_visible(const TestInputs& inputs,
                            const torch::Tensor& tables,
                            const torch::Tensor& lengths) {
  const auto offsets = inputs.metadata.q_cu_seq_lens.cpu().to(torch::kInt64);
  const auto kv_lengths = inputs.metadata.kv_seq_lens.cpu().to(torch::kInt64);
  const auto pages = inputs.metadata.block_table.cpu().to(torch::kInt64);
  const auto actual = tables.cpu();
  const auto lens = lengths.cpu();
  const int64_t block_size = inputs.k_cache.size(2);
  for (int64_t batch = 0; batch < kv_lengths.numel(); ++batch) {
    const int64_t begin = offsets[batch].item<int64_t>();
    const int64_t end = offsets[batch + 1].item<int64_t>();
    for (int64_t row = begin; row < end; ++row) {
      const int64_t visible =
          kv_lengths[batch].item<int64_t>() - (end - begin) + row - begin + 1;
      ASSERT_LE(visible, actual.size(1));
      ASSERT_EQ(lens[row].item<int32_t>(), visible);
      std::vector<int32_t> expected;
      expected.reserve(visible);
      for (int64_t token = 0; token < visible; ++token) {
        expected.emplace_back(pages[batch][token / block_size].item<int64_t>() *
                                  block_size +
                              token % block_size);
      }
      EXPECT_TRUE(torch::equal(
          std::get<0>(actual[row].narrow(0, 0, visible).sort()),
          std::get<0>(torch::tensor(expected, torch::kInt32).sort())));
    }
  }
}
void idx_expect_quantized_cache_updated(const TestInputs& inputs) {
  ASSERT_EQ(inputs.k_cache.scalar_type(), torch::kInt8);
  ASSERT_TRUE(inputs.k_cache_scale.has_value());
  ASSERT_EQ(inputs.k_cache_scale->scalar_type(), torch::kFloat32);
  ASSERT_EQ(inputs.expected_k.scalar_type(), torch::kFloat64);

  ASSERT_TRUE(inputs.k_cache_scale.has_value());
  const auto cache =
      inputs.k_cache.cpu().reshape({-1, inputs.k_cache.size(-1)});
  const auto scales = inputs.k_cache_scale->cpu().flatten();
  const auto slots = inputs.metadata.slot_mapping.cpu().to(torch::kInt64);
  const auto selected_scales = scales.index_select(0, slots);
  ASSERT_TRUE(torch::isfinite(selected_scales).all().item<bool>());
  ASSERT_TRUE(selected_scales.gt(0).all().item<bool>());
  const auto dequantized = cache.index_select(0, slots).to(torch::kFloat32) *
                           selected_scales.unsqueeze(1);
  const auto expected = inputs.expected_k.to(torch::kFloat32);
  ASSERT_EQ(dequantized.sizes(), expected.sizes());
  EXPECT_TRUE((dequantized - expected)
                  .abs()
                  .le(selected_scales.unsqueeze(1) * 1.01 + 1e-6)
                  .all()
                  .item<bool>());
  auto untouched = torch::ones({scales.numel()}, torch::kBool);
  untouched.index_fill_(0, slots, false);
  EXPECT_TRUE(
      torch::equal(cache.index({untouched}),
                   inputs.initial_cache.reshape_as(cache).index({untouched})));
  EXPECT_TRUE(torch::equal(scales.index({untouched}),
                           inputs.initial_scale.flatten().index({untouched})));
}
v32_cp::DeepseekV32CPContext idx_make_single_rank_cp_context(
    torch::TensorOptions& options_,
    torch::TensorOptions& int_option_,
    const TestInputs& inputs,
    int64_t token_num) {
  const int32_t token_num_i32 = static_cast<int32_t>(token_num);
  const int32_t context_len = static_cast<int32_t>(
      inputs.metadata.is_chunked_prefill ? inputs.metadata.total_kv_len
                                         : token_num);
  v32_cp::DeepseekV32CPSegment segment;
  segment.req_idx = 0;
  segment.rank = 0;
  segment.q_tokens = token_num_i32;
  segment.suffix_k_len = token_num_i32;
  segment.ctx_k_len = context_len;
  segment.world_begin = 0;

  torch::Tensor segment_prefix =
      torch::tensor({0, token_num_i32}, int_option_).view({1, 2});
  torch::Tensor context_prefix =
      torch::tensor({0, context_len}, int_option_).view({1, 2});
  v32_cp::DeepseekV32CPContext cp_ctx;
  cp_ctx.local_attn_metadata = inputs.metadata;
  cp_ctx.local_segments = {segment};
  cp_ctx.seg_q_starts_cpu = {0};
  cp_ctx.req_q_offsets_cpu = {0};
  cp_ctx.req_ctx_offsets_cpu = {0};
  cp_ctx.seg_q_cu_lens_2col = segment_prefix;
  cp_ctx.seg_suffix_k_cu_lens_2col = segment_prefix;
  cp_ctx.seg_ctx_k_cu_lens_2col = context_prefix;
  cp_ctx.seg_ctx_lens_1col = torch::tensor({context_len}, int_option_);
  cp_ctx.gathered_reorder_index =
      torch::arange(token_num, options_.dtype(torch::kInt64));
  cp_ctx.gathered_slot_mapping = inputs.metadata.slot_mapping;
  cp_ctx.total_tokens = token_num_i32;
  cp_ctx.rank = 0;
  return cp_ctx;
}
}  // namespace

namespace {
class IndexerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "MLU device is required";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    original_block_size_ = KVCacheConfig::get_instance().block_size();
    device_guard_ = std::make_unique<c10::DeviceGuard>(
        torch::Device(Platform::type_torch(), 0));
  }
  void TearDown() override {
    if (original_block_size_) {
      KVCacheConfig::get_instance().block_size(*original_block_size_);
    }
  }

 private:
  std::optional<int32_t> original_block_size_;
  std::unique_ptr<c10::DeviceGuard> device_guard_;
};
using DcpIndexerScoreTest = IndexerTest;
}  // namespace

TEST_F(IndexerTest, PrefillBatch) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  int64_t batch_size = 2;
  int64_t max_query_len = 33;
  const bool is_prefill = true;
  const bool enable_fused_qk = false;
  int64_t num_tokens = batch_size * max_query_len;
  TestInputs inputs = idx_create_inputs(test_config_,
                                        options_,
                                        int_option_,
                                        rotary_emb_,
                                        batch_size,
                                        max_query_len,
                                        is_prefill);
  auto [new_block_tables, new_context_lens] = idx_run_indexer(parallel_args_,
                                                              test_config_,
                                                              options_,
                                                              rotary_emb_,
                                                              inputs,
                                                              is_prefill,
                                                              enable_fused_qk);

  EXPECT_EQ(new_block_tables.sizes().size(), 2)
      << "new_block_tables should be 2D tensor";
  EXPECT_EQ(new_context_lens.sizes().size(), 1)
      << "new_context_lens should be 1D tensor";
  EXPECT_EQ(new_block_tables.size(0), num_tokens) << "Batch size should match";
  EXPECT_EQ(new_block_tables.size(1), test_config_.index_topk)
      << "Top-k should match";

  idx_expect_all_visible(inputs, new_block_tables, new_context_lens);
}

TEST_F(IndexerTest, ChunkedPrefillBatch) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  const int64_t batch_size = 2;
  const int64_t history_len = 128;
  const int64_t current_len = 64;
  int64_t num_new_tokens = batch_size * current_len;
  const bool is_prefill = true;
  const bool is_chunked = true;
  const bool enable_fused_qk = false;
  TestInputs inputs = idx_create_inputs(test_config_,
                                        options_,
                                        int_option_,
                                        rotary_emb_,
                                        batch_size,
                                        current_len,
                                        is_prefill,
                                        is_chunked,
                                        history_len);
  auto [new_block_tables, new_context_lens] = idx_run_indexer(parallel_args_,
                                                              test_config_,
                                                              options_,
                                                              rotary_emb_,
                                                              inputs,
                                                              is_prefill,
                                                              enable_fused_qk);

  // Validations
  // Shape Verification
  EXPECT_EQ(new_block_tables.dim(), 2);
  EXPECT_EQ(new_block_tables.size(0), num_new_tokens);  // [batch * current_len]
  EXPECT_EQ(new_block_tables.size(1), test_config_.index_topk);

  idx_expect_all_visible(inputs, new_block_tables, new_context_lens);
}

TEST_F(IndexerTest, DcpLocalCausalPrefillSelectsRankLocalCandidates) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  test_config_.index_n_heads = 32;
  TestInputs inputs = idx_create_inputs(test_config_,
                                        options_,
                                        int_option_,
                                        rotary_emb_,
                                        /*batch_size=*/1,
                                        /*max_query_len=*/24,
                                        /*is_prefill=*/true,
                                        /*chunked_prefill=*/false,
                                        /*history_len=*/0,
                                        /*use_default_rope=*/false,
                                        /*quantized_cache=*/false,
                                        /*cache_block_size=*/16);
  idx_expect_dcp_local_prefill_candidates(
      parallel_args_, test_config_, options_, rotary_emb_, inputs);
}

TEST_F(IndexerTest, DcpLocalCausalChunkedPrefillSelectsRankLocalCandidates) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  test_config_.index_n_heads = 32;
  TestInputs inputs = idx_create_inputs(test_config_,
                                        options_,
                                        int_option_,
                                        rotary_emb_,
                                        /*batch_size=*/1,
                                        /*max_query_len=*/24,
                                        /*is_prefill=*/true,
                                        /*chunked_prefill=*/true,
                                        /*history_len=*/24,
                                        /*use_default_rope=*/false,
                                        /*quantized_cache=*/false,
                                        /*cache_block_size=*/16);
  idx_expect_dcp_local_prefill_candidates(
      parallel_args_, test_config_, options_, rotary_emb_, inputs);
}

namespace {
void idx_expect_scores(const DcpIndexerLocalCandidates& candidates,
                       const IndexerCPPreOut& pre_out,
                       const torch::Tensor& cache,
                       const std::optional<torch::Tensor>& cache_scale,
                       const KVShardLayout& layout);
void idx_expect_entry_scores(ParallelArgs& parallel_args_,
                             TestConfig& test_config_,
                             torch::TensorOptions& options_,
                             torch::TensorOptions& int_option_,
                             std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
                             bool is_prefill,
                             bool chunked);
void idx_expect_scores(const DcpIndexerLocalCandidates& candidates,
                       const IndexerCPPreOut& pre_out,
                       const torch::Tensor& cache,
                       const std::optional<torch::Tensor>& cache_scale,
                       const KVShardLayout& layout) {
  CHECK(!cache_scale.has_value() && !pre_out.q_scale.has_value());
  const torch::Tensor golden = dcp_test::aligned_golden(pre_out.q,
                                                        pre_out.weights,
                                                        cache,
                                                        candidates.global_slots,
                                                        layout.dcp_size());
  EXPECT_TRUE(dcp_test::score_precision(
      candidates.scores, golden.to(torch::kBFloat16), golden));
  EXPECT_TRUE((candidates.global_slots >= 0).any().item<bool>());
  EXPECT_TRUE((candidates.global_slots < 0).any().item<bool>());
}
void idx_expect_entry_scores(ParallelArgs& parallel_args_,
                             TestConfig& test_config_,
                             torch::TensorOptions& options_,
                             torch::TensorOptions& int_option_,
                             std::shared_ptr<RotaryEmbeddingBase>& rotary_emb_,
                             bool is_prefill,
                             bool chunked) {
  TestInputs inputs = idx_create_inputs(test_config_,
                                        options_,
                                        int_option_,
                                        rotary_emb_,
                                        /*batch_size=*/1,
                                        /*max_query_len=*/is_prefill ? 8 : 1,
                                        is_prefill,
                                        chunked,
                                        /*history_len=*/chunked ? 8 : 0,
                                        /*use_default_rope=*/true,
                                        /*quantized_cache=*/false,
                                        /*cache_block_size=*/16);
  parallel_args_.world_size() = 2;
  parallel_args_.kv_split_size() = 2;
  const KVShardLayout layout(
      /*physical_dim=*/16, /*dcp_size=*/2, /*dcp_rank=*/0);
  // Fixed projections produce opposite query heads and signed head weights.
  inputs.x.fill_(1);
  inputs.q_norm.fill_(1);
  inputs.weights.at("wq_b.weight").zero_();
  inputs.weights.at("wq_b.weight").index_put_({0, 0}, 1);
  inputs.weights.at("wq_b.weight")
      .index_put_({test_config_.index_head_dim, 0}, -1);
  inputs.weights.at("weights_proj.weight").zero_();
  inputs.weights.at("weights_proj.weight").index_put_({0, 0}, 1);
  inputs.weights.at("weights_proj.weight").index_put_({1, 0}, -2);
  if (!is_prefill) {
    inputs.metadata.kv_seq_lens.fill_(8);
    inputs.metadata.kv_cu_seq_lens = torch::tensor({0, 8}, int_option_);
    inputs.metadata.max_seq_len = 8;
    inputs.metadata.total_kv_len = 8;
    inputs.positions.fill_(7);
  }
  idx_set_dcp_metadata(inputs, /*shards=*/2);
  DeepseekV32Indexer indexer = idx_create_indexer(parallel_args_,
                                                  test_config_,
                                                  options_,
                                                  rotary_emb_,
                                                  inputs,
                                                  /*enable_fused_qk=*/true);
  const IndexerCPPreOut reference = indexer->cp_pre(
      inputs.x, inputs.q_norm, inputs.positions, inputs.metadata);
  DcpIndexerLocalCandidates candidates;
  if (is_prefill) {
    const KVShardIndexerMetadata selector =
        build_kv_shard_indexer_metadata(inputs.metadata);
    auto local_metadata = inputs.metadata;
    local_metadata.slot_mapping =
        localize_kv_shard_slots(inputs.metadata.slot_mapping, layout);
    candidates = indexer->forward_dcp_local_prefill(inputs.x,
                                                    inputs.q_norm,
                                                    inputs.positions,
                                                    inputs.k_cache,
                                                    local_metadata,
                                                    selector);
  } else {
    // Include cached history with both signs in the decode score check.

    const auto selector = build_kv_shard_indexer_metadata(inputs.metadata);
    auto local_metadata = inputs.metadata;
    local_metadata.slot_mapping =
        localize_kv_shard_slots(inputs.metadata.slot_mapping, layout);
    candidates = indexer->forward_dcp_local(inputs.x,
                                            inputs.q_norm,
                                            inputs.positions,
                                            inputs.k_cache,
                                            local_metadata,
                                            selector);
  }
  const auto local_slots =
      localize_kv_shard_slots(inputs.metadata.slot_mapping, layout)
          .to(torch::kInt64);
  const auto owned = local_slots.ge(0);
  const auto actual_keys =
      inputs.k_cache.reshape({-1, test_config_.index_head_dim})
          .index_select(0, local_slots.masked_select(owned));
  const auto expected_keys = reference.k_local.index({owned});
  test::verify_tensor_close(actual_keys, expected_keys, 0.02, 0.002);
  idx_expect_scores(
      candidates, reference, inputs.k_cache, inputs.k_cache_scale, layout);
}
}  // namespace

TEST_F(DcpIndexerScoreTest, DecodeMatchesReluReference) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(16);

  test_config_.index_n_heads = 32;

  idx_expect_entry_scores(parallel_args_,
                          test_config_,
                          options_,
                          int_option_,
                          rotary_emb_,
                          /*is_prefill=*/false,
                          /*chunked=*/false);
}

TEST_F(DcpIndexerScoreTest, PrefillMatchesReluReference) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(16);

  test_config_.index_n_heads = 32;

  idx_expect_entry_scores(parallel_args_,
                          test_config_,
                          options_,
                          int_option_,
                          rotary_emb_,
                          /*is_prefill=*/true,
                          /*chunked=*/false);
}

TEST_F(DcpIndexerScoreTest, ChunkedPrefillMatchesReluReference) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(16);

  test_config_.index_n_heads = 32;

  idx_expect_entry_scores(parallel_args_,
                          test_config_,
                          options_,
                          int_option_,
                          rotary_emb_,
                          /*is_prefill=*/true,
                          /*chunked=*/true);
}

TEST_F(DcpIndexerScoreTest, CpCandidatesPreserveGlobalTopk) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(16);

  test_config_.index_n_heads = 32;
  parallel_args_.world_size() = 4;
  parallel_args_.kv_split_size() = 4;
  rotary_emb_ = std::make_shared<RotaryEmbeddingImpl>(
      test_config_.qk_rope_head_dim,
      test_config_.max_position_embeddings,
      test_config_.rope_theta,
      test_config_.rope_interleaved,
      options_);
  IndexerCPPreOut pre_out;
  pre_out.q = torch::zeros({2, 32, 128}, options_);
  pre_out.q.select(/*dim=*/2, /*index=*/0).fill_(1);
  pre_out.q.select(/*dim=*/1, /*index=*/1).zero_();
  pre_out.q.index_put_({torch::indexing::Slice(), 1, 1}, 1);
  pre_out.k_local = torch::zeros({2, 128}, options_);
  pre_out.weights = torch::zeros({2, 32}, options_);
  pre_out.weights.index_put_({0, 0}, 1);
  pre_out.weights.index_put_({0, 1}, 1);
  pre_out.weights.index_put_({1, 0}, -1);
  pre_out.weights.index_put_({1, 1}, 2);
  std::vector<torch::Tensor> rank_scores;
  std::vector<torch::Tensor> rank_slots;
  rank_scores.reserve(4);
  rank_slots.reserve(4);
  for (int32_t rank = 0; rank < 4; ++rank) {
    SCOPED_TRACE(rank);
    parallel_args_.rank() = rank;
    const KVShardLayout layout(
        /*physical_dim=*/16, /*dcp_size=*/4, rank);
    TestInputs inputs;
    DeepseekV32Indexer indexer = idx_create_indexer(parallel_args_,
                                                    test_config_,
                                                    options_,
                                                    rotary_emb_,
                                                    inputs,
                                                    /*enable_fused_qk=*/true);
    torch::Tensor keys = torch::zeros({33, 1, 16, 128}, options_);
    keys.select(/*dim=*/3, /*index=*/0).fill_(20);
    keys.index_put_({0, 0, 0, 0}, rank == 0 ? 10 : (rank == 1 ? 6 : 20));
    keys.index_put_({0, 0, 0, 1}, rank == 0 ? -9 : 0);
    std::optional<torch::Tensor> scales = std::nullopt;
    AttentionMetadata prefill;
    prefill.slot_mapping = torch::full({2}, -1, int_option_);
    KVShardIndexerMetadata selector;
    selector.block_table =
        torch::arange(33, int_option_).unsqueeze(0).repeat({2, 1});
    selector.kv_seq_lens = torch::full({2}, 2049, int_option_);
    selector.query_positions = selector.kv_seq_lens - 1;
    selector.q_cu_seq_lens = torch::arange(3, int_option_);
    const DcpIndexerLocalCandidates candidates =
        indexer->forward_dcp_local_prefill_from_cp(
            pre_out, keys, prefill, selector);
    idx_expect_scores(candidates, pre_out, keys, scales, layout);
    rank_scores.emplace_back(candidates.scores);
    rank_slots.emplace_back(candidates.global_slots);
  }
  DcpIndexerGatherAsyncCtx gathered;
  gathered.gathered_scores = torch::stack(rank_scores);
  gathered.gathered_global_slots = torch::stack(rank_slots);
  const DcpDecodeContext context(
      KVShardLayout(/*physical_dim=*/16, /*dcp_size=*/4, /*dcp_rank=*/0),
      /*dcp_group=*/nullptr);
  const DsaTopkState merged = context.finish_indexer_candidate_merge(
      std::move(gathered), /*topk=*/2048, torch::zeros({2}, int_option_));
  torch::Tensor selected = merged.block_tables()[0].cpu();
  torch::Tensor expected = torch::arange(2049, torch::kInt32);
  expected = expected.masked_select(expected.ne(16));
  EXPECT_TRUE(torch::equal(std::get<0>(selected.sort()), expected));
  EXPECT_TRUE(selected.eq(0).any().item<bool>());
  EXPECT_FALSE(selected.eq(16).any().item<bool>());
  EXPECT_EQ(merged.context_lens()[0].item<int32_t>(), 2048);
}

TEST(TritonLaunchCfgTest, CompileOptionsSeparateCachedKernels) {
  using triton_jit::LaunchCfg;
  using triton_jit::serialize_key;
  const LaunchCfg defaults;
  for (std::optional<bool> LaunchCfg::* field :
       {&LaunchCfg::enable_fp_fusion,
        &LaunchCfg::enable_soft_i64,
        &LaunchCfg::force_use_shared_memory}) {
    LaunchCfg disabled;
    disabled.*field = false;
    LaunchCfg enabled;
    enabled.*field = true;
    EXPECT_NE(serialize_key({}, defaults, 0), serialize_key({}, disabled, 0));
    EXPECT_NE(serialize_key({}, defaults, 0), serialize_key({}, enabled, 0));
    EXPECT_NE(serialize_key({}, disabled, 0), serialize_key({}, enabled, 0));
  }
}

TEST_F(IndexerTest, DcpRejectsUnsupportedHeads) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  parallel_args_.kv_split_size() = 2;
  parallel_args_.world_size() = 2;
  TestInputs inputs;
  EXPECT_DEATH(idx_create_indexer(parallel_args_,
                                  test_config_,
                                  options_,
                                  rotary_emb_,
                                  inputs,
                                  /*enable_fused_qk=*/false),
               "requires BF16, 32 heads");
}

TEST_F(DcpIndexerScoreTest, RejectsUnsupportedDtype) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  test_config_.index_n_heads = 32;

  parallel_args_.kv_split_size() = 2;
  parallel_args_.world_size() = 2;
  TestInputs inputs;
  EXPECT_DEATH(
      {
        options_ = options_.dtype(torch::kFloat16);
        idx_create_indexer(parallel_args_,
                           test_config_,
                           options_,
                           rotary_emb_,
                           inputs,
                           /*enable_fused_qk=*/false);
      },
      "requires BF16, 32 heads");
}

TEST_F(DcpIndexerScoreTest, RejectsUnsupportedHeadDim) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  test_config_.index_n_heads = 32;

  parallel_args_.kv_split_size() = 2;
  parallel_args_.world_size() = 2;
  TestInputs inputs;
  test_config_.index_head_dim = 64;
  EXPECT_DEATH(idx_create_indexer(parallel_args_,
                                  test_config_,
                                  options_,
                                  rotary_emb_,
                                  inputs,
                                  /*enable_fused_qk=*/false),
               "head dimension 128");
}

TEST_F(DcpIndexerScoreTest, RejectsInt8CacheConfig) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  test_config_.index_n_heads = 32;

  parallel_args_.kv_split_size() = 2;
  parallel_args_.world_size() = 2;
  TestInputs inputs;
  EXPECT_DEATH(
      {
        KVCacheConfig::get_instance().indexer_cache_dtype("int8");
        idx_create_indexer(parallel_args_,
                           test_config_,
                           options_,
                           rotary_emb_,
                           inputs,
                           /*enable_fused_qk=*/false);
      },
      "does not support INT8");
}

TEST_F(IndexerTest, CompareFusedVsNonFusedDecode) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(Platform::type_torch(), 0);
  auto ints = options.dtype(torch::kInt32);
  std::unique_ptr<ProcessGroup> group;
  auto parallel = test::create_default_parallel_args(group);
  for (bool default_rope : {false, true}) {
    SCOPED_TRACE(default_rope);
    TestConfig config;
    std::shared_ptr<RotaryEmbeddingBase> rope;
    auto opts = options;
    auto input = idx_create_inputs(
        config, opts, ints, rope, 2, 1, false, true, 4096, default_rope);
    input.metadata.is_prefill = false;
    input.metadata.is_chunked_prefill = false;
    const auto original = input.k_cache.clone();
    auto [base, base_lens] =
        idx_run_indexer(parallel, config, opts, rope, input, false, false);
    const auto slots = input.metadata.slot_mapping.to(torch::kInt64);
    const auto base_keys = input.k_cache.reshape({-1, config.index_head_dim})
                               .index_select(0, slots)
                               .clone();
    input.k_cache = original;
    auto [fused, fused_lens] =
        idx_run_indexer(parallel, config, opts, rope, input, false, true);
    EXPECT_TRUE(torch::equal(base_lens, fused_lens));
    const auto fused_keys = input.k_cache.reshape({-1, config.index_head_dim})
                                .index_select(0, slots);
    const torch::Tensor golden_keys = idx_reference_fused_k(input, rope);
    test::verify_tensor_close(
        fused_keys.cpu().to(torch::kFloat64), golden_keys, 0.02, 0.002);
    const double scale =
        torch::norm(base_keys.to(torch::kFloat32)).item<double>();
    ASSERT_GT(scale, 0.0);
    EXPECT_LE(torch::norm((fused_keys - base_keys).to(torch::kFloat32))
                      .item<double>() /
                  scale,
              0.02);
    for (int64_t row = 0; row < 2; ++row) {
      const auto expected = base[row].cpu().narrow(0, 0, config.index_topk);
      const auto actual = fused[row].cpu().narrow(0, 0, config.index_topk);
      EXPECT_TRUE(std::get<0>(actual.sort()).diff().gt(0).all().item<bool>());
      EXPECT_TRUE(actual.ge(row * 4097).all().item<bool>());
      EXPECT_TRUE(actual.lt((row + 1) * 4097).all().item<bool>());
      const int64_t matches =
          torch::isin(actual, expected).sum().item<int64_t>();
      EXPECT_GE(100 * matches, 99 * config.index_topk);
    }
  }
}

TEST_F(IndexerTest, Int8NormalPrefillWritesCacheScaleAndSelectsBlocks) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kQueryLen = 128;
  TestInputs inputs = idx_create_quantized_inputs(test_config_,
                                                  options_,
                                                  int_option_,
                                                  rotary_emb_,
                                                  kBatchSize,
                                                  kQueryLen,
                                                  /*is_prefill=*/true);

  auto [block_tables, context_lens] = idx_run_indexer(parallel_args_,
                                                      test_config_,
                                                      options_,
                                                      rotary_emb_,
                                                      inputs,
                                                      /*is_prefill=*/true,
                                                      /*enable_fused_qk=*/true);

  idx_expect_select_output(
      test_config_, block_tables, context_lens, kBatchSize * kQueryLen);
  idx_expect_quantized_cache_updated(inputs);
  idx_expect_all_visible(inputs, block_tables, context_lens);
}

TEST_F(IndexerTest, Int8ChunkedPrefillSelectsAcrossNoncontiguousPages) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kHistoryLen = 24;
  constexpr int64_t kQueryLen = 24;
  constexpr int64_t kBlockSize = 16;
  TestInputs inputs =
      idx_create_quantized_inputs(test_config_,
                                  options_,
                                  int_option_,
                                  rotary_emb_,
                                  kBatchSize,
                                  kQueryLen,
                                  /*is_prefill=*/true,
                                  /*chunked_prefill=*/true,
                                  kHistoryLen,
                                  kBlockSize,
                                  /*use_noncontiguous_blocks=*/true);
  idx_fill_quantized_cache(options_, inputs);

  auto [block_tables, context_lens] = idx_run_indexer(parallel_args_,
                                                      test_config_,
                                                      options_,
                                                      rotary_emb_,
                                                      inputs,
                                                      /*is_prefill=*/true,
                                                      /*enable_fused_qk=*/true);

  idx_expect_select_output(
      test_config_, block_tables, context_lens, kBatchSize * kQueryLen);
  idx_expect_quantized_cache_updated(inputs);
  idx_expect_all_visible(inputs, block_tables, context_lens);
}

TEST_F(IndexerTest, Int8DecodeWritesCacheScaleAndSelectsBlocks) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  constexpr int64_t kBatchSize = 16;
  TestInputs inputs = idx_create_quantized_inputs(test_config_,
                                                  options_,
                                                  int_option_,
                                                  rotary_emb_,
                                                  kBatchSize,
                                                  /*max_query_len=*/1,
                                                  /*is_prefill=*/false);

  auto [block_tables, context_lens] = idx_run_indexer(parallel_args_,
                                                      test_config_,
                                                      options_,
                                                      rotary_emb_,
                                                      inputs,
                                                      /*is_prefill=*/false,
                                                      /*enable_fused_qk=*/true);

  idx_expect_select_output(
      test_config_, block_tables, context_lens, kBatchSize);
  idx_expect_quantized_cache_updated(inputs);
  idx_expect_all_visible(inputs, block_tables, context_lens);
}

TEST_F(IndexerTest, Int8CpPrefillWritesCacheScaleAndSelectsBlocks) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  constexpr int64_t kTokenNum = 128;
  TestInputs inputs = idx_create_quantized_inputs(test_config_,
                                                  options_,
                                                  int_option_,
                                                  rotary_emb_,
                                                  /*batch_size=*/1,
                                                  kTokenNum,
                                                  /*is_prefill=*/true);
  DeepseekV32Indexer indexer = idx_create_indexer(parallel_args_,
                                                  test_config_,
                                                  options_,
                                                  rotary_emb_,
                                                  inputs,
                                                  /*enable_fused_qk=*/true);
  v32_cp::DeepseekV32CPContext cp_ctx =
      idx_make_single_rank_cp_context(options_, int_option_, inputs, kTokenNum);

  IndexerCPPreOut pre_out = indexer->cp_pre(
      inputs.x, inputs.q_norm, inputs.positions, inputs.metadata);
  EXPECT_EQ(pre_out.q.scalar_type(), torch::kBFloat16);
  EXPECT_FALSE(pre_out.q_scale.has_value());

  auto [block_tables, context_lens] =
      indexer->cp_post(pre_out,
                       pre_out.k_local,
                       inputs.k_cache,
                       inputs.metadata,
                       cp_ctx.gathered_slot_mapping,
                       cp_ctx,
                       inputs.k_cache_scale);

  idx_expect_select_output(test_config_, block_tables, context_lens, kTokenNum);
  idx_expect_quantized_cache_updated(inputs);
  idx_expect_all_visible(inputs, block_tables, context_lens);
}

TEST_F(IndexerTest, Int8CpChunkedPrefillMatchesSingleRankNormalPath) {
  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
  }
  torch::Device torch_device(Platform::type_torch(), 0);
  Device setup_device(torch_device);
  setup_device.set_seed();
  options_ = torch::TensorOptions()
                 .dtype(torch::kBFloat16)
                 .device(torch_device)
                 .requires_grad(false);
  int_option_ = options_.dtype(torch::kInt32);

  parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  KVCacheConfig::get_instance().block_size(1);

  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kHistoryLen = 24;
  constexpr int64_t kQueryLen = 24;
  constexpr int64_t kBlockSize = 16;
  TestInputs normal_inputs =
      idx_create_quantized_inputs(test_config_,
                                  options_,
                                  int_option_,
                                  rotary_emb_,
                                  kBatchSize,
                                  kQueryLen,
                                  /*is_prefill=*/true,
                                  /*chunked_prefill=*/true,
                                  kHistoryLen,
                                  kBlockSize,
                                  /*use_noncontiguous_blocks=*/true);
  idx_fill_quantized_cache(options_, normal_inputs);

  TestInputs cp_inputs = normal_inputs;
  cp_inputs.k_cache = normal_inputs.k_cache.clone();
  cp_inputs.k_cache_scale = normal_inputs.k_cache_scale->clone();

  auto [normal_block_tables, normal_context_lens] =
      idx_run_indexer(parallel_args_,
                      test_config_,
                      options_,
                      rotary_emb_,
                      normal_inputs,
                      /*is_prefill=*/true,
                      /*enable_fused_qk=*/true);

  DeepseekV32Indexer indexer = idx_create_indexer(parallel_args_,
                                                  test_config_,
                                                  options_,
                                                  rotary_emb_,
                                                  cp_inputs,
                                                  /*enable_fused_qk=*/true);
  v32_cp::DeepseekV32CPContext cp_ctx = idx_make_single_rank_cp_context(
      options_, int_option_, cp_inputs, kQueryLen);
  IndexerCPPreOut pre_out = indexer->cp_pre(
      cp_inputs.x, cp_inputs.q_norm, cp_inputs.positions, cp_inputs.metadata);
  auto [cp_block_tables, cp_context_lens] =
      indexer->cp_post(pre_out,
                       pre_out.k_local,
                       cp_inputs.k_cache,
                       cp_inputs.metadata,
                       cp_ctx.gathered_slot_mapping,
                       cp_ctx,
                       cp_inputs.k_cache_scale);

  idx_expect_select_output(
      test_config_, cp_block_tables, cp_context_lens, kQueryLen);
  idx_expect_quantized_cache_updated(cp_inputs);
  EXPECT_TRUE(torch::equal(cp_context_lens, normal_context_lens));
  idx_expect_all_visible(
      normal_inputs, normal_block_tables, normal_context_lens);
  idx_expect_all_visible(cp_inputs, cp_block_tables, cp_context_lens);
  EXPECT_TRUE(torch::equal(cp_inputs.k_cache, normal_inputs.k_cache));
  EXPECT_TRUE(
      torch::equal(*cp_inputs.k_cache_scale, *normal_inputs.k_cache_scale));
}

}  // namespace layer
}  // namespace xllm
