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

#include <cstdint>
#include <memory>
#include <vector>

#include "core/platform/layer_synchronizer.h"
#include "models/llm/mlu/glm5_next.h"

namespace xllm::mlu::model {
namespace {

TEST(Glm5NextMluDeathTest, RejectsNonSpanVerificationBeforeUsingInputs) {
  const torch::Device device(torch::kCPU);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &group);
  parallel_args.tp_group_ = &group;
  ModelArgs args;
  args.model_type("glm5_next");
  args.n_layers(0);
  args.hidden_size(4);
  args.vocab_size(8);
  args.hc_mult(1);
  args.linear_num_key_heads(1);
  const ModelContext context(
      parallel_args,
      args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  Glm5NextModelImpl model(context);
  std::vector<KVCache> caches;

  // Undefined inputs must never reach embedding, metadata building, or layers.
  const torch::Tensor tokens;
  const torch::Tensor positions;
  ModelInputParams params;
  params.is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");

  params.is_spec_verify = false;
  params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
  params.attn_metadata->is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");
}

TEST(Glm5NextMluMetadataTest,
     CancelledForwardPreservesCallerAcrossPrefixReuse) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const torch::Device device(torch::kCPU);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.model_type("glm5_next")
      .n_layers(1)
      .layer_types({"linear_attention"})
      .mlp_layer_types({"dense"})
      .hidden_size(128)
      .intermediate_size(256)
      .hidden_act("silu")
      .vocab_size(8)
      .hc_mult(4)
      .hc_sinkhorn_iters(4)
      .hc_eps(1e-6)
      .rms_norm_eps(1e-6)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .linear_lower_bound(-5.0f)
      .num_speculative_tokens(1);
  Glm5NextModelImpl model(ModelContext(parallel, args, QuantArgs(), options));
  std::vector<KVCache> caches;
  caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::zeros({4, 4, 384}, options),
                                    torch::zeros({8, 1, 128, 128}, options)});
  // This is the real failed-copy lifecycle: metadata is prepared before the
  // layer wait aborts, so no decoder weights or device kernels are consumed.
  auto synchronizer = create_layer_synchronizer(/*num_layers=*/1);
  ASSERT_NE(synchronizer, nullptr);
  synchronizer->abort();
  ASSERT_FALSE(synchronizer->synchronize_layer(/*layer_index=*/0));
  const auto make_input = [&](BatchForwardType type) {
    ModelInputParams params;
    params.is_spec_verify = type.is_chunked_prefill();
    const int32_t width = params.is_spec_verify ? 2 : 1;
    params.meta.batch_forward_type = type;
    params.meta.num_sequences = 2;
    params.meta.actual_num_sequences = 2;
    params.meta.q_max_seq_len = width;
    params.meta.kv_max_seq_len = 4 + width;
    params.attention.host.q_seq_lens = {0, width, 2 * width};
    params.attention.host.q_cu_seq_lens = params.attention.host.q_seq_lens;
    params.attention.host.kv_seq_lens = {0, 4 + width, 2 * (4 + width)};
    params.attention.host.kv_cache_tokens_nums = {4, 4};
    params.attention.device.q_seq_lens =
        torch::tensor({0, width, 2 * width}, torch::kInt32);
    params.attention.device.q_cu_seq_lens = params.attention.device.q_seq_lens;
    params.attention.device.block_tables =
        torch::tensor({{1}, {2}}, torch::kInt32);
    params.attention.device.kv_seq_lens =
        torch::tensor({0, 4 + width, 2 * (4 + width)}, torch::kInt32);
    params.attention.device.new_cache_slots =
        torch::arange(2 * width, torch::kInt32);
    params.embedding.input_embedding = torch::ones({2 * width, 128}, options);
    params.embedding.linear_state_ids = {2, 3};
    params.embedding.linear_state_indices =
        torch::tensor({2, 3}, torch::kInt32);
    params.embedding.linear_state_read_ids = {1, 1};
    params.linear_state_validity_mask = {1, 1};
    params.parallel.dp_global_token_nums = {2 * width};
    params.parallel.dp_is_decode = {type.is_decode() ? 1 : 0};
    params.parallel.layers_per_event = 1;
    params.parallel.layer_wise_load_synchronizer = synchronizer;
    if (params.is_spec_verify) {
      params.num_accepted_tokens = torch::ones({2}, torch::kInt32);
    }
    return params;
  };
  const auto run_forward = [&](const ModelInputParams& params) {
    const int32_t width = params.meta.q_max_seq_len;
    const torch::Tensor tokens = torch::ones({2 * width}, torch::kInt32);
    const torch::Tensor positions =
        torch::arange(4, 4 + width, torch::kInt32).repeat({2});
    return model.forward(tokens, positions, caches, params);
  };
  auto cold = make_input(BatchForwardType::PREFILL);
  EXPECT_FALSE(run_forward(cold).hidden_states.defined());
  EXPECT_EQ(cold.attn_metadata, nullptr);
  cold.embedding.linear_state_read_indices =
      torch::tensor({1, 1}, torch::kInt32);
  EXPECT_FALSE(run_forward(cold).hidden_states.defined());
  EXPECT_EQ(cold.attn_metadata, nullptr);
  cold.embedding.linear_state_read_indices = torch::tensor({1}, torch::kInt32);
  EXPECT_DEATH(run_forward(cold), "source index count");

  const auto attach_cached_metadata = [](ModelInputParams& params) {
    params.attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(params, /*enable_mla=*/false));
    // A caller may reuse graph metadata that still owns a previous prefix's
    // read slots. All slots here are in the allocated recurrent pool.
    params.attn_metadata->linear_state_read_indices =
        torch::tensor({3, 3}, torch::kInt32);
  };
  const auto expect_caller_preserved = [&](const ModelInputParams& params) {
    ASSERT_NE(params.attn_metadata, nullptr);
    const auto caller = params.attn_metadata;
    const torch::Tensor sources = caller->linear_state_read_indices;
    const torch::Tensor source_values = sources.clone();
    const torch::Tensor destination = caller->linear_state_indices;
    const torch::Tensor destination_values = destination.clone();
    const torch::Tensor query_lengths = caller->q_cu_seq_lens;
    const torch::Tensor query_values = query_lengths.clone();
    const torch::Tensor embeddings = params.embedding.input_embedding.clone();
    EXPECT_FALSE(run_forward(params).hidden_states.defined());
    EXPECT_EQ(params.attn_metadata, caller);
    ASSERT_TRUE(caller->linear_state_read_indices.defined());
    EXPECT_EQ(caller->linear_state_read_indices.data_ptr(), sources.data_ptr());
    EXPECT_TRUE(torch::equal(caller->linear_state_read_indices, source_values));
    EXPECT_EQ(caller->linear_state_indices.data_ptr(), destination.data_ptr());
    EXPECT_TRUE(torch::equal(caller->linear_state_indices, destination_values));
    EXPECT_EQ(caller->q_cu_seq_lens.data_ptr(), query_lengths.data_ptr());
    EXPECT_TRUE(torch::equal(caller->q_cu_seq_lens, query_values));
    EXPECT_TRUE(torch::equal(params.embedding.input_embedding, embeddings));
  };
  auto cached = make_input(BatchForwardType::PREFILL);
  attach_cached_metadata(cached);
  expect_caller_preserved(cached);
  cached.embedding.linear_state_read_indices =
      torch::tensor({1, 1}, torch::kInt32);
  expect_caller_preserved(cached);
  cached.embedding.linear_state_read_ids = {2, 2};
  cached.embedding.linear_state_read_indices =
      torch::tensor({2, 2}, torch::kInt32);
  expect_caller_preserved(cached);
  // Validation must read this call's source buffer, not silently reuse the
  // valid cached buffer from the previous prefix.
  cached.embedding.linear_state_read_indices =
      torch::tensor({2}, torch::kInt32);
  EXPECT_DEATH(run_forward(cached), "source index count");
  cached.embedding.linear_state_read_indices =
      torch::tensor({2, 2}, torch::kInt32).view({1, 2});
  EXPECT_DEATH(run_forward(cached), "source indices must be 1D");
  cached.embedding.linear_state_read_indices =
      torch::tensor({2, 2}, torch::kFloat32);
  EXPECT_DEATH(run_forward(cached), "source indices must be integers");

  auto decode = make_input(BatchForwardType::DECODE);
  attach_cached_metadata(decode);
  expect_caller_preserved(decode);
  auto mixed = make_input(BatchForwardType::MIXED);
  attach_cached_metadata(mixed);
  expect_caller_preserved(mixed);
  auto verify = make_input(BatchForwardType::CHUNKED_PREFILL);
  attach_cached_metadata(verify);
  expect_caller_preserved(verify);
}

}  // namespace
}  // namespace xllm::mlu::model
