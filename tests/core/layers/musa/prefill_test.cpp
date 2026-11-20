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

#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/layers/musa/flashinfer_attention.h"
#include "core/layers/musa/qwen3_next_attention.h"
#include "core/layers/musa/qwen3_next_gated_delta_net.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace xllm::layer {
namespace {

TEST(MusaPrefillTest, GdnCheckpointStridePreservesPrefillOutputAndState) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::NoGradGuard no_grad;
  torch::manual_seed(831);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.hidden_size(256);
  args.linear_num_key_heads(2);
  args.linear_num_value_heads(4);
  args.linear_key_head_dim(128);
  args.linear_value_head_dim(128);
  args.linear_conv_kernel_dim(4);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  Qwen3_5GatedDeltaNet layer(args, QuantArgs(), parallel, options);
  for (auto& parameter : layer->named_parameters()) {
    parameter.value().copy_(torch::randn(parameter.value().sizes()) * 0.05);
  }
  layer->named_parameters()["norm.weight"].fill_(1.0);
  const torch::Tensor hidden = torch::randn({7, 256}).to(options);
  AttentionMetadata metadata{};
  metadata.max_query_len = 4;
  metadata.q_seq_lens_vec = {3, 4};
  metadata.q_cu_seq_lens =
      torch::tensor({0, 3, 7}, options.dtype(torch::kInt32));
  metadata.has_initial_states = torch::zeros({2}, options.dtype(torch::kBool));
  ModelInputParams params;
  params.embedding.linear_state_ids = {0, 1};
  params.linear_state_validity_mask = {0, 0};
  // The MTP layout has four SSM checkpoints and six conv-history tokens per
  // slot, but ordinary prefill must retain the non-MTP numerical contract.
  for (const bool chunked : {false, true}) {
    SCOPED_TRACE(chunked);
    metadata.is_prefill = !chunked;
    metadata.is_chunked_prefill = chunked;
    KVCache baseline(LinearAttentionKVCacheTensors{
        torch::zeros({2, 1024, 3}, options),
        torch::zeros({2, 4, 128, 128}, options.dtype(torch::kFloat32))});
    KVCache checkpointed(LinearAttentionKVCacheTensors{
        torch::zeros({2, 1024, 6}, options),
        torch::zeros({8, 4, 128, 128}, options.dtype(torch::kFloat32))});
    const torch::Tensor expected =
        layer->forward(hidden, metadata, baseline, params).cpu();
    const torch::Tensor expected_state = baseline.get_ssm_cache().cpu();
    const torch::Tensor actual =
        layer->forward(hidden, metadata, checkpointed, params).cpu();
    const torch::Tensor actual_state =
        checkpointed.get_ssm_cache()
            .index_select(0,
                          torch::tensor({0, 4}, options.dtype(torch::kInt64)))
            .cpu();
    ASSERT_GT(expected_state.abs().max().item<float>(), 0.0f);
    EXPECT_TRUE(torch::equal(actual, expected));
    EXPECT_TRUE(torch::equal(actual_state, expected_state));
  }
}

TEST(MusaPrefillTest, FusedQkvScalesFollowTensorParallelWeightPartitions) {
  QuantArgs quant;
  quant.quant_method(kQuantMethodFp8);
  quant.weight_block_size() = {128, 128};
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor weight = (torch::arange(3072, torch::kFloat32) / 512)
                                   .unsqueeze(1)
                                   .expand({3072, 256})
                                   .to(torch::kFloat8_e4m3fn);
  const torch::Tensor scale =
      torch::arange(48, torch::kFloat32).view({24, 2}) + 1;
  const StateDict weights({{"weight", weight}});
  const StateDict scales({{"weight_scale_inv", scale}});
  const auto weight_parts = weight.chunk(3, 0);
  const auto scale_parts = scale.chunk(3, 0);
  for (const int32_t world_size : {4, 8}) {
    const int32_t rank = world_size - 1;
    SCOPED_TRACE(world_size);
    ProcessGroup group(rank, world_size, torch::Device(torch::kCPU));
    ParallelArgs parallel(rank, world_size, &group);
    parallel.tp_group_ = &group;
    const torch::Tensor expected_weight =
        torch::cat({weight_parts[0].chunk(world_size, 0)[rank],
                    weight_parts[1].chunk(world_size, 0)[rank],
                    weight_parts[2].chunk(world_size, 0)[rank]},
                   0);
    const torch::Tensor expected_scale =
        torch::cat({scale_parts[0].chunk(world_size, 0)[rank],
                    scale_parts[1].chunk(world_size, 0)[rank],
                    scale_parts[2].chunk(world_size, 0)[rank]},
                   0);
    QKVParallelLinear qkv(/*hidden_size=*/256,
                          /*num_heads=*/8 / world_size,
                          /*num_kv_heads=*/8 / world_size,
                          /*head_size=*/128,
                          /*num_kv_head_replicas=*/1,
                          /*bias=*/false,
                          /*gather_output=*/false,
                          parallel,
                          options,
                          quant);
    // Exercise both delayed scale and delayed weight without separate cases.
    qkv->load_state_dict(world_size == 4 ? weights : scales);
    qkv->load_state_dict(world_size == 4 ? scales : weights);
    EXPECT_TRUE(torch::equal(qkv->weight().to(torch::kFloat32),
                             expected_weight.to(torch::kFloat32)));
    EXPECT_TRUE(torch::equal(qkv->weight_scale_inv(), expected_scale));
  }
}

TEST(MusaPrefillTest, LoadsCheckpointShardsWithoutPrematureQgReorder) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.n_heads(12);
  args.n_kv_heads(2);
  args.head_dim(256);
  args.hidden_size(256);
  args.max_position_embeddings(128);
  args.attn_output_gate(true);
  args.partial_rotary_factor(0.25f);
  QuantArgs quant;
  quant.quant_method(kQuantMethodFp8);
  quant.weight_block_size() = {128, 128};
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  const torch::Tensor q =
      (torch::floor(torch::arange(6144, torch::kFloat32) / 256) / 4)
          .unsqueeze(1)
          .expand({6144, 256})
          .to(torch::kFloat8_e4m3fn)
          .to(torch::kFloat32);
  const torch::Tensor kv = torch::ones({512, 256}, torch::kFloat32);
  const torch::Tensor q_scale =
      torch::arange(96, torch::kFloat32).view({48, 2}) + 1;
  const StateDict scales({{"q_proj.weight_scale_inv", q_scale},
                          {"k_proj.weight_scale_inv", torch::ones({4, 2})},
                          {"v_proj.weight_scale_inv", torch::full({4, 2}, 2)}});
  const torch::Tensor expected_q = torch::cat(
      {q.view({12, 512, 256}).slice(1, 0, 256).reshape({3072, 256}),
       q.view({12, 512, 256}).slice(1, 256, 512).reshape({3072, 256})},
      0);
  const torch::Tensor expected_scale =
      torch::cat({q_scale.view({12, 4, 2}).slice(1, 0, 2).reshape({24, 2}),
                  q_scale.view({12, 4, 2}).slice(1, 2, 4).reshape({24, 2})},
                 0);
  // Weight-first FP8, scale-first FP8, and a BF16 checkpoint in FP8 config.
  for (const int32_t scenario : {0, 1, 2}) {
    SCOPED_TRACE(scenario);
    Qwen3NextAttentionImpl attention(args, quant, parallel, options, 0);
    const torch::ScalarType dtype =
        scenario == 2 ? torch::kBFloat16 : torch::kFloat8_e4m3fn;
    const StateDict weights({{"q_proj.weight", q.to(dtype)},
                             {"k_proj.weight", kv.to(dtype)},
                             {"v_proj.weight", kv.to(dtype)}});
    const StateDict& first = scenario == 1 ? scales : weights;
    std::unordered_map<std::string, torch::Tensor> first_shard(first.begin(),
                                                               first.end());
    first_shard.emplace("o_proj.weight", torch::full({256, 3072}, 3, options));
    first_shard.emplace("q_norm.weight", torch::full({256}, 4, options));
    first_shard.emplace("k_norm.weight", torch::full({256}, 5, options));
    attention.load_state_dict(StateDict(std::move(first_shard)));
    auto named = attention.named_parameters();
    EXPECT_TRUE(torch::equal(named["o_proj.weight"],
                             torch::full({256, 3072}, 3, options)));
    EXPECT_TRUE(
        torch::equal(named["q_norm.weight"], torch::full({256}, 4, options)));
    EXPECT_TRUE(
        torch::equal(named["k_norm.weight"], torch::full({256}, 5, options)));
    if (scenario == 0) {
      EXPECT_TRUE(torch::equal(
          named["qkv_proj.weight"].slice(0, 0, 6144).to(torch::kFloat32), q));
    }
    attention.load_state_dict(scenario == 1 ? weights : scales);
    attention.load_state_dict(weights);
    attention.load_state_dict(scales);
    named = attention.named_parameters();
    EXPECT_EQ(named["qkv_proj.weight"].scalar_type(), dtype);
    EXPECT_TRUE(torch::equal(
        named["qkv_proj.weight"].slice(0, 0, 6144).to(torch::kFloat32),
        expected_q));
    EXPECT_TRUE(torch::equal(
        named["qkv_proj.weight"].slice(0, 6144).to(torch::kFloat32),
        torch::cat({kv, kv}, 0)));
    if (scenario != 2) {
      EXPECT_TRUE(torch::equal(
          named["qkv_proj.weight_scale_inv"].slice(0, 0, 48), expected_scale));
    }
  }
  for (const bool split_shards : {false, true}) {
    EXPECT_DEATH(
        ([&] {
          Qwen3NextAttentionImpl attention(args, quant, parallel, options, 0);
          const StateDict q_weight(
              {{"q_proj.weight", q.to(torch::kFloat8_e4m3fn)}});
          if (split_shards) {
            attention.load_state_dict(q_weight);
            attention.load_state_dict(
                StateDict({{"k_proj.weight", kv.to(torch::kBFloat16)}}));
          } else {
            attention.load_state_dict(
                StateDict({{"q_proj.weight", q.to(torch::kFloat8_e4m3fn)},
                           {"k_proj.weight", kv.to(torch::kBFloat16)}}));
          }
        }()),
        "mixed FP8 and unquantized weights");
  }
}

TEST(MusaPrefillTest, ThreeAxisPositionsRequireMropeConfiguration) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.n_heads(12);
  args.n_kv_heads(2);
  args.head_dim(256);
  args.hidden_size(256);
  args.max_position_embeddings(128);
  args.partial_rotary_factor(0.25f);
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  Qwen3NextAttentionImpl attention(args, QuantArgs(), parallel, options, 0);
  AttentionMetadata metadata{};
  KVCache cache;
  EXPECT_DEATH(attention.forward(torch::zeros({3, 1}, torch::kInt64),
                                 torch::zeros({1, 256}, options),
                                 metadata,
                                 cache,
                                 torch::Tensor()),
               "require mrope_section in the model configuration");
}

TEST(MusaPrefillTest, CachedPrefixesMatchCausalReference) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::manual_seed(758);
  const c10::musa::MUSAStream stream = c10::musa::getStreamFromPool();
  const c10::musa::MUSAStreamGuard stream_guard(stream);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  for (const int64_t ratio : {3, 6, 8}) {
    SCOPED_TRACE(ratio);
    const int64_t heads = 2 * ratio;
    const torch::Tensor query =
        torch::randn({5, heads, 256}).to(torch::kBFloat16);
    const torch::Tensor full_key =
        torch::randn({13, 2, 256}).to(torch::kBFloat16);
    const torch::Tensor full_value =
        torch::randn({13, 2, 256}).to(torch::kBFloat16);
    torch::Tensor k_cache = torch::zeros({2, 64, 2, 256}, options);
    torch::Tensor v_cache = torch::zeros_like(k_cache);
    k_cache[0].narrow(0, 0, 3).copy_(full_key.narrow(0, 0, 3));
    k_cache[1].narrow(0, 0, 5).copy_(full_key.narrow(0, 5, 5));
    v_cache[0].narrow(0, 0, 3).copy_(full_value.narrow(0, 0, 3));
    v_cache[1].narrow(0, 0, 5).copy_(full_value.narrow(0, 5, 5));
    KVCache cache(KVCacheTensors{k_cache, v_cache});
    const torch::Tensor new_key =
        torch::cat({full_key.narrow(0, 3, 2), full_key.narrow(0, 10, 3)}, 0);
    const torch::Tensor new_value = torch::cat(
        {full_value.narrow(0, 3, 2), full_value.narrow(0, 10, 3)}, 0);
    torch::Tensor device_query = query.to(device.unwrap());
    torch::Tensor device_key = new_key.to(device.unwrap());
    torch::Tensor device_value = new_value.to(device.unwrap());
    torch::Tensor output = torch::empty_like(device_query);
    AttentionMetadata metadata{};
    metadata.is_prefill = true;
    metadata.is_causal = true;
    metadata.max_query_len = 3;
    metadata.max_seq_len = 8;
    metadata.q_seq_lens_vec = {2, 3};
    metadata.kv_seq_lens_vec = {5, 8};
    const auto int_options = options.dtype(torch::kInt32);
    metadata.q_cu_seq_lens = torch::tensor({0, 2, 5}, int_options);
    metadata.kv_seq_lens = torch::tensor({5, 8}, int_options);
    metadata.block_table = torch::tensor({{0}, {1}}, int_options);
    metadata.slot_mapping = torch::tensor({3, 4, 69, 70, 71}, int_options);
    FlashInferAttentionImpl attention(heads, 256, 0.0625f, 2, -1);
    const auto initial_key = k_cache.clone();
    const auto initial_value = v_cache.clone();
    for (const bool chunked : {false, true}) {
      cache =
          KVCache(KVCacheTensors{initial_key.clone(), initial_value.clone()});
      SCOPED_TRACE(chunked);
      metadata.is_prefill = !chunked;
      metadata.is_chunked_prefill = chunked;
      attention.forward(
          metadata, device_query, device_key, device_value, output, cache);
      const torch::Tensor actual = output.cpu().view({5, heads, 256});
      int64_t query_offset = 0;
      int64_t kv_offset = 0;
      for (const int64_t row : {0, 1}) {
        const int64_t q_len = metadata.q_seq_lens_vec[row];
        const int64_t kv_len = metadata.kv_seq_lens_vec[row];
        const torch::Tensor q = query.narrow(0, query_offset, q_len)
                                    .to(torch::kFloat32)
                                    .transpose(0, 1);
        const torch::Tensor k = full_key.narrow(0, kv_offset, kv_len)
                                    .repeat_interleave(ratio, 1)
                                    .to(torch::kFloat32)
                                    .transpose(0, 1);
        const torch::Tensor v = full_value.narrow(0, kv_offset, kv_len)
                                    .repeat_interleave(ratio, 1)
                                    .to(torch::kFloat32)
                                    .transpose(0, 1);
        const torch::Tensor mask =
            torch::arange(kv_len).unsqueeze(0) >
            torch::arange(q_len).unsqueeze(1) + kv_len - q_len;
        const torch::Tensor scores =
            (torch::matmul(q, k.transpose(1, 2)) * 0.0625)
                .masked_fill(mask, -std::numeric_limits<float>::infinity());
        const torch::Tensor expected =
            torch::matmul(torch::softmax(scores, -1), v).transpose(0, 1);
        EXPECT_TRUE(torch::allclose(
            actual.narrow(0, query_offset, q_len).to(torch::kFloat32),
            expected,
            /*rtol=*/0.015,
            /*atol=*/0.015));
        const auto observed =
            actual.narrow(0, query_offset, q_len).to(torch::kFloat32);
        const double reference_norm = torch::norm(expected, 2.0).item<double>();
        ASSERT_GT(reference_norm, 0.0);
        EXPECT_LE(torch::norm(observed - expected, 2.0).item<double>() /
                      reference_norm,
                  0.015);
        query_offset += q_len;
        kv_offset += kv_len;
      }
      EXPECT_TRUE(torch::equal(cache.get_k_cache()[0].narrow(0, 0, 5).cpu(),
                               full_key.narrow(0, 0, 5)));
      EXPECT_TRUE(torch::equal(cache.get_k_cache()[1].narrow(0, 0, 8).cpu(),
                               full_key.narrow(0, 5, 8)));
      EXPECT_TRUE(torch::equal(cache.get_v_cache()[0].narrow(0, 0, 5).cpu(),
                               full_value.narrow(0, 0, 5)));
      EXPECT_TRUE(torch::equal(cache.get_v_cache()[1].narrow(0, 0, 8).cpu(),
                               full_value.narrow(0, 5, 8)));
    }
    // Reuse the checked two-sequence result for decode warp boundaries and
    // paged prefill beyond the 32-warp single-CTA capacity.
    const bool decode = ratio != 8;
    const std::vector<int64_t> batch_sizes =
        decode ? std::vector<int64_t>{32, 62, 64} : std::vector<int64_t>{994};
    for (const int64_t batch_size : batch_sizes) {
      SCOPED_TRACE(batch_size);
      const int64_t copies = batch_size / 2;
      const int64_t query_length = decode ? 1 : 3;
      const torch::Tensor rows = torch::tensor({1, 4}, int_options);
      torch::Tensor batched_query =
          (decode ? device_query.index_select(0, rows) : device_query)
              .repeat({copies, 1, 1});
      torch::Tensor expected = output.view_as(device_query);
      expected = (decode ? expected.index_select(0, rows) : expected)
                     .repeat({copies, 1, 1});
      const torch::Tensor cu_query =
          decode ? torch::arange(batch_size + 1, int_options)
                 : torch::cat({torch::zeros({1}, int_options),
                               torch::tensor({2, 3}, int_options)
                                   .repeat({copies})
                                   .cumsum(0, torch::kInt32)});
      torch::Tensor lengths = metadata.kv_seq_lens.repeat({copies});
      const torch::Tensor pages = metadata.block_table.repeat({copies, 1});
      torch::Tensor batched_output = torch::empty_like(batched_query);
      torch::Tensor lse = torch::empty({heads, batched_query.size(0)},
                                       options.dtype(torch::kFloat32));
      const auto make_metadata = [&]() {
        const auto scheduler =
            decode ? kernel::musa::fa3_decode_scheduler_metadata
                   : kernel::musa::fa3_prefill_scheduler_metadata;
        return scheduler(device.unwrap(),
                         batch_size,
                         heads,
                         /*num_heads_kv=*/2,
                         /*head_dim_qk=*/256,
                         /*head_dim_vo=*/256,
                         query_length,
                         /*max_seq_len=*/8,
                         /*window_size_left=*/-1,
                         /*window_size_right=*/0,
                         cu_query,
                         lengths);
      };
      torch::Tensor scheduler_metadata = make_metadata();
      const torch::Tensor populated_k_cache = cache.get_k_cache();
      const torch::Tensor populated_v_cache = cache.get_v_cache();
      const auto forward = [&]() {
        const auto run_attention =
            decode ? kernel::musa::fa3_decode : kernel::musa::fa3_prefill_paged;
        run_attention(batched_query,
                      populated_k_cache,
                      populated_v_cache,
                      cu_query,
                      lengths,
                      pages,
                      scheduler_metadata,
                      query_length,
                      /*window_left=*/-1,
                      /*window_right=*/0,
                      /*sm_scale=*/0.0625,
                      batched_output,
                      lse);
      };
      forward();
      EXPECT_TRUE(torch::allclose(batched_output.to(torch::kFloat32),
                                  expected.to(torch::kFloat32),
                                  /*rtol=*/0.015,
                                  /*atol=*/0.003));
      if (decode && batch_size == 32) {
        ASSERT_EQ(musaDeviceSynchronize(), musaSuccess);
        at::musa::MUSAGraph graph;
        graph.capture_begin();
        forward();
        graph.capture_end();
        lengths.sub_(1);
        batched_query.mul_(0.5);
        scheduler_metadata.copy_(make_metadata());
        forward();
        expected = batched_output.clone();
        batched_output.fill_(std::numeric_limits<float>::quiet_NaN());
        graph.replay();
        EXPECT_TRUE(torch::allclose(batched_output.to(torch::kFloat32),
                                    expected.to(torch::kFloat32),
                                    /*rtol=*/0.015,
                                    /*atol=*/0.003));
      }
    }
  }
}

}  // namespace
}  // namespace xllm::layer
