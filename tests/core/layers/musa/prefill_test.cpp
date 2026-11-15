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
#include "core/layers/musa/flashinfer_attention.h"
#include "core/layers/musa/qwen3_next_attention.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm::layer {
namespace {

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
  AttentionMetadata metadata;
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
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  for (const int64_t ratio : {6, 8}) {
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
    AttentionMetadata metadata;
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
    for (const bool chunked : {false, true}) {
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
        query_offset += q_len;
        kv_offset += kv_len;
      }
    }
  }
}

}  // namespace
}  // namespace xllm::layer
