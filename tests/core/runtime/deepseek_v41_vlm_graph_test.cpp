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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "core/framework/model_context.h"
#include "core/runtime/mlu_graph_executor_impl.h"
#include "models/model_registry.h"
#include "tests/core/layers/mlu/tests_utils.h"
#include "tests/core/runtime/deepseek_v41_test_utils.h"

namespace xllm {
namespace {

TEST(DeepseekV41VlmGraphTest, EncodesImagesAndReplaysDecodeWithFreshMetadata) {
  const torch::NoGradGuard no_grad;
  torch::manual_seed(41);
  const torch::Device device("mlu:0");
  const torch::DeviceGuard guard(device);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const auto ints = options.dtype(torch::kInt32);
  ModelArgs args;
  args.model_type("deepseek_v41")
      .dtype("bfloat16")
      .hidden_size(5120)
      .vocab_size(128)
      .n_layers(1)
      .n_heads(8)
      .n_kv_heads(1)
      .head_dim(512)
      .q_lora_rank(1280)
      .qk_rope_head_dim(64)
      .o_groups(1)
      .o_lora_rank(1024)
      .window_size(16)
      .compress_ratios({0})
      .max_position_embeddings(512)
      .compress_rope_theta(10000)
      .factor(1)
      .beta_fast(32)
      .beta_slow(1)
      .hc_mult(4)
      .hc_sinkhorn_iters(20)
      .hc_eps(1e-6)
      .rms_norm_eps(1e-20)
      .n_routed_experts(4)
      .num_experts_per_tok(2)
      .n_group(2)
      .topk_group(2)
      .routed_scaling_factor(1)
      .norm_topk_prob(true)
      .moe_intermediate_size(128)
      .hidden_act("silu")
      .scoring_func("sqrtsoftplus")
      .topk_method("noaux_tc")
      .swiglu_limit(10)
      .image_token_id(126)
      .mm_hidden_size(8)
      .mm_num_attention_heads(1)
      .mm_head_dim(8)
      .mm_num_hidden_layers(1)
      .mm_intermediate_size(8)
      .mm_patch_size(2)
      .mm_num_channels(3)
      .mm_downsample_ratio(2)
      .mm_rope_theta(10000)
      .mm_projection_dim(5120);
  layer::test::MockProcessGroup group(device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  const ModelContext context(parallel, args, QuantArgs(), options);
  auto model = create_vlm_model(context);
  ASSERT_NE(model, nullptr);
  ASSERT_TRUE(model->requires_graph_forward_metadata());

  const auto random = [&](std::vector<int64_t> shape) {
    return (torch::randn(shape,
                         options.device(torch::kCPU).dtype(torch::kFloat32)) *
            0.05)
        .to(options);
  };
  std::unordered_map<std::string, torch::Tensor> weights{
      {"embed.weight", random({128, 5120})},
      {"head.weight", random({128, 5120})},
      {"norm.weight", torch::ones({5120}, options)},
      {"layers.0.attn_norm.weight", torch::ones({5120}, options)},
      {"layers.0.ffn_norm.weight", torch::ones({5120}, options)},
      {"layers.0.attn.wq_a.weight", random({1280, 5120})},
      {"layers.0.attn.wkv.weight", random({512, 5120})},
      {"layers.0.attn.q_norm.weight", torch::ones({1280}, options)},
      {"layers.0.attn.kv_norm.weight", torch::ones({512}, options)},
      {"layers.0.attn.wq_b.weight", random({4096, 1280})},
      {"layers.0.attn.wo_a.weight_scale",
       torch::full({1024, 1}, 1.0 / 256, options)},
      {"layers.0.attn.wo_b.weight_scale",
       torch::full({5120, 1}, 1.0 / 256, options)},
      {"layers.0.attn.wo_a.smooth", torch::ones({4096}, options)},
      {"layers.0.attn.wo_b.smooth", torch::ones({1024}, options)},
      {"layers.0.attn.wo_a.weight",
       torch::randint(-16, 17, {1024, 4096}, torch::kInt8)
           .to(options.device())},
      {"layers.0.attn.wo_b.weight",
       torch::randint(-16, 17, {5120, 1024}, torch::kInt8)
           .to(options.device())},
      {"layers.0.attn.attn_sink", torch::zeros({8}, options)},
      {"layers.0.ffn.gate.weight", random({4, 5120})},
      {"layers.0.ffn.gate.bias",
       torch::tensor({20.0f, 10.0f, 0.0f, -10.0f},
                     options.dtype(torch::kFloat32))}};
  for (const std::string& name : {"hc_attn_", "hc_ffn_"}) {
    const std::string prefix = "layers.0." + name;
    weights.emplace(prefix + "fn",
                    random({24, 4 * args.hidden_size()}).to(torch::kFloat32));
    weights.emplace(prefix + "base", random({24}).to(torch::kFloat32));
    weights.emplace(prefix + "scale",
                    torch::full({3}, 0.1, options.dtype(torch::kFloat32)));
  }
  for (int32_t expert = 0; expert < 4; ++expert) {
    const std::string prefix =
        "layers.0.ffn.experts." + std::to_string(expert) + ".";
    weights.emplace(prefix + "w1.weight", random({128, 5120}));
    weights.emplace(prefix + "w3.weight", random({128, 5120}));
    weights.emplace(prefix + "w2.weight", random({5120, 128}));
  }
  weights.emplace("layers.0.ffn.gate.bias_vl",
                  torch::tensor({-10.0f, 0.0f, 10.0f, 20.0f}));
  const std::vector<std::pair<std::string, std::vector<int64_t>>> vision_shapes{
      {"vision.patch_embed.proj.weight", {8, 12}},
      {"vision.patch_embed.proj.bias", {8}},
      {"vision.norm.weight", {8}},
      {"image_start", {5120}},
      {"image_end", {5120}},
      {"image_newline", {5120}},
      {"aligner.w1.weight", {5120, 32}},
      {"aligner.w1.bias", {5120}},
      {"aligner.w2.weight", {5120, 5120}},
      {"aligner.w2.bias", {5120}},
      {"vision.blocks.0.attn.wqkv.weight", {24, 8}},
      {"vision.blocks.0.attn.wqkv.bias", {24}},
      {"vision.blocks.0.attn.wo.weight", {8, 8}},
      {"vision.blocks.0.attn.wo.bias", {8}},
      {"vision.blocks.0.mlp.w1.weight", {16, 8}},
      {"vision.blocks.0.mlp.w2.weight", {8, 8}},
      {"vision.blocks.0.norm1.weight", {8}},
      {"vision.blocks.0.norm2.weight", {8}}};
  for (const auto& [name, shape] : vision_shapes) {
    weights.emplace(name, random(shape));
  }
  model->load_model(
      std::make_unique<test::TensorModelLoader>(StateDict(std::move(weights))));
  ModelInputParams image_input;
  image_input.multimodal.mm_data =
      MMBatchData(MMType::IMAGE,
                  {{"pixel_values", random({4, 12})},
                   {"image_grid_thw", torch::tensor({2, 2}, torch::kInt64)}});
  MMDict encoded = model->encode(image_input);
  auto images =
      std::get<std::vector<torch::Tensor>>(encoded.at("image|embedding"));
  ASSERT_EQ(images.size(), 1);
  ASSERT_EQ(images[0].size(0), 4);
  ModelInputParams merged;
  merged.multimodal.mm_data = MMBatchData(
      MMType::IMAGE,
      {{"image|embedding", images[0]},
       {"image|mask", torch::tensor({true, true, false, true, true})}});
  torch::Tensor image_tokens = torch::tensor({126, 126, 127, 126, 126}, ints);
  torch::Tensor embeddings = model->get_input_embeddings(image_tokens, merged);
  EXPECT_TRUE(torch::equal(embeddings.index({torch::tensor(
                               {0, 1, 3, 4}, options.dtype(torch::kLong))}),
                           images[0]));
  EXPECT_TRUE(torch::equal(
      embeddings[2],
      model->get_input_embeddings(image_tokens, ModelInputParams())[2]));
  DeepSeekV4KVCacheTensors cache_tensors;
  cache_tensors.swa_cache = random({8, 128, 1, 512});
  std::vector<KVCache> caches;
  caches.reserve(1);
  caches.emplace_back(cache_tensors);
  const torch::Tensor original_cache = caches[0].get_swa_cache().clone();

  merged.meta.batch_forward_type = BatchForwardType::PREFILL;
  merged.meta.num_sequences = 1;
  merged.meta.q_max_seq_len = 5;
  merged.meta.kv_max_seq_len = 5;
  merged.attention.host.q_seq_lens = {0, 5};
  merged.attention.host.kv_seq_lens = {0, 5};
  merged.attention.device.q_seq_lens = torch::tensor({0, 5}, ints);
  merged.attention.device.kv_seq_lens = torch::tensor({0, 5}, ints);
  merged.multi_block_tables = {torch::tensor({{0, 1}}, ints)};
  const torch::Tensor prompt_positions = torch::arange(5, ints);
  const auto text_output =
      model->forward(image_tokens, prompt_positions, caches, merged);
  merged.embedding.input_embedding = embeddings;
  const auto image_output =
      model->forward(image_tokens, prompt_positions, caches, merged);
  EXPECT_FALSE(torch::allclose(
      text_output.hidden_states, image_output.hidden_states, 1e-3, 1e-3));
  caches[0].get_swa_cache().copy_(original_cache);

  runtime::Options runtime_options;
  runtime_options.block_size(128);
  mlu::GraphPlannerConfig config;
  config.options = runtime_options;
  config.capabilities = ModelRegistry::get_mtp_capabilities(args.model_type());
  config.state_required = model->requires_graph_forward_metadata();
  config.linear_state_required = has_linear_attention_layers(args);
  config.max_tokens = 64;
  const mlu::MluGraphPlanner planner(config);
  std::unique_ptr<mlu::MluGraph> graph;
  mlu::GraphKey captured_key;
  for (int32_t step = 0; step < 4; ++step) {
    SCOPED_TRACE(step);
    // Warm up, capture, then grow/shrink the request count and remap cache
    // pages.
    const int32_t requests = step % 2 == 0 ? 4 : 3;
    const int32_t begin = step == 0 ? 2 : (step == 1 ? 7 : 128 * step + 1);
    ModelInputParams input;
    input.clear_linear_attention_state();
    input.meta.batch_forward_type = BatchForwardType::DECODE;
    input.meta.num_sequences = requests;
    input.meta.q_max_seq_len = 1;
    input.meta.kv_max_seq_len = begin + 1;
    input.parallel.dp_global_token_nums = {requests};
    input.parallel.dp_is_decode = {1};
    input.attention.host.q_seq_lens.reserve(requests + 1);
    input.attention.host.kv_seq_lens.reserve(requests + 1);
    for (int32_t row = 0; row <= requests; ++row) {
      input.attention.host.q_seq_lens.emplace_back(row);
      input.attention.host.kv_seq_lens.emplace_back(row * (begin + 1));
    }
    input.attention.device.q_seq_lens =
        torch::tensor(input.attention.host.q_seq_lens, ints);
    input.attention.device.kv_seq_lens =
        torch::tensor(input.attention.host.kv_seq_lens, ints);
    // Remap every request onto a different physical ring on each replay.
    input.multi_block_tables = {(torch::arange(requests * 2, ints) + step * 2)
                                    .remainder(8)
                                    .view({requests, 2})};
    torch::Tensor tokens = torch::arange(requests, ints) + step;
    tokens[0].fill_(step % 2 == 0 ? 5 : 126);
    tokens[requests - 1].fill_(127);
    const torch::Tensor positions =
        torch::arange(begin, begin + 1, options.dtype(torch::kInt64))
            .repeat({requests});
    const torch::Tensor cache_before = caches[0].get_swa_cache().clone();
    const ModelOutput eager = model->forward(tokens, positions, caches, input);
    const torch::Tensor expected = eager.hidden_states.clone();
    const torch::Tensor expected_cache = caches[0].get_swa_cache().clone();
    caches[0].get_swa_cache().copy_(cache_before);
    if (step == 0) {
      continue;
    }
    const auto decision = planner.plan({tokens.numel(), input});
    ASSERT_TRUE(std::holds_alternative<mlu::GraphPlan>(decision));
    const auto& plan = std::get<mlu::GraphPlan>(decision);
    ASSERT_GE(plan.layout.padded_num_tokens, tokens.numel());
    ModelOutput output;
    if (graph == nullptr) {
      auto buffers = std::make_unique<mlu::GraphPersistentParam>(
          tokens,
          positions,
          input,
          plan.layout,
          /*main_block_table_columns=*/4,
          config.capabilities.graph_dynamic_multi_block_tables,
          config.capabilities.graph_requires_cache_slots);
      graph = std::make_unique<mlu::MluGraph>(
          std::move(buffers), config.capabilities.enable_piecewise_graph);
      graph->update_input_buffer(
          model.get(), tokens, positions, input, plan.layout);
      output = graph->capture(model.get(),
                              caches,
                              torch_mlu::graph_pool_handle(),
                              torch_mlu::getStreamFromPool(false, 0),
                              runtime_options);
      ASSERT_GT(graph->num_graphs(), 0);
      captured_key = plan.key;
    } else {
      ASSERT_TRUE(captured_key == plan.key);
      graph->update_input_buffer(
          model.get(), tokens, positions, input, plan.layout);
      output = graph->replay();
    }
    torch_mlu::synchronize();
    EXPECT_TRUE(
        torch::allclose(output.hidden_states.narrow(0, 0, tokens.numel()),
                        expected,
                        0.02,
                        0.02));
    EXPECT_TRUE(
        torch::allclose(caches[0].get_swa_cache(), expected_cache, 0.02, 0.02));
  }
}

}  // namespace
}  // namespace xllm
