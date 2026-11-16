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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "models/dit/transformers/transformer_qwen_image_21.h"

namespace xllm {
namespace {

QwenImage21TransformerConfig small_transformer_config() {
  QwenImage21TransformerConfig config;
  config.in_channels = 4;
  config.out_channels = 4;
  config.context_in_dim = 8;
  config.num_heads = 2;
  config.head_dim = 8;
  config.num_layers = 2;
  config.axes_dims_rope = {2, 2, 4};
  return config;
}

void initialize_weights(torch::nn::Module& module) {
  torch::NoGradGuard guard;
  torch::manual_seed(21);
  for (auto& tensor : module.parameters()) {
    tensor.normal_(0.0, 0.02);
  }
}

void check_prefix_cache_case(
    const torch::Tensor& image_mask,
    const torch::Tensor& valid_mask,
    const std::vector<qwen_image_21::ImageShape>& shapes) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  QwenImage21Transformer2DModel model(config, &process_group);
  initialize_weights(*model);
  auto layout =
      model->prepare_layout(image_mask, valid_mask, shapes, torch::kCPU);
  auto prompt = torch::randn({1, image_mask.size(1), config.context_in_dim});
  auto full =
      torch::randn({1, layout.image_positions.numel(), config.in_channels});
  qwen_image_21::PrefixKVCache cache(config.num_layers);
  auto timestep = torch::tensor({1.0f});
  auto uncached = model->forward(full, prompt, timestep, layout);
  auto prefill = model->forward(full, prompt, timestep, layout, &cache);
  EXPECT_TRUE(torch::equal(uncached, prefill));
  ASSERT_TRUE(cache.ready());
  const int64_t expected_bytes = config.num_layers * 2 * layout.prefix_tokens *
                                 config.num_heads * config.head_dim *
                                 sizeof(float);
  EXPECT_EQ(cache.bytes(), expected_bytes);
  EXPECT_EQ(cache.decode_layout().allowed_attention.defined(),
            layout.has_padding);
  EXPECT_EQ(cache.decode_layout().rope_cos.size(0), layout.target_tokens);
  auto saved_key = cache.layer(0).key.clone();
  for (int64_t index = 0; index < config.num_layers; ++index) {
    const auto& kv = cache.layer(index);
    EXPECT_TRUE(kv.key.is_contiguous());
    EXPECT_TRUE(kv.value.is_contiguous());
    EXPECT_EQ(kv.key.storage_offset(), 0);
    EXPECT_EQ(kv.key.storage().nbytes(),
              kv.key.numel() * kv.key.element_size());
  }
  for (float time : {0.7f, 0.4f, 0.05f}) {
    auto target = torch::randn({1, layout.target_tokens, config.in_channels});
    full.slice(1, full.size(1) - layout.target_tokens).copy_(target);
    auto t = torch::tensor({time});
    auto expected = model->forward(full, prompt, t, layout);
    auto actual = model->forward(target, prompt, t, layout, &cache);
    EXPECT_TRUE(torch::allclose(actual, expected, 1e-4, 1e-6))
        << "max abs=" << (actual - expected).abs().max().item<float>();
    EXPECT_TRUE(torch::equal(saved_key, cache.layer(0).key));
  }
  qwen_image_21::PrefixKVCache fresh(config.num_layers);
  EXPECT_FALSE(fresh.ready());
  EXPECT_EQ(fresh.bytes(), 0);
}

TaylorSeerOptions taylor_options(int32_t warmup,
                                 int32_t interval,
                                 int32_t order) {
  TaylorSeerOptions result;
  result.warmup_steps = warmup;
  result.skip_interval_steps = interval;
  result.n_derivatives = order;
  return result;
}

TEST(QwenImage21Structure, PrefixCannotSeeTargetNoise) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  qwen_image_21::TransformerBlock block(
      config, &process_group, torch::TensorOptions().dtype(torch::kFloat32));
  initialize_weights(*block);
  auto layout = qwen_image_21::prepare_sequence_layout(
      torch::tensor({{false, true, false}}, torch::kBool),
      {},
      {{1, 2, 2}, {1, 2, 2}},
      {2, 2, 4},
      torch::kCPU);
  auto input = torch::randn({1, 10, 16});
  auto altered = input.clone();
  altered.slice(1, layout.prefix_tokens).normal_(2, 3);
  auto modulation = torch::randn({2, 64});
  const auto block_modulation = qwen_image_21::prepare_block_modulation(
      modulation, layout, config.causal_condition, /*decode=*/false);
  auto a = block->forward(input, block_modulation, layout, /*zero_shift=*/{});
  auto b = block->forward(altered, block_modulation, layout, /*zero_shift=*/{});
  EXPECT_TRUE(torch::equal(a.slice(1, 0, layout.prefix_tokens),
                           b.slice(1, 0, layout.prefix_tokens)));
}

TEST(QwenImage21Structure, MergedSwiGluPreservesGateProjectionOrder) {
  torch::NoGradGuard guard;
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  qwen_image_21::SwiGlu mlp(
      4, 6, &process_group, torch::TensorOptions().dtype(torch::kFloat32));
  auto gate = torch::arange(-12, 12, torch::kFloat32).reshape({6, 4}) / 20.0;
  auto up = torch::arange(24, torch::kFloat32).reshape({6, 4}) / 30.0 + 0.1;
  auto output = torch::arange(24, torch::kFloat32).reshape({4, 6}) / 25.0 - 0.4;
  // Match the unmodified checkpoint, including an independently loaded gate.
  mlp->load_state_dict(
      StateDict({{"proj.weight", up}, {"out.weight", output}}));
  mlp->load_state_dict(StateDict({{"gate_layer.weight", gate}}));
  mlp->verify_loaded_weights("img_mlp.");
  auto input =
      torch::arange(-20, 20, torch::kFloat32).reshape({2, 5, 4}) / 16.0;
  for (const auto& x : {input, input.transpose(0, 1), input[0]}) {
    auto expected = torch::nn::functional::linear(
        torch::silu(torch::nn::functional::linear(x, gate)) *
            torch::nn::functional::linear(x, up),
        output);
    EXPECT_TRUE(torch::allclose(mlp->forward(x), expected, 1e-5, 1e-6));
  }
}

TEST(QwenImage21PrefixCache, TextOnlyMatchesUncachedAcrossSteps) {
  check_prefix_cache_case(torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}});
}

TEST(QwenImage21PrefixCache, AdjacentConditionImagesMatchAcrossSteps) {
  check_prefix_cache_case(
      torch::tensor({{false, true, true, false}}, torch::kBool),
      {},
      {{1, 2, 2}, {1, 2, 2}, {1, 2, 2}});
}

TEST(QwenImage21PrefixCache, PaddingKeysRemainMaskedDuringDecode) {
  check_prefix_cache_case(
      torch::tensor({{false, true, false, false}}, torch::kBool),
      torch::tensor({{1, 1, 1, 0}}, torch::kLong),
      {{1, 2, 2}, {1, 2, 2}});
}

TEST(QwenImage21PrefixCache, CfgBranchesHaveIndependentStorage) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  QwenImage21Transformer2DModel model(config, &process_group);
  initialize_weights(*model);
  auto positive_layout = model->prepare_layout(
      torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto negative_layout = model->prepare_layout(
      torch::zeros({1, 5}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto positive = torch::randn({1, 3, config.context_in_dim});
  auto negative = torch::randn({1, 5, config.context_in_dim});
  auto target = torch::randn({1, 4, config.in_channels});
  qwen_image_21::PrefixKVCache cond(config.num_layers),
      uncond(config.num_layers);
  model->forward(
      target, positive, torch::tensor({1.0f}), positive_layout, &cond);
  auto original = cond.layer(0).key.clone();
  model->forward(
      target, negative, torch::tensor({1.0f}), negative_layout, &uncond);
  EXPECT_TRUE(torch::equal(original, cond.layer(0).key));
  EXPECT_NE(cond.layer(0).key.data_ptr(), uncond.layer(0).key.data_ptr());
  EXPECT_EQ(cond.layer(0).key.size(2), 3);
  EXPECT_EQ(uncond.layer(0).key.size(2), 5);
  auto time = torch::tensor({0.25f});
  auto cached =
      model->forward(target, negative, time, negative_layout, &uncond);
  auto expected = model->forward(target, negative, time, negative_layout);
  EXPECT_TRUE(torch::allclose(cached, expected, 1e-4, 1e-6));
}

TEST(QwenImage21PrefixCache, RejectsUnsafeConfigurationAndBranchReuse) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  QwenImage21Transformer2DModel model(config, &process_group);
  initialize_weights(*model);
  auto layout = model->prepare_layout(
      torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto prompt = torch::randn({1, 3, config.context_in_dim});
  auto target = torch::randn({1, 4, config.in_channels});
  qwen_image_21::PrefixKVCache cache(config.num_layers);
  model->forward(target, prompt, torch::tensor({1.0f}), layout, &cache);
  EXPECT_DEATH(
      model->forward(
          target, prompt.clone(), torch::tensor({0.5f}), layout, &cache),
      "prompt/CFG branch changed");
  for (bool disable_condition : {true, false}) {
    auto unsafe = config;
    unsafe.causal_condition = !disable_condition;
    unsafe.causal_block = disable_condition;
    QwenImage21Transformer2DModel invalid(unsafe, &process_group);
    EXPECT_DEATH(
        invalid->forward(target, prompt, torch::tensor({0.5f}), layout, &cache),
        "requires causal_condition and causal_block");
  }
}

TEST(QwenImage21SpDegree, AllHeadPartitionsRecombinePaddedAttention) {
  torch::manual_seed(21);
  auto full = qwen_image_21::prepare_sequence_layout(
      torch::zeros({1, 3}, torch::kBool),
      torch::tensor({{1, 1, 0}}, torch::kLong),
      {{1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  const int64_t total = full.prefix_tokens + full.target_tokens;
  auto query = torch::randn({1, 32, total, 8});
  auto key = torch::randn_like(query);
  auto value = torch::randn_like(query);
  auto expected = qwen_image_21::transformer_attention(
      query, key, value, full.allowed_attention);
  for (int32_t degree : {2, 32}) {
    qwen_image_21::SpSequenceLayout geometry(full, 0, degree);
    const int64_t padding = geometry.padding();
    auto pad = [&](const torch::Tensor& input) {
      return torch::cat(
          {input, torch::zeros({1, 32, padding, 8}, input.options())}, 2);
    };
    auto q = pad(query), k = pad(key), v = pad(value);
    std::vector<torch::Tensor> parts, decode_parts;
    parts.reserve(degree);
    decode_parts.reserve(degree);
    const int64_t heads = 32 / degree;
    for (int32_t rank = 0; rank < degree; ++rank) {
      auto hq = q.slice(1, rank * heads, (rank + 1) * heads);
      auto hk = k.slice(1, rank * heads, (rank + 1) * heads);
      auto hv = v.slice(1, rank * heads, (rank + 1) * heads);
      parts.emplace_back(qwen_image_21::transformer_attention(
          hq, hk, hv, geometry.attention_mask(false)));
      decode_parts.emplace_back(
          qwen_image_21::transformer_attention(hq.slice(2, full.prefix_tokens),
                                               hk,
                                               hv,
                                               geometry.attention_mask(true)));
    }
    EXPECT_TRUE(torch::allclose(
        torch::cat(parts, 1).slice(2, 0, total), expected, 1e-6, 1e-6));
    EXPECT_TRUE(torch::allclose(torch::cat(decode_parts, 1).slice(2, 0, 12),
                                expected.slice(2, full.prefix_tokens),
                                1e-6,
                                1e-6));
  }
}

TEST(QwenImage21TaylorSeer,
     NoHitTransformerPathMatchesBaselineWithAndWithoutKv) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  QwenImage21Transformer2DModel model(config, &process_group);
  initialize_weights(*model);
  auto layout =
      model->prepare_layout(torch::tensor({{false, true, false}}, torch::kBool),
                            {},
                            {{1, 2, 2}, {1, 2, 2}},
                            torch::kCPU);
  auto prompt = torch::randn({1, 3, config.context_in_dim});
  auto full = torch::randn({1, 8, config.in_channels});
  for (bool use_kv : {false, true}) {
    qwen_image_21::PrefixKVCache kv(config.num_layers);
    qwen_image_21::RequestTaylorSeer ts(taylor_options(0, 1, 1), 4);
    for (int64_t step = 1; step <= 4; ++step) {
      auto t = torch::tensor({1.0f / step});
      auto expected = model->forward(full, prompt, t, layout);
      auto input = use_kv && kv.ready() ? full.slice(1, 4) : full;
      auto actual = model->forward(
          input, prompt, t, layout, use_kv ? &kv : nullptr, nullptr, &ts, step);
      EXPECT_TRUE(torch::allclose(actual, expected, 1e-5, 1e-6));
    }
    EXPECT_EQ(ts.hit_steps(), 0);
  }
}

TEST(QwenImage21TaylorSeer, TransformerHitsRemainFiniteAndPreservePrefixKv) {
  torch::NoGradGuard guard;
  auto config = small_transformer_config();
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  QwenImage21Transformer2DModel model(config, &process_group);
  initialize_weights(*model);
  auto layout = model->prepare_layout(
      torch::zeros({1, 3}, torch::kBool), {}, {{1, 2, 2}}, torch::kCPU);
  auto prompt = torch::randn({1, 3, config.context_in_dim});
  auto target = torch::randn({1, 4, config.in_channels});
  qwen_image_21::PrefixKVCache kv(config.num_layers);
  qwen_image_21::RequestTaylorSeer ts(taylor_options(2, 2, 1), 6);
  torch::Tensor saved;
  for (int64_t step = 1; step <= 6; ++step) {
    auto out = model->forward(target,
                              prompt,
                              torch::tensor({1.0f / step}),
                              layout,
                              &kv,
                              nullptr,
                              &ts,
                              step);
    EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
    EXPECT_EQ(out.sizes(), target.sizes());
    if (step == 1) {
      saved = kv.layer(0).key.clone();
    }
    EXPECT_TRUE(torch::equal(saved, kv.layer(0).key));
  }
  EXPECT_EQ(ts.full_steps(), 4);
  EXPECT_EQ(ts.hit_steps(), 2);
}

TEST(QwenImage21Native, AdaptiveNormSupportsTokenwiseAndBroadcastScales) {
  torch::NoGradGuard guard;
  constexpr double kEps = 1e-5;
  layer::AdaLayerNorm norm(/*dim=*/2,
                           kEps,
                           /*elementwise_affine=*/false,
                           torch::TensorOptions().dtype(torch::kFloat32));
  auto hidden = torch::tensor({{{-1.0f, 1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}}});
  auto tokenwise =
      torch::tensor({{{-0.25f, -0.5f}, {0.25f, 0.5f}, {-0.25f, -0.5f}}});
  auto broadcast = torch::tensor({{{0.25f, 0.5f}}});
  auto expected_norm = hidden / std::sqrt(1.0 + kEps);
  auto actual = qwen_image_21::apply_adaptive_layer_norm(
      norm, hidden, tokenwise, /*zero_shift=*/{});
  EXPECT_TRUE(torch::allclose(actual, expected_norm * (1 + tokenwise)));
  actual = qwen_image_21::apply_adaptive_layer_norm(
      norm, hidden, broadcast, /*zero_shift=*/{});
  EXPECT_TRUE(torch::allclose(
      actual, expected_norm * torch::tensor({{{1.25f, 1.5f}}})));
}

}  // namespace
}  // namespace xllm
