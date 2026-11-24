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
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraph.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "layers/mlu/moe_gate.h"

namespace xllm::layer {
namespace {

std::tuple<torch::Tensor, torch::Tensor> reference_route(
    const torch::Tensor& logits,
    const torch::Tensor& bias,
    int64_t topk) {
  torch::Tensor scores = torch::softplus(logits.to(torch::kFloat32)).sqrt();
  torch::Tensor ids = std::get<1>((scores + bias).topk(topk, /*dim=*/-1));
  torch::Tensor weights = scores.gather(-1, ids);
  weights = weights / (weights.sum(-1, /*keepdim=*/true) + 1e-20) * 1.5;
  return {weights, ids.to(torch::kInt32)};
}

TEST(MoeRouterTest, RoutesAcrossEveryDispatchBoundaryAndExtremeLogit) {
  const torch::Device device(torch::kPrivateUse1, 0);
  for (int64_t experts : {128, 384}) {
    const int64_t topk = experts == 128 ? 3 : 6;
    ModelArgs args;
    args.hidden_size() = 5120;
    args.n_routed_experts() = experts;
    args.num_experts_per_tok() = topk;
    args.scoring_func() = "sqrtsoftplus";
    args.routed_scaling_factor() = 1.5;
    args.vocab_size() = 8;
    const bool visual_router = experts == 384;
    if (visual_router) {
      args.model_type() = "deepseek_v41";
      args.mm_num_hidden_layers() = 32;
      args.image_token_id() = 7;
    }
    const auto options =
        torch::TensorOptions().device(device).dtype(torch::kBFloat16);
    MoEGate gate(args, QuantArgs(), options);
    torch::Tensor weight = torch::zeros({experts, 5120}, torch::kBFloat16);
    weight.select(1, 0).copy_(1 + torch::arange(experts, torch::kFloat32) / 64);
    weight.select(1, 5119).copy_(torch::arange(experts, torch::kFloat32) / 128);
    torch::Tensor bias = torch::arange(experts, torch::kFloat32) * 0.03;
    // Checkpoint shards may deliver bias and projection in separate calls.
    gate->load_state_dict(StateDict({{"bias", bias}}));
    torch::Tensor visual_bias = bias.roll({experts / 2}, {0});
    if (visual_router) {
      gate->load_state_dict(StateDict({{"bias_vl", visual_bias}}));
    }
    gate->load_state_dict(StateDict({{"weight", weight}}));
    for (int64_t rows : {0,
                         1,
                         47,
                         48,
                         49,
                         199,
                         200,
                         201,
                         449,
                         450,
                         451,
                         949,
                         950,
                         951,
                         1024,
                         1281,
                         // Token-split tile and pass boundaries: a wide final
                         // tile (65 and 96 tokens per core), two tiles (97),
                         // and uneven multi-pass splits.
                         2049,
                         3072,
                         3073,
                         // Few-expert resident path: multi-chunk, 1-pass and
                         // multi-pass wave boundaries (including +1 edges).
                         3584,
                         3585,
                         4000,
                         4001,
                         4096,
                         4097,
                         4129,
                         6145,
                         8192,
                         8193,
                         9000,
                         16384,
                         16385}) {
      SCOPED_TRACE(::testing::Message()
                   << "experts=" << experts << " rows=" << rows);
      torch::Tensor hidden = torch::zeros({rows, 5120}, torch::kBFloat16);
      torch::Tensor values = torch::tensor(
          {-1000.f, -50.f, -20.f, -11.f, 0.f, 10.f, 100.f, 1000.f});
      hidden.select(1, 0).copy_(values.index_select(
          0, torch::arange(rows, torch::kLong).remainder(8)));
      hidden.select(1, 5119).fill_(0.5);
      torch::Tensor logits =
          (hidden.select(1, 0).to(torch::kFloat32).unsqueeze(1) *
               weight.select(1, 0).to(torch::kFloat32).unsqueeze(0) +
           0.5 * weight.select(1, 5119).to(torch::kFloat32).unsqueeze(0))
              .to(torch::kBFloat16);
      torch::Tensor tokens = torch::arange(rows, torch::kInt32).remainder(11);
      torch::Tensor row_bias =
          visual_router
              ? torch::where(tokens.unsqueeze(1).eq(7), visual_bias, bias)
              : bias;
      auto [expected_weights, expected_ids] =
          reference_route(logits, row_bias, topk);
      torch::Tensor input = hidden.to(device);
      auto [weights, ids] = gate->forward(input, tokens.to(device));
      EXPECT_EQ(weights.scalar_type(), torch::kFloat32);
      EXPECT_EQ(ids.scalar_type(), torch::kInt32);
      EXPECT_TRUE(torch::equal(ids.cpu(), expected_ids));
      EXPECT_TRUE(torch::allclose(weights.cpu(), expected_weights, 2e-5, 2e-6));
    }
  }
}

TEST(MoeRouterTest, DenseGateMatchesReferenceAndGraphUsesUpdatedInput) {
  torch::manual_seed(20261117);
  const torch::Device device(torch::kPrivateUse1, 0);
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  for (int64_t rows : {16, 128, 512, 1025}) {
    SCOPED_TRACE(rows);
    torch::Tensor input = (torch::randn({rows, 5120}) * 0.1).to(options);
    torch::Tensor raw_weight = (torch::randn({384, 5120}) * 0.02).to(options);
    torch::Tensor bias =
        (torch::arange(384, torch::kFloat32) * 0.02).to(device);
    MOESoftPlusTopK router(384, 6, 1.5f, 8, false, options);
    router->load_state_dict(StateDict({{"bias", bias}}));
    router->prepare_fused_router(raw_weight);
    auto result = router->forward_fused_router(input);
    ASSERT_TRUE(result.has_value());
    torch_mlu::synchronize();
    torch_mlu::MLUGraph graph;
    {
      torch_mlu::mlu::MLUStreamGuard stream(
          torch_mlu::getStreamFromPool(false, 0));
      graph.capture_begin();
      result = router->forward_fused_router(input);
      graph.capture_end();
    }
    for (int32_t step = 0; step < 3; ++step) {
      input.mul_(-0.75);
      graph.replay();
      auto [weights, ids] = result.value();
      torch::Tensor logits =
          torch::matmul(input, raw_weight.transpose(0, 1)).cpu();
      auto [expected_weights, expected_ids] =
          reference_route(logits, bias.cpu(), 6);
      EXPECT_TRUE(torch::equal(ids.cpu(), expected_ids));
      EXPECT_TRUE(torch::allclose(weights.cpu(), expected_weights, 1e-3, 1e-5));
    }
  }
}

TEST(MoeRouterTest, HashAndUnsupportedLayoutsKeepTheirScoringSemantics) {
  const auto options = torch::TensorOptions()
                           .device(torch::kPrivateUse1, 0)
                           .dtype(torch::kBFloat16);
  MOESoftPlusTopK hash_router(128, 3, 1.5f, 2, true, options);
  torch::Tensor mapping = torch::tensor({{9, 1, 7}, {3, 6, 2}}, torch::kInt32);
  hash_router->load_state_dict(StateDict({{"tid2eid", mapping}}));
  hash_router->prepare_fused_router(torch::zeros({128, 5120}, options));
  EXPECT_FALSE(
      hash_router->forward_fused_router(torch::zeros({2, 5120}, options))
          .has_value());
  auto [weights, ids] =
      hash_router->forward(torch::full({2, 128}, -20, options),
                           torch::tensor({1, 0}, options.dtype(torch::kInt32)));
  EXPECT_TRUE(torch::equal(ids.cpu(), mapping.flip({0})));
  EXPECT_TRUE(torch::allclose(weights.cpu(), torch::full({2, 3}, 0.5)));
  MOESoftPlusTopK other_router(128, 3, 1.5f, 2, false, options);
  other_router->prepare_fused_router(torch::zeros({128, 64}, options));
  EXPECT_FALSE(
      other_router->forward_fused_router(torch::zeros({2, 64}, options))
          .has_value());
}

TEST(MoeRouterTest, VisualGateUsesCurrentTokenIdsOnGraphReplay) {
  const auto options = torch::TensorOptions()
                           .device(torch::kPrivateUse1, 0)
                           .dtype(torch::kBFloat16);
  constexpr int64_t kImageToken = 129264;
  MOESoftPlusTopK router(128,
                         3,
                         1.5f,
                         8,
                         false,
                         options,
                         /*use_visual_bias=*/true,
                         kImageToken);
  torch::Tensor bias = torch::arange(128, torch::kFloat32);
  router->load_state_dict(StateDict({{"bias", -bias}, {"bias_vl", bias}}));
  torch::Tensor weight = torch::zeros({128, 5120}, options);
  weight.select(1, 0).copy_(bias.to(options) / 8);
  router->prepare_fused_router(weight);
  torch::Tensor input = torch::ones({4, 5120}, options);
  torch::Tensor logits = weight.select(1, 0).expand({4, 128}).contiguous();
  auto text = router->forward_fused_router(input);
  ASSERT_TRUE(text.has_value());
  auto [text_weights, text_ids] = router->forward(logits);
  EXPECT_TRUE(torch::equal(std::get<1>(*text), text_ids));
  EXPECT_TRUE(torch::allclose(std::get<0>(*text), text_weights, 1e-4, 1e-6));

  // The second ID has the same low 32 bits as the image ID.
  torch::Tensor initial_ids = torch::tensor(
      {kImageToken, kImageToken + (int64_t{1} << 32), int64_t{0}, kImageToken},
      options.dtype(torch::kInt64));
  torch::Tensor ids = initial_ids.clone();
  auto fused = router->forward_fused_router(input, ids);
  ASSERT_TRUE(fused.has_value());
  torch_mlu::synchronize();
  torch_mlu::MLUGraph graph;
  {
    torch_mlu::mlu::MLUStreamGuard stream(
        torch_mlu::getStreamFromPool(false, 0));
    graph.capture_begin();
    fused = router->forward_fused_router(input, ids);
    graph.capture_end();
  }
  for (int32_t shift : {0, 1}) {
    ids.copy_(initial_ids.roll({shift}, {0}));
    graph.replay();
    auto [expected_weights, expected_ids] = router->forward(logits, ids);
    EXPECT_TRUE(torch::equal(std::get<1>(*fused), expected_ids));
    EXPECT_TRUE(
        torch::allclose(std::get<0>(*fused), expected_weights, 1e-4, 1e-6));
  }
}

}  // namespace
}  // namespace xllm::layer
