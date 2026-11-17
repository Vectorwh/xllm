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

#include "layers/mlu/moe_gate.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

TEST(MoEGateWeightLoadingTest, RejectsMissingRoutingBias) {
  torch::InferenceMode inference;
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  MOESoftPlusTopK text_gate(/*n_routed_experts=*/4,
                            /*n_activated_experts=*/2,
                            /*route_scale=*/1.0f,
                            /*vocab_size=*/8,
                            /*use_hash=*/false,
                            options);
  EXPECT_DEATH(text_gate->verify_loaded_weights(),
               "sqrtsoftplus routing requires bias");
  MOESoftPlusTopK visual_gate(/*n_routed_experts=*/4,
                              /*n_activated_experts=*/2,
                              /*route_scale=*/1.0f,
                              /*vocab_size=*/8,
                              /*use_hash=*/false,
                              options,
                              /*use_visual_bias=*/true,
                              /*image_token_id=*/7);
  visual_gate->load_state_dict(StateDict({{"bias", torch::zeros({4})}}));
  EXPECT_DEATH(visual_gate->verify_loaded_weights(),
               "Missing visual router bias_vl");
  visual_gate->load_state_dict(StateDict({{"bias_vl", torch::zeros({4})}}));
  visual_gate->verify_loaded_weights();
  text_gate->load_state_dict(StateDict({{"bias", torch::zeros({4})}}));
  text_gate->verify_loaded_weights();
  MOESoftPlusTopK hash_gate(/*n_routed_experts=*/4,
                            /*n_activated_experts=*/2,
                            /*route_scale=*/1.0f,
                            /*vocab_size=*/8,
                            /*use_hash=*/true,
                            options);
  hash_gate->load_state_dict(
      StateDict({{"tid2eid", torch::zeros({8, 2}, torch::kInt32)}}));
  hash_gate->verify_loaded_weights();
}

class MoEGateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "MLU device is required";
    }
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
  }

  // Set MoE gate related fields on model_args_. Call before constructing
  // MoEGateImpl in each test.
  void set_moe_gate_params(int64_t num_experts,
                           int64_t top_k,
                           int64_t num_expert_group,
                           int64_t topk_group,
                           double route_scale,
                           int64_t hidden_size,
                           bool renormalize,
                           const std::string& scoring_func,
                           const std::string& topk_method) {
    model_args_.n_routed_experts() = static_cast<int32_t>(num_experts);
    model_args_.num_experts_per_tok() = static_cast<int32_t>(top_k);
    model_args_.n_group() = static_cast<int32_t>(num_expert_group);
    model_args_.topk_group() = static_cast<int32_t>(topk_group);
    model_args_.routed_scaling_factor() = static_cast<float>(route_scale);
    model_args_.hidden_size() = hidden_size;
    model_args_.norm_topk_prob() = renormalize;
    model_args_.scoring_func() = scoring_func;
    model_args_.topk_method() = topk_method;
  }

  void verify_routes(const std::string& scoring,
                     bool corrected,
                     bool grouped) const {
    constexpr int64_t kExperts = 8;
    constexpr int64_t kHidden = 128;
    constexpr int64_t kTopk = 2;
    constexpr float kScale = 2.5f;
    ModelArgs args;
    args.n_routed_experts() = kExperts;
    args.num_experts_per_tok() = kTopk;
    args.hidden_size() = kHidden;
    args.n_group() = grouped ? 2 : 1;
    args.topk_group() = 1;
    args.norm_topk_prob() = true;
    args.routed_scaling_factor() = kScale;
    args.scoring_func() = scoring;
    args.topk_method() = corrected ? "noaux_tc" : "";
    auto weights = torch::zeros({kExperts, kHidden}, torch::kBFloat16);
    weights.select(1, 0).copy_(
        torch::tensor({-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f}));
    auto hidden = torch::zeros({4, kHidden}, torch::kBFloat16);
    hidden.select(1, 0).copy_(torch::tensor({1.0f, 0.5f, -1.0f, -0.5f}));
    const auto bias = torch::tensor(
        {0.0f, 0.25f, 0.0f, 0.0f, 0.3f, 0.0f, 0.0f, -0.3f}, torch::kBFloat16);
    std::unordered_map<std::string, torch::Tensor> state;
    state["weight"] = weights.to(options_.device());
    if (corrected) {
      state["e_score_correction_bias"] = bias.to(options_.device());
    }
    MoEGateImpl gate(args, QuantArgs(), options_);
    gate.load_state_dict(StateDict(state));
    auto input = hidden.to(options_.device());
    auto [actual_weights, actual_ids] = gate.forward(input);
    Device(options_.device()).synchronize_default_stream();

    const auto logits = torch::matmul(hidden.to(torch::kFloat32),
                                      weights.to(torch::kFloat32).t());
    const auto scores = scoring == "softmax" ? torch::softmax(logits, -1)
                                             : torch::sigmoid(logits);
    auto selection = corrected ? scores + bias.to(torch::kFloat32) : scores;
    if (grouped) {
      const auto group_scores = std::get<0>(selection.view({4, 2, 4}).max(-1));
      const auto group_ids = std::get<1>(group_scores.max(-1));
      const auto groups = torch::arange(kExperts, torch::kInt64) / 4;
      selection = selection.masked_fill(
          groups.to(torch::kInt64) != group_ids.unsqueeze(1),
          -std::numeric_limits<float>::infinity());
    }
    const auto expected_ids = std::get<1>(selection.topk(kTopk, -1));
    auto expected_weights = scores.gather(1, expected_ids);
    expected_weights /= expected_weights.sum(-1, true);
    expected_weights *= kScale;
    EXPECT_TRUE(torch::equal(actual_ids.cpu().to(torch::kLong), expected_ids));
    EXPECT_TRUE(
        torch::allclose(actual_weights.cpu(), expected_weights, 1e-3, 1e-4));
  }

  torch::TensorOptions options_;
  ModelArgs model_args_;
  QuantArgs quant_args_;
};

TEST_F(MoEGateTest, Sigmoid) {
  verify_routes("sigmoid", /*corrected=*/true, /*grouped=*/false);
}
TEST_F(MoEGateTest, Softmax) {
  verify_routes("softmax", /*corrected=*/false, /*grouped=*/false);
}
TEST_F(MoEGateTest, SigmoidTopkGroup1) {
  verify_routes("sigmoid", /*corrected=*/false, /*grouped=*/true);
}

TEST_F(MoEGateTest, SqrtSoftplusKeepsNegativeRoutingScores) {
  constexpr int64_t kExperts = 256;
  constexpr int64_t kTopk = 8;
  constexpr float kRouteScale = 1.5f;
  MOESoftPlusTopK gate(kExperts, kTopk, kRouteScale, 1, false, options_);
  torch::Tensor bias = torch::arange(kExperts, torch::kFloat32) * 1000;
  gate->load_state_dict(StateDict({{"bias", bias}}));
  torch::Tensor scores = torch::empty({4, kExperts}, torch::kFloat32);
  scores[0].copy_(-20 + torch::arange(kExperts, torch::kFloat32) / 64);
  scores[1].copy_(-50 + torch::arange(kExperts, torch::kFloat32) / 64);
  scores[2].fill_(-1000);
  scores[3].copy_(1000 + torch::arange(kExperts, torch::kFloat32));
  scores = scores.to(torch::kBFloat16);
  auto [weights, indices] = gate->forward(scores.to(options_.device()));
  weights = weights.to(torch::kCPU);
  indices = indices.to(torch::kCPU);
  ASSERT_TRUE(torch::isfinite(weights).all().item<bool>());
  for (int64_t row = 0; row < scores.size(0); ++row) {
    std::vector<double> expected;
    expected.reserve(kTopk);
    double sum = 1e-20;
    for (int64_t col = 0; col < kTopk; ++col) {
      const int64_t expert = kExperts - col - 1;
      EXPECT_EQ(indices[row][col].item<int32_t>(), expert);
      const double logit = scores[row][expert].item<double>();
      const double score = std::sqrt(std::max(logit, 0.0) +
                                     std::log1p(std::exp(-std::abs(logit))));
      expected.emplace_back(score);
      sum += score;
    }
    for (int64_t col = 0; col < kTopk; ++col) {
      EXPECT_NEAR(weights[row][col].item<float>(),
                  expected[col] / sum * kRouteScale,
                  1e-6);
    }
  }
}

TEST_F(MoEGateTest, SqrtSoftplusHashRoutingPreservesTokenExpertMapping) {
  constexpr int64_t kExperts = 256;
  constexpr int64_t kTopk = 8;
  constexpr int64_t kVocab = 3;
  MOESoftPlusTopK gate(kExperts, kTopk, 1.5f, kVocab, true, options_);
  torch::Tensor mapping =
      torch::arange(kVocab * kTopk, torch::kInt32).view({kVocab, kTopk});
  gate->load_state_dict(StateDict({{"tid2eid", mapping}}));
  torch::Tensor tokens = torch::tensor({2, 0, 1}, torch::kInt32);
  torch::Tensor scores = torch::full({3, kExperts}, -20, options_);
  auto [weights, indices] = gate->forward(scores, tokens.to(options_.device()));
  EXPECT_TRUE(torch::equal(indices.to(torch::kCPU),
                           mapping.index_select(0, tokens.to(torch::kLong))));
  EXPECT_TRUE(torch::allclose(weights.to(torch::kCPU),
                              torch::full({3, kTopk}, 1.5f / kTopk),
                              1e-6,
                              1e-6));
}

TEST_F(MoEGateTest, VisualBiasRespectsCapabilityAndTokenType) {
  torch::InferenceMode inference;
  set_moe_gate_params(4, 2, 1, 1, 1.5, 4, true, "sqrtsoftplus", "noaux_tc");
  for (const std::string& model_type :
       {"deepseek_v41", "deepseek_v41_dspark"}) {
    SCOPED_TRACE(model_type);
    for (const int32_t vision_layers : {0, 1}) {
      SCOPED_TRACE(vision_layers);
      model_args_.model_type(model_type)
          .image_token_id(129264)
          .mm_num_hidden_layers(vision_layers);
      MoEGate gate(model_args_, quant_args_, options_);
      gate->load_state_dict(StateDict(
          {{"weight", torch::eye(4, torch::kBFloat16)},
           {"bias", torch::tensor({20.0f, 10.0f, 0.0f, -10.0f})},
           {"bias_vl", torch::tensor({-10.0f, 0.0f, 10.0f, 20.0f})}}));
      gate->verify_loaded_weights();
      torch::Tensor logits = torch::tensor({{-4.0f, -1.0f, 1.0f, 3.0f},
                                            {-4.0f, -1.0f, 1.0f, 3.0f},
                                            {-4.0f, -1.0f, 1.0f, 3.0f}},
                                           options_);
      torch::Tensor tokens =
          torch::tensor({7, 129264, 129265}, options_.dtype(torch::kInt32));
      auto [weights, indices] = gate->forward(logits, tokens);
      torch::Tensor expected_ids =
          torch::tensor({{0, 1},
                         {vision_layers > 0 ? 3 : 0, vision_layers > 0 ? 2 : 1},
                         {0, 1}},
                        torch::kInt64);
      EXPECT_TRUE(torch::equal(indices.cpu().to(torch::kLong), expected_ids));
      torch::Tensor scores =
          torch::softplus(logits.cpu().to(torch::kFloat32)).sqrt();
      torch::Tensor expected = scores.gather(1, expected_ids);
      expected = expected / expected.sum(-1, true) * 1.5;
      EXPECT_TRUE(torch::allclose(weights.cpu(), expected, 1e-6, 1e-6));
    }
  }
}

}  // namespace layer
}  // namespace xllm
