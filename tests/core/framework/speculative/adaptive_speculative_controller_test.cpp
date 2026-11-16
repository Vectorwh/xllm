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

#include "core/framework/speculative/adaptive_speculative_controller.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>
#include <vector>

#include "core/framework/speculative/speculative_profile_registry.h"
#include "runtime/options.h"

namespace xllm {
namespace {

runtime::Options make_options() {
  runtime::Options options;
  options.enable_adaptive_speculative_decode(true)
      .num_speculative_tokens(4)
      .speculative_algorithm("MTP")
      .enable_graph(false)
      .adaptive_speculative_min_gain(0.0);
  return options;
}

class ScopedPredictor final {
 public:
  ScopedPredictor()
      : previous_(SpeculativeProfileRegistry::get_instance()
                      .validate_time_predictor()) {
    SpeculativeProfileRegistry::ValidateTimePredictor predictor;
    predictor.intercept_ms = 1.0;
    predictor.query_token_ms = 0.5;
    predictor.query_prefix_ms = 0.001;
    SpeculativeProfileRegistry::get_instance().set_validate_time_predictor(
        predictor);
  }
  ~ScopedPredictor() {
    if (previous_) {
      SpeculativeProfileRegistry::get_instance().set_validate_time_predictor(
          *previous_);
    } else {
      SpeculativeProfileRegistry::get_instance()
          .reset_validate_time_predictor();
    }
  }

 private:
  std::optional<SpeculativeProfileRegistry::ValidateTimePredictor> previous_;
};

TEST(AdaptiveSpeculativeControllerTest, SelectsPrunedPrefixesByPathProb) {
  ScopedPredictor predictor;
  AdaptiveSpeculativeController controller(make_options());
  torch::Tensor probs = torch::tensor({
      {0.95, 0.90, 0.20},
      {0.80, 0.00, 0.10},
  });
  std::vector<double> per_seq_kv_lens = {100.0, 100.0};

  const std::vector<int32_t> prefix_lengths =
      controller.select_pruned_prefix_lengths(probs,
                                              /*full_draft_time_ms=*/0.0,
                                              per_seq_kv_lens);

  ASSERT_EQ(prefix_lengths.size(), 2);
  EXPECT_EQ(prefix_lengths, (std::vector<int32_t>{2, 1}));
}

TEST(AdaptiveSpeculativeControllerTest, AllowsZeroPrefixWhenNoGain) {
  ScopedPredictor predictor;
  AdaptiveSpeculativeController controller(make_options());
  torch::Tensor probs = torch::tensor({
      {0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0},
  });
  std::vector<double> per_seq_kv_lens = {100.0, 100.0};

  const std::vector<int32_t> prefix_lengths =
      controller.select_pruned_prefix_lengths(probs,
                                              /*full_draft_time_ms=*/0.0,
                                              per_seq_kv_lens);

  ASSERT_EQ(prefix_lengths.size(), 2);
  EXPECT_EQ(prefix_lengths[0], 0);
  EXPECT_EQ(prefix_lengths[1], 0);
}

TEST(AdaptiveSpeculativeControllerTest, ContinuesAfterNonImprovingCandidate) {
  ScopedPredictor predictor;
  runtime::Options options = make_options();
  options.adaptive_speculative_min_gain(0.09);
  AdaptiveSpeculativeController controller(options);
  torch::Tensor probs = torch::tensor({
      {0.99, 0.98},
      {0.90, 0.00},
  });
  // Serving callers use full_draft_time_ms=0 for deterministic pruning.
  // These costs give about 13% gain for the first prefix, 7% for its second,
  // and 11% for the later cheap row. The 9% threshold must keep scanning.
  std::vector<double> per_seq_kv_lens = {200.0, 1.0};

  const std::vector<int32_t> prefix_lengths =
      controller.select_pruned_prefix_lengths(probs,
                                              /*full_draft_time_ms=*/0.0,
                                              per_seq_kv_lens);

  ASSERT_EQ(prefix_lengths.size(), 2);
  EXPECT_EQ(prefix_lengths, (std::vector<int32_t>{1, 1}));
}

}  // namespace
}  // namespace xllm
