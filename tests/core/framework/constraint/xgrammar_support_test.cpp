/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/constraint/xgrammar_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace xllm {
namespace {

bool contains(const std::vector<std::string>& names, const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

TEST(XGrammarSupportTest, PlainEagerDecodeIsSupported) {
  EXPECT_TRUE(xgrammar_unsupported_features(XGrammarSupportInputs{}).empty());
}

TEST(XGrammarSupportTest, ChannelMigratedSpeculativeAlgorithmsAreSupported) {
  for (const char* algorithm : {"MTP", "dflash", "dflash2", "DSpark", "mtp"}) {
    XGrammarSupportInputs inputs;
    inputs.num_speculative_tokens = 3;
    inputs.speculative_algorithm = algorithm;
    EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty()) << algorithm;
  }
}

TEST(XGrammarSupportTest, SuffixStyleSpeculativeStaysGated) {
  // SuffixWorkerImpl never reads json_object_states, so suffix-style
  // proposals would run unconstrained; keep them gated even though the
  // channel-migrated algorithms are unlocked.
  XGrammarSupportInputs inputs;
  inputs.num_speculative_tokens = 3;
  inputs.speculative_algorithm = "suffix";
  const auto unsupported = xgrammar_unsupported_features(inputs);
  ASSERT_EQ(unsupported.size(), 1u);
  EXPECT_TRUE(contains(unsupported, "speculative_suffix"));
}

TEST(XGrammarSupportTest, ZeroSpeculativeTokensNeverGates) {
  XGrammarSupportInputs inputs;
  inputs.num_speculative_tokens = 0;
  inputs.speculative_algorithm = "suffix";
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());
}

TEST(XGrammarSupportTest, GraphStaysGated) {
  XGrammarSupportInputs inputs;
  inputs.enable_graph = true;
  const auto unsupported = xgrammar_unsupported_features(inputs);
  ASSERT_EQ(unsupported.size(), 1u);
  EXPECT_TRUE(contains(unsupported, "graph"));
}

TEST(XGrammarSupportTest, GraphGatesEvenWithSpeculativeUnlocked) {
  // A channel-migrated speculative algorithm is supported, but enabling graph
  // must still gate: the draft-side graph is safe, the target-side is not.
  XGrammarSupportInputs inputs;
  inputs.num_speculative_tokens = 3;
  inputs.speculative_algorithm = "MTP";
  inputs.enable_graph = true;
  const auto unsupported = xgrammar_unsupported_features(inputs);
  ASSERT_EQ(unsupported.size(), 1u);
  EXPECT_TRUE(contains(unsupported, "graph"));
}

TEST(XGrammarSupportTest, PiecewiseGraphStaysGated) {
  XGrammarSupportInputs inputs;
  inputs.enable_prefill_piecewise_graph = true;
  EXPECT_TRUE(contains(xgrammar_unsupported_features(inputs),
                       "prefill_piecewise_graph"));
}

TEST(XGrammarSupportTest, DisaggPdStaysGated) {
  XGrammarSupportInputs inputs;
  inputs.enable_disagg_pd = true;
  EXPECT_TRUE(contains(xgrammar_unsupported_features(inputs), "disagg_pd"));
}

TEST(XGrammarSupportTest, ServiceRoutingStaysGated) {
  XGrammarSupportInputs inputs;
  inputs.enable_service_routing = true;
  EXPECT_TRUE(
      contains(xgrammar_unsupported_features(inputs), "service_routing"));
}

TEST(XGrammarSupportTest, PrefixCacheStaysGated) {
  XGrammarSupportInputs inputs;
  inputs.enable_prefix_cache = true;
  EXPECT_TRUE(contains(xgrammar_unsupported_features(inputs), "prefix_cache"));
}

TEST(XGrammarSupportTest, OnlyTheGatedFeaturesAreReported) {
  // Speculative decoding and schedule overlap are not inputs (they never
  // gate), so enabling graph and PD surfaces exactly those two.
  XGrammarSupportInputs inputs;
  inputs.enable_graph = true;
  inputs.enable_disagg_pd = true;
  const auto unsupported = xgrammar_unsupported_features(inputs);
  EXPECT_EQ(unsupported.size(), 2u);
  EXPECT_TRUE(contains(unsupported, "graph"));
  EXPECT_TRUE(contains(unsupported, "disagg_pd"));
}

TEST(XGrammarSupportTest, ChunkedPrefillStaysSupportedWithSpeculation) {
  // chunked_prefill is deliberately NOT gated: it only splits the prompt, and
  // the matcher initializes at the first decode token, so the chunk boundary
  // cannot affect the decode-time grammar. It is also a hard requirement of
  // the block-diffusion family, so gating it would make the unlocked
  // speculative algorithms unstartable. Pin that decision here: the previous
  // startup CHECK rejected this combination.
  XGrammarSupportInputs inputs;
  inputs.num_speculative_tokens = 4;
  inputs.speculative_algorithm = "MTP";
  EXPECT_TRUE(xgrammar_unsupported_features(inputs).empty());

  // The draft families the channel migrated must stay accepted too, including
  // DSpark whose block-diffusion path requires chunked prefill.
  for (const char* algorithm : {"DFlash", "DFlash2", "DSpark"}) {
    XGrammarSupportInputs draft;
    draft.num_speculative_tokens = 4;
    draft.speculative_algorithm = algorithm;
    SCOPED_TRACE(algorithm);
    EXPECT_TRUE(xgrammar_unsupported_features(draft).empty());
  }
}

TEST(XGrammarSupportTest, AllGatedFeaturesAreListedTogether) {
  XGrammarSupportInputs inputs;
  inputs.enable_graph = true;
  inputs.enable_prefill_piecewise_graph = true;
  inputs.enable_disagg_pd = true;
  inputs.enable_service_routing = true;
  inputs.enable_prefix_cache = true;
  // Five gated features (chunked_prefill is supported); speculative and
  // overlap are not among them.
  EXPECT_EQ(xgrammar_unsupported_features(inputs).size(), 5u);
}

}  // namespace
}  // namespace xllm
