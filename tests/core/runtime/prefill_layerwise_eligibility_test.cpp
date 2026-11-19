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

#include "runtime/prefill_layerwise_eligibility.h"

#include <gtest/gtest.h>

#include "runtime/options.h"

namespace xllm {
namespace {

PrefillLayerwiseEligibility baseline_eligibility() {
  PrefillLayerwiseEligibility eligibility;
  eligibility.npu = true;
  eligibility.torch_kernel_backend = true;
  eligibility.llm_backend = true;
  eligibility.generate_task = true;
  eligibility.pd_prefill_only = true;
  eligibility.push_transfer = true;
  eligibility.kv_cache_dtype = "auto";
  eligibility.model_type = "glm_moe_dsa";
  return eligibility;
}

TEST(PrefillLayerwiseEligibilityTest, AdmitsBaselineEagerPrefill) {
  EXPECT_TRUE(baseline_eligibility().supported());
}

// CP>1 is admitted as long as the KV layout stays replicated: the layerwise
// split must stay 1 and the effective kv split must stay 1. A raw
// kv_split_size=0 inherits cp_size, so CP=4 exposes effective=4 and selects
// the unsupported sharded path.
TEST(PrefillLayerwiseEligibilityTest, AdmitsContextParallelWithReplicatedKv) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  for (int32_t cp_size : {1, 2, 4}) {
    eligibility.cp_size = cp_size;
    eligibility.layerwise_split_size = 1;
    eligibility.kv_split_size_effective = 1;
    EXPECT_TRUE(eligibility.supported());
  }

  eligibility.cp_size = 4;
  eligibility.layerwise_split_size = 2;
  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("layerwise_split_size=1"),
            std::string::npos);

  eligibility.layerwise_split_size = 1;
  eligibility.kv_split_size_effective = 2;
  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("effective kv_split_size<=1"),
            std::string::npos);

  eligibility.kv_split_size_effective = 4;
  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("effective kv_split_size<=1"),
            std::string::npos);

  eligibility.kv_split_size_effective = 1;
  eligibility.cp_size = 0;
  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("CP >= 1"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsAtbKernelBackend) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.torch_kernel_backend = false;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("TORCH"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, AdmitsDflash2TargetWithGraph) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.enable_graph = true;
  eligibility.has_speculative_decode = true;
  eligibility.dflash2_target = true;

  EXPECT_TRUE(eligibility.supported());
}

TEST(PrefillLayerwiseEligibilityTest, RejectsDraftEngine) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.has_speculative_decode = true;
  eligibility.dflash2_target = true;
  eligibility.is_draft_engine = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("draft engine"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsOtherSpeculativeAlgorithms) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.has_speculative_decode = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("DFlash2 target"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsGraphOutsideDflash2Target) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.enable_graph = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("eager execution"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsGraphWithoutActiveSpeculation) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.enable_graph = true;
  eligibility.dflash2_target = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("eager execution"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsPiecewisePrefillGraph) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.enable_graph = true;
  eligibility.has_speculative_decode = true;
  eligibility.dflash2_target = true;
  eligibility.enable_prefill_piecewise_graph = true;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("piecewise"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsDecodeCapablePdInstance) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.pd_prefill_only = false;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("PREFILL"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsHostTierCombination) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.host_blocks_factor = 2.0;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("Host/Store"), std::string::npos);
}

TEST(PrefillLayerwiseEligibilityTest, RejectsEmbeddingWorker) {
  PrefillLayerwiseEligibility eligibility = baseline_eligibility();
  eligibility.generate_task = false;

  ASSERT_FALSE(eligibility.supported());
  EXPECT_NE(eligibility.error()->find("task=generate"), std::string::npos);
}

// The factory, not the value object, is the seam to the config system: a
// DFlash2 deployment sets only num_speculative_tokens, and the three-argument
// predicate must see model_type, algorithm and model_impl together.
TEST(PrefillLayerwiseEligibilityTest, FactoryTreatsSpeculativeTokensAsIntent) {
  runtime::Options options;
  options.backend("llm");
  options.task_type("generate");
  options.num_speculative_tokens(2);
  options.enable_speculative_decode(false);
  options.speculative_algorithm("DFlash2");

  const PrefillLayerwiseEligibility eligibility =
      build_prefill_layerwise_eligibility(
          options, "glm_moe_dsa", /*model_impl=*/"python");

  EXPECT_TRUE(eligibility.has_speculative_decode);
  EXPECT_TRUE(eligibility.dflash2_target);
  EXPECT_TRUE(eligibility.llm_backend);
  EXPECT_TRUE(eligibility.generate_task);
}

TEST(PrefillLayerwiseEligibilityTest, FactoryRequiresAllDflash2Predicates) {
  runtime::Options options;
  options.num_speculative_tokens(2);
  options.speculative_algorithm("DFlash2");

  EXPECT_FALSE(build_prefill_layerwise_eligibility(
                   options, "glm_moe_dsa", /*model_impl=*/"native")
                   .dflash2_target);
  EXPECT_FALSE(build_prefill_layerwise_eligibility(
                   options, "glm5_next", /*model_impl=*/"python")
                   .dflash2_target);
  options.speculative_algorithm("mtp");
  EXPECT_FALSE(build_prefill_layerwise_eligibility(
                   options, "glm_moe_dsa", /*model_impl=*/"python")
                   .dflash2_target);
}

}  // namespace
}  // namespace xllm
