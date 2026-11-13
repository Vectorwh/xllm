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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

// Runtime feature switches that gate the xgrammar structured-output backend.
// Mirrors the engine config so the support check is a pure, testable function.
struct XGrammarSupportInputs final {
  bool enable_graph = false;
  bool enable_prefill_piecewise_graph = false;
  bool enable_disagg_pd = false;
  bool enable_service_routing = false;
  bool enable_prefix_cache = false;
  // Speculative decoding is supported only for the algorithms migrated to the
  // unified constraint channel (MTP / DFlash / DSpark / DFlash2); suffix-style
  // proposers never read the channel, so they stay gated. schedule_overlap is
  // supported (single-advance through the channel view) and needs no input.
  int32_t num_speculative_tokens = 0;
  std::string speculative_algorithm = "MTP";
};

// Features the xgrammar backend does not yet support, as human-readable names
// for the startup error. Empty means every listed feature is supported.
// Speculative decoding and schedule overlap were unlocked by the constraint
// state-machine migration (batches 4-6); the rest stay gated until their
// combinations are verified.
std::vector<std::string> xgrammar_unsupported_features(
    const XGrammarSupportInputs& inputs);

}  // namespace xllm
