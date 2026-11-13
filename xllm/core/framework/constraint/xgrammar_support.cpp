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

#include "core/framework/config/speculative_config.h"

namespace xllm {

namespace {
// Only these speculative algorithms drive their per-position verify masks
// from the unified constraint channel. Suffix-style proposers never read
// json_object_states, so they would emit unconstrained drafts and targets.
// Reuse SpeculativeConfig's centralized algorithm classification (MTP plus the
// block-diffusion family) instead of re-listing the strings here.
bool channel_supported_speculative(const std::string& algorithm) {
  return SpeculativeConfig::is_mtp_algorithm(algorithm) ||
         SpeculativeConfig::is_block_diffusion_algorithm(algorithm);
}
}  // namespace

std::vector<std::string> xgrammar_unsupported_features(
    const XGrammarSupportInputs& inputs) {
  // schedule_overlap is supported (single-advance through the channel view)
  // and needs no input. Speculative decoding is supported only for the
  // channel-migrated algorithms; suffix-style proposers stay gated.
  std::vector<std::string> unsupported;
  if (inputs.num_speculative_tokens > 0 &&
      !channel_supported_speculative(inputs.speculative_algorithm)) {
    unsupported.emplace_back("speculative_" +
                             (inputs.speculative_algorithm.empty()
                                  ? std::string("<unnamed>")
                                  : inputs.speculative_algorithm));
  }
  // Each gated feature names the unverified constraint-state interaction that
  // keeps it off (the gate is conservative: unverified, not known-broken).
  if (inputs.enable_graph) {
    // Target graph capture freezes the constraint mask as a graph input; the
    // per-position (draft/target phase) mask timing against graph replay is
    // unverified. The draft side is safe (fixed-shape packed bitmask), the
    // target side is not.
    unsupported.emplace_back("graph");
  }
  if (inputs.enable_prefill_piecewise_graph) {
    // Piecewise prefill runs the prompt in graph segments; the constraint
    // matcher's first-token initialization across those segment boundaries is
    // unverified.
    unsupported.emplace_back("prefill_piecewise_graph");
  }
  // chunked_prefill is not gated: it only splits the prompt across steps, and
  // the constraint matcher is initialized at the first decode token (after
  // prefill), so the chunk boundary cannot affect the decode-time grammar. It
  // is also a hard requirement of the block-diffusion family, so gating it
  // would make the unlocked speculative algorithms unstartable.
  if (inputs.enable_disagg_pd) {
    // PD disaggregation must ship the constraint state between prefill and
    // decode via the spec-transport snapshot, which is only implemented for
    // the builtin backend (xgrammar's transport lands with G9).
    unsupported.emplace_back("disagg_pd");
  }
  if (inputs.enable_service_routing) {
    // Service routing may migrate a request between workers; constraint-state
    // consistency across that migration is unverified.
    unsupported.emplace_back("service_routing");
  }
  if (inputs.enable_prefix_cache) {
    // A prefix-cache hit must leave the matcher aligned with the cached token
    // prefix (the matcher state that prefix implies) - unverified.
    unsupported.emplace_back("prefix_cache");
  }
  return unsupported;
}

}  // namespace xllm
