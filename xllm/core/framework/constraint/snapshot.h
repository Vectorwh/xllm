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
#include <memory>
#include <string>
#include <vector>

namespace xllm {

// Identity of a grammar compiled from a per-request source (xgrammar is the
// first such backend: every request carries its own schema or regex, unlike
// builtin's single fixed JSON grammar). A snapshot receiver rebuilds the
// machine from this spec; an empty source means no per-request grammar and
// the receiver falls back to the backend's constructed-at-init grammar.
class GrammarSpec final {
 public:
  // Shared by compile-time request validation and the packed-input reader.
  // Reject oversized user input before it can reach a descriptor CHECK.
  static constexpr uint64_t kMaxSourceBytes = 1ULL << 24;
  enum class Type : int8_t { JSON_SCHEMA = 0, REGEX = 1 };
  Type type = Type::JSON_SCHEMA;
  std::string source;
  bool strict = false;

  bool empty() const { return source.empty(); }

  // Single source-size policy shared by the compiler entry points and the
  // machine cache, so the two enforcement sites cannot drift apart.
  static bool source_within_limit(const std::string& source) {
    return source.size() <= kMaxSourceBytes;
  }
};

// Backend-neutral transport form of a per-sequence constraint state.
struct ConstraintSnapshot final {
  bool enabled = false;
  bool reasoning_enabled = false;
  std::vector<int32_t> token_ids;
  // Frozen (halted) states must round-trip their invalid flag; builtin
  // freezes on RESPECT rejections and piece-level violations, xgrammar on
  // RESPECT rejections.
  bool invalid = false;
  // Grammar identity for per-request grammars; the machine fills it when it
  // snapshots so a cross-process receiver can rebuild the exact grammar
  // before replaying the committed history. Immutable and shared: the
  // source can reach kMaxSourceBytes while snapshots are copied per decode
  // step, so the machine hands out its one retained spec by pointer. Null
  // means no per-request grammar (the builtin or unconstrained case).
  std::shared_ptr<const GrammarSpec> grammar_spec;
};

}  // namespace xllm
