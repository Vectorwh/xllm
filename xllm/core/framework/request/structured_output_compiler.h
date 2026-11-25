/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "core/framework/constraint/snapshot.h"
#include "core/framework/sampling/xgrammar_constraint.h"
#include "framework/request/request_output.h"
#include "framework/request/request_state.h"
#include "framework/sampling/json_object_grammar.h"

namespace xllm {

class ModelArgs;
class Options;
class RequestParams;
class Tokenizer;

// Model-scoped structured-output compilation state shared by the LLM and VLM
// request factories: owns the lazily built XGrammar compiler and the cached
// builtin JSON-object grammars (with and without reasoning). The factories
// keep only the generation-mode policy that differs between the two paths
// (notably the VLM UNKNOWN-mode fallback) and delegate the mechanics here.
//
// The owner must outlive the compiler (the tokenizer, model args and options
// are non-owning references held by the owning factory).
class StructuredOutputCompiler final {
 public:
  StructuredOutputCompiler(const Tokenizer& tokenizer,
                           const ModelArgs& model_args,
                           const Options& options);

  StructuredOutputCompiler(const StructuredOutputCompiler&) = delete;
  StructuredOutputCompiler& operator=(const StructuredOutputCompiler&) = delete;

  // Validates the speculative algorithm and compiles the requested
  // response_format. On failure fires `callback` with
  // Status::invalid_parameter(..., "response_format") and returns false; on
  // success `constraint` and `spec` carry the result (null/empty for NONE and
  // for the builtin json_object path). Owns the shared admission-then-compile
  // protocol so the LLM and VLM factories stay in lockstep.
  bool build_response_format_constraint(
      const RequestParams& sp,
      const OutputCallback& callback,
      std::shared_ptr<const CompiledXGrammar>& constraint,
      GrammarSpec& spec);

  // Installs the compiled xgrammar constraint on req_state. Builtin keeps
  // priority: when a builtin machine already claims the slot, only the
  // constraint pointer is recorded. Preserves the constraint/machine
  // invariant CHECK.
  void attach_xgrammar_constraint(
      RequestState& req_state,
      std::shared_ptr<const CompiledXGrammar> constraint,
      GrammarSpec spec);

  // True when the request should use the builtin JSON-object machine rather
  // than the compiled xgrammar path. Single owner of the three-condition
  // routing rule so the LLM and VLM factories cannot drift.
  bool uses_builtin_json_object(const RequestParams& sp) const;

  // Attaches the builtin JSON-object grammar and machine to req_state once the
  // caller has resolved `reasoning_enabled`. Returns false after firing an
  // error callback on failure. Shared by the LLM and VLM paths; only their
  // generation-mode parsing differs.
  bool apply_json_object_grammar(RequestState& req_state,
                                 const RequestParams& sp,
                                 bool reasoning_enabled,
                                 const OutputCallback& callback);

 private:
  // Rejects a structured-output request when the configured speculative
  // algorithm does not drive the per-position constraint channel. This is
  // backend-independent: the builtin json_object machine applies the same
  // target-vocabulary mask as the xgrammar path, so the builtin route must be
  // rejected too.
  bool validate_speculative_support(const RequestParams& sp,
                                    std::string& error) const;

  // Compiles the per-request grammar; on success `spec` carries the exact
  // source the grammar was built from, so snapshot receivers rebuild an
  // equivalent grammar (G9).
  std::shared_ptr<const CompiledXGrammar> compile_response_format(
      const RequestParams& sp,
      GrammarSpec& spec,
      std::string& error);

  std::shared_ptr<const JsonObjectGrammar> get_json_object_grammar(
      bool reasoning_enabled,
      std::string& error);

  const Tokenizer& tokenizer_;
  const ModelArgs& model_args_;
  const Options& options_;

  const bool enable_xgrammar_output_;
  std::unique_ptr<XGrammarCompiler> xgrammar_compiler_;
  std::mutex json_object_grammar_mutex_;
  std::shared_ptr<const JsonObjectGrammar> json_object_grammar_;
  std::shared_ptr<const JsonObjectGrammar> json_reasoning_grammar_;
};

}  // namespace xllm
