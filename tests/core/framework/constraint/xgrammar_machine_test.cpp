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

#include "core/framework/constraint/xgrammar_machine.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "core/framework/constraint/draft_validation.h"
#include "core/framework/sampling/xgrammar_constraint.h"

namespace xllm {
namespace {

constexpr int32_t kStopTokenId = 5;

std::shared_ptr<XGrammarConstraintMachine> make_machine() {
  auto compiler = std::make_unique<XGrammarCompiler>(
      std::vector<std::string>{"A", "B", "-", "1", "x", "", "{", "}"},
      /*model_vocab_size=*/32,
      std::vector<int32_t>{kStopTokenId},
      /*cache_max_memory_bytes=*/0);
  std::string error;
  auto grammar = compiler->compile_regex("[A-Z]{2}-[0-9]{4}", error);
  if (grammar == nullptr) {
    ADD_FAILURE() << error;
    return nullptr;
  }
  // Carry the source as the spec so snapshots identify the grammar
  // cross-process, exactly like the request factory wires it.
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::REGEX;
  spec.source = "[A-Z]{2}-[0-9]{4}";
  return wrap_xgrammar_machine(std::move(grammar), spec);
}

TEST(XGrammarMachineTest, InitialStateAcceptsRegexPrefix) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  ASSERT_TRUE(static_cast<bool>(state));
  EXPECT_TRUE(machine->is_valid(state));
  EXPECT_FALSE(machine->is_complete(state));
  EXPECT_TRUE(machine->can_accept(state, 0));   // "A"
  EXPECT_FALSE(machine->can_accept(state, 4));  // "x"
}

TEST(XGrammarMachineTest, StopTokenIsAcceptedOnlyAtATerminalState) {
  // Stop handling lives in the tokenizer layer (this compiler is built with
  // kStopTokenId as a terminator), not in the commit paths: neither
  // append_token nor the overlap commit special-cases stops, they both ask the
  // matcher. Pin that decision here - a mid-pattern stop is a grammar
  // violation (the request should fail), while the stop at a fully matched
  // pattern is legitimate and must be accepted.
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);

  ConstraintState mid = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(mid, /*A=*/0, HaltPolicy::RESPECT));
  EXPECT_FALSE(machine->advance(mid, kStopTokenId, HaltPolicy::RESPECT));

  ConstraintState matched = machine->initial_state(false);
  for (const int32_t token : {0, 0, 2, 3, 3, 3, 3}) {  // matches "AA-1111"
    ASSERT_TRUE(machine->advance(matched, token, HaltPolicy::RESPECT));
  }
  EXPECT_TRUE(machine->advance(matched, kStopTokenId, HaltPolicy::RESPECT));
  EXPECT_TRUE(machine->is_complete(matched));
  EXPECT_TRUE(machine->is_terminated(matched));
  // Schedule overlap must observe the terminal state instead of advancing a
  // matcher that has already consumed its stop token.
  EXPECT_FALSE(machine->can_accept(matched, 0));
  EXPECT_FALSE(machine->advance(matched, 0, HaltPolicy::IGNORE));
  const auto terminated_mask = machine->allowed_bitmask(matched);
  ASSERT_FALSE(terminated_mask.empty());
  for (const uint32_t word : terminated_mask) {
    EXPECT_EQ(word, 0u);
  }
}

TEST(XGrammarMachineTest, AdvanceCommitsTokensAndDetectsCompletion) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  for (const int32_t token_id : {0, 1, 2, 3, 3, 3, 3}) {
    EXPECT_TRUE(machine->advance(state, token_id, HaltPolicy::IGNORE));
  }
  EXPECT_TRUE(machine->is_complete(state));
}

TEST(XGrammarMachineTest, HaltRespectFreezesOnFirstRejection) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);

  EXPECT_FALSE(machine->advance(state, 4, HaltPolicy::RESPECT));  // "x"
  EXPECT_FALSE(machine->is_valid(state));
  // Frozen: everything reports rejected, including valid tokens.
  EXPECT_FALSE(machine->can_accept(state, 0));
  EXPECT_FALSE(machine->advance(state, 0, HaltPolicy::RESPECT));
  // allowed_bitmask agrees with can_accept on a frozen row: all-deny, so the
  // frozen mask/can_accept invariant holds (the grammar-legal token is denied).
  const auto frozen_mask = machine->allowed_bitmask(state);
  ASSERT_FALSE(frozen_mask.empty());
  for (const uint32_t word : frozen_mask) {
    EXPECT_EQ(word, 0u);
  }

  // IGNORE keeps the matcher usable after a rejection (target-verify rows).
  ConstraintState verify = machine->initial_state(false);
  EXPECT_FALSE(machine->advance(verify, 4, HaltPolicy::IGNORE));
  EXPECT_TRUE(machine->is_valid(verify));
  EXPECT_TRUE(machine->advance(verify, 0, HaltPolicy::IGNORE));
}

TEST(XGrammarMachineTest, ForkedViewDoesNotAdvanceTheSequenceOwnedMatcher) {
  // After the overlap contract was unified to fork, the channel view is an
  // independent copy: advancing it must NOT touch the sequence-owned matcher
  // (the scheduler advances that via update_last_step_token's accept_token).
  auto compiler = std::make_unique<XGrammarCompiler>(
      std::vector<std::string>{"A", "B", "-", "1", "x", "", "{", "}"},
      /*model_vocab_size=*/32,
      std::vector<int32_t>{kStopTokenId},
      /*cache_max_memory_bytes=*/0);
  std::string error;
  auto grammar = compiler->compile_regex("[A-Z]{2}-[0-9]{4}", error);
  ASSERT_NE(grammar, nullptr) << error;
  auto machine = wrap_xgrammar_machine(grammar);
  ASSERT_NE(machine, nullptr);
  // The sequence owns the matcher (unique_ptr), exactly like Sequence.
  auto owned = std::make_unique<XGrammarState>(*grammar);
  ConstraintState view(std::shared_ptr<XGrammarState>(
                           std::shared_ptr<XGrammarState>(), owned.get()),
                       machine);
  ConstraintState forked = view.fork();
  const uint64_t before = owned->fingerprint();
  ASSERT_TRUE(machine->advance(forked, 0, HaltPolicy::IGNORE));
  // The forked copy advanced, but the sequence-owned matcher is untouched:
  // its fingerprint is unchanged and it holds no accepted tokens.
  EXPECT_EQ(owned->fingerprint(), before);
  EXPECT_TRUE(owned->accepted_tokens().empty());
}

TEST(XGrammarMachineTest, ForkDivergesIndependentlyFromSource) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState base = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(base, 0, HaltPolicy::IGNORE));

  ConstraintState branch = machine->fork(base);
  EXPECT_EQ(machine->fingerprint(branch), machine->fingerprint(base));
  // "x" is not accepted at the second-letter position; RESPECT freezes the
  // branch while the source keeps its position.
  EXPECT_FALSE(machine->advance(branch, 4, HaltPolicy::RESPECT));
  EXPECT_FALSE(machine->is_valid(branch));
  EXPECT_TRUE(machine->is_valid(base));
  EXPECT_NE(machine->fingerprint(branch), machine->fingerprint(base));
}

TEST(XGrammarMachineTest, SnapshotRestoreRoundTripsHistory) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  ASSERT_TRUE(machine->advance(state, 0, HaltPolicy::IGNORE));
  ASSERT_TRUE(machine->advance(state, 1, HaltPolicy::IGNORE));

  const ConstraintSnapshot snap = machine->snapshot(state);
  ASSERT_TRUE(snap.enabled);
  ASSERT_EQ(snap.token_ids, std::vector<int32_t>({0, 1}));

  ConstraintState restored = machine->restore(snap);
  EXPECT_TRUE(machine->is_valid(restored));
  EXPECT_EQ(machine->fingerprint(restored), machine->fingerprint(state));
  EXPECT_TRUE(machine->can_accept(restored, 2));  // "-" after "AB"

  ConstraintState replayed = machine->replay(snap.token_ids, false);
  EXPECT_EQ(machine->fingerprint(replayed), machine->fingerprint(state));

  const std::vector<int32_t> bad_history = {4};
  ConstraintState bad = machine->replay(bad_history, false);
  EXPECT_FALSE(machine->is_valid(bad));
}

TEST(XGrammarMachineTest, FrozenStateRoundTripsItsInvalidFlag) {
  // A RESPECT-frozen state must stay frozen across snapshot/restore: replay
  // alone would return a valid matcher whose fingerprint differs (the invalid
  // bit is hashed), losing the halt.
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  // Token 4 ("x") is illegal at the first position under RESPECT: it freezes.
  EXPECT_FALSE(machine->advance(state, 4, HaltPolicy::RESPECT));
  ASSERT_FALSE(machine->is_valid(state));

  const ConstraintSnapshot snap = machine->snapshot(state);
  ASSERT_TRUE(snap.invalid);

  ConstraintState restored = machine->restore(snap);
  EXPECT_FALSE(machine->is_valid(restored));
  EXPECT_EQ(machine->fingerprint(restored), machine->fingerprint(state));
  // A frozen matcher rejects everything, even a token the bare grammar allows.
  EXPECT_FALSE(machine->can_accept(restored, 0));
}

TEST(XGrammarMachineTest, SnapshotAdvertisesSpecSupportAndCarriesIt) {
  // G9: the machine-level snapshot carries the grammar spec so a
  // cross-process receiver can rebuild the exact grammar and replay; the
  // support flag is what routes rows through the snapshot channel instead
  // of the live-state preservation fallback.
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  EXPECT_TRUE(machine->supports_spec_snapshot());
  ConstraintState state = machine->initial_state(false);
  const ConstraintSnapshot snap = machine->snapshot(state);
  EXPECT_TRUE(snap.enabled);
  // make_machine compiles the REGEX source "[A-Z]{2}-[0-9]{4}", so the
  // spec rides along with that exact source.
  ASSERT_NE(snap.grammar_spec, nullptr);
  EXPECT_FALSE(snap.grammar_spec->empty());
  EXPECT_EQ(snap.grammar_spec->source, "[A-Z]{2}-[0-9]{4}");
}

TEST(XGrammarMachineTest, SnapshotRebuildAcrossMachinesRoundTrips) {
  // G9 core semantics: a snapshot from machine A (built with a spec)
  // carries the grammar identity, so machine B - compiled independently
  // from the same spec, like a decode worker - restores to the exact same
  // FSM state: identical allowed bitmask and fingerprint.
  std::shared_ptr<XGrammarConstraintMachine> source = make_machine();
  ASSERT_NE(source, nullptr);
  ConstraintState state = source->initial_state(false);
  ASSERT_TRUE(source->advance(state, 0 /* 'A' */, HaltPolicy::RESPECT));
  const ConstraintSnapshot snap = source->snapshot(state);
  ASSERT_TRUE(snap.enabled);
  ASSERT_NE(snap.grammar_spec, nullptr);
  EXPECT_EQ(snap.grammar_spec->type, GrammarSpec::Type::REGEX);
  EXPECT_EQ(snap.grammar_spec->source, "[A-Z]{2}-[0-9]{4}");
  EXPECT_EQ(snap.token_ids, std::vector<int32_t>({0}));

  // The receiver rebuilds from the spec (fresh compiler, like a worker).
  auto compiler = std::make_unique<XGrammarCompiler>(
      std::vector<std::string>{"A", "B", "-", "1", "x", "", "{", "}"},
      /*model_vocab_size=*/32,
      std::vector<int32_t>{kStopTokenId},
      /*cache_max_memory_bytes=*/0);
  std::string error;
  auto grammar = compiler->compile_regex(snap.grammar_spec->source, error);
  ASSERT_NE(grammar, nullptr) << error;
  auto receiver = wrap_xgrammar_machine(grammar, *snap.grammar_spec);
  ConstraintState restored = receiver->restore(snap);
  EXPECT_TRUE(restored.is_valid());
  EXPECT_EQ(receiver->fingerprint(restored), source->fingerprint(state));
  EXPECT_EQ(receiver->allowed_bitmask(restored),
            source->allowed_bitmask(state));
  // The restored state advances identically.
  ConstraintState restored_copy = restored;
  ASSERT_TRUE(
      receiver->advance(restored_copy, 1 /* 'B' */, HaltPolicy::RESPECT));
  EXPECT_TRUE(receiver->can_accept(restored_copy, 2 /* '-' */));
}

TEST(XGrammarMachineTest, FrozenSnapshotStaysFrozenAcrossRebuild) {
  // A RESPECT freeze must survive a cross-machine rebuild: the receiver
  // restores the invalid flag instead of replaying the clean history back
  // to a valid matcher, and the worker keeps the row (all-deny mask)
  // instead of failing the request.
  std::shared_ptr<XGrammarConstraintMachine> source = make_machine();
  ASSERT_NE(source, nullptr);
  ConstraintState state = source->initial_state(false);
  // Freeze through a RESPECT rejection.
  ASSERT_FALSE(
      source->advance(state, 5 /* stop mid-pattern */, HaltPolicy::RESPECT));
  EXPECT_FALSE(source->is_valid(state));
  const ConstraintSnapshot snap = source->snapshot(state);
  EXPECT_TRUE(snap.invalid);
  ASSERT_NE(snap.grammar_spec, nullptr);

  auto compiler = std::make_unique<XGrammarCompiler>(
      std::vector<std::string>{"A", "B", "-", "1", "x", "", "{", "}"},
      /*model_vocab_size=*/32,
      std::vector<int32_t>{kStopTokenId},
      /*cache_max_memory_bytes=*/0);
  std::string error;
  auto grammar = compiler->compile_regex(snap.grammar_spec->source, error);
  ASSERT_NE(grammar, nullptr) << error;
  auto receiver = wrap_xgrammar_machine(grammar, *snap.grammar_spec);
  ConstraintState restored = receiver->restore(snap);
  EXPECT_TRUE(restored.initialized());
  EXPECT_FALSE(receiver->is_valid(restored));
  // The frozen state rejects everything, including grammar-legal tokens.
  EXPECT_FALSE(receiver->can_accept(restored, 0));
}

TEST(XGrammarMachineTest, StrictJsonSchemaSpecRoundTrips) {
  // strict participates in the machine-cache key and the compile flags, so
  // its round trip must be pinned: a strict spec rebuilds a strict grammar
  // with the same fingerprint.
  auto compiler = std::make_unique<XGrammarCompiler>(
      std::vector<std::string>{"A", "B", "-", "1", "x", "", "{", "}"},
      /*model_vocab_size=*/32,
      std::vector<int32_t>{kStopTokenId},
      /*cache_max_memory_bytes=*/0);
  std::string error;
  GrammarSpec spec;
  spec.type = GrammarSpec::Type::JSON_SCHEMA;
  spec.source = R"({"type":"object"})";
  spec.strict = true;
  auto grammar = compiler->compile_json_schema(spec.source, spec.strict, error);
  ASSERT_NE(grammar, nullptr) << error;
  auto source = wrap_xgrammar_machine(grammar, spec);
  ConstraintState state = source->initial_state(false);

  const ConstraintSnapshot snap = source->snapshot(state);
  ASSERT_NE(snap.grammar_spec, nullptr);
  EXPECT_EQ(snap.grammar_spec->type, GrammarSpec::Type::JSON_SCHEMA);
  EXPECT_TRUE(snap.grammar_spec->strict);

  auto receiver = wrap_xgrammar_machine(grammar, *snap.grammar_spec);
  ConstraintState restored = receiver->restore(snap);
  EXPECT_TRUE(restored.is_valid());
  EXPECT_EQ(receiver->fingerprint(restored), source->fingerprint(state));
}

TEST(XGrammarMachineTest, DraftValidationAndVerifyWorkOnXgrammarStates) {
  // The shared speculative helpers must treat xgrammar states exactly like
  // builtin ones: draft rows fork before advancing, verification probes
  // per position.
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  // advance_json_draft_step mutates the states it advances; keep the
  // pristine rows for building validation states.
  const std::vector<ConstraintState> initial = {machine->initial_state(false),
                                                ConstraintState()};
  std::vector<ConstraintState> states = initial;
  std::vector<uint8_t> invalid_suffix(2, 0);
  detail::JsonDraftValidationScratch scratch;

  EXPECT_FALSE(
      detail::advance_json_draft_step(states, invalid_suffix, {0, 4}, scratch));
  EXPECT_FALSE(
      detail::advance_json_draft_step(states, invalid_suffix, {1, 4}, scratch));
  // Unconstrained row 1 accepts anything, row 0 stays on the regex.
  EXPECT_EQ(invalid_suffix, std::vector<uint8_t>({0, 0}));

  std::vector<uint8_t> invalid_draft;
  const std::vector<ConstraintState> validation_states =
      detail::build_json_validation_states(initial, scratch, invalid_draft);
  ASSERT_EQ(validation_states.size(), 6u);  // 2 sequences x (2 draft + live)

  // Target verify: the legal prefix row reports no mismatch, while a row
  // starting with the illegal "x" reports it at offset 0.
  const int64_t legal[] = {0, 1};
  EXPECT_TRUE(
      detail::verify_json_accepted_rows({validation_states[0]}, legal, 1, 2)
          .empty());
  const int64_t illegal[] = {4, -1};
  const std::vector<detail::JsonAcceptedTokenMismatch> mismatches =
      detail::verify_json_accepted_rows({validation_states[0]}, illegal, 1, 2);
  ASSERT_EQ(mismatches.size(), 1u);
  EXPECT_EQ(mismatches[0].sequence_index, 0u);
  EXPECT_EQ(mismatches[0].token_id, 4);
}

TEST(XGrammarMachineTest, AllowedBitmaskMatchesCanAccept) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  const std::vector<uint32_t> bitmask = machine->allowed_bitmask(state);
  ASSERT_EQ(bitmask.size(), 1u);  // 32-token vocab -> one word

  for (int32_t token = 0; token < 8; ++token) {
    const bool allowed = (bitmask[0] & (1U << token)) != 0U;
    EXPECT_EQ(allowed, machine->can_accept(state, token)) << "token " << token;
  }
  EXPECT_EQ(machine->fingerprint(state),
            machine->fingerprint(machine->initial_state(false)));
}

TEST(XGrammarMachineTest, ErrorDetailNamesTheRejectedState) {
  std::shared_ptr<XGrammarConstraintMachine> machine = make_machine();
  ASSERT_NE(machine, nullptr);
  ConstraintState state = machine->initial_state(false);
  const std::string detail = machine->error_detail(state);
  EXPECT_NE(detail.find("state_fingerprint"), std::string::npos);
  EXPECT_NE(detail.find("committed_tokens"), std::string::npos);
}

}  // namespace
}  // namespace xllm
