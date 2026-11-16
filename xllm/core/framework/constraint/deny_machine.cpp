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

#include "core/framework/constraint/deny_machine.h"

namespace xllm {
namespace {

// Stable fingerprint for deny states: distinct from any grammar's mask hash
// so cache entries can never collide with a real row ("deny" in ASCII).
constexpr uint64_t kDenyFingerprint = 0x64656E79ULL;

}  // namespace

DenyConstraintMachine::DenyConstraintMachine(size_t num_words)
    : num_words_(num_words) {}

ConstraintState DenyConstraintMachine::make_state() const {
  return ConstraintState(
      std::make_shared<DenyConstraintPayload>(),
      std::shared_ptr<const ConstraintStateMachine>(shared_from_this(), this));
}

ConstraintState DenyConstraintMachine::initial_state(
    bool /*reasoning_phase*/) const {
  return make_state();
}

ConstraintState DenyConstraintMachine::restore(
    const ConstraintSnapshot& /*snapshot*/) const {
  // Restoring a deny row is a deny row: there is no history to replay.
  return make_state();
}

bool DenyConstraintMachine::can_accept(const ConstraintState& /*state*/,
                                       int32_t /*token_id*/) const {
  return false;
}

bool DenyConstraintMachine::is_valid(const ConstraintState& /*state*/) const {
  return false;
}

bool DenyConstraintMachine::is_complete(
    const ConstraintState& /*state*/) const {
  return false;
}

bool DenyConstraintMachine::is_terminated(
    const ConstraintState& /*state*/) const {
  // A deny state never completes and never terminates; every token is
  // rejected through the normal error channel instead.
  return false;
}

bool DenyConstraintMachine::advance(ConstraintState& /*state*/,
                                    int32_t /*token_id*/,
                                    HaltPolicy /*halt*/) const {
  return false;
}

ConstraintState DenyConstraintMachine::fork(
    const ConstraintState& state) const {
  return state;
}

ConstraintState DenyConstraintMachine::replay(
    const std::vector<int32_t>& /*committed_tokens*/,
    bool /*reasoning_phase*/) const {
  return make_state();
}

std::vector<uint32_t> DenyConstraintMachine::allowed_bitmask(
    const ConstraintState& /*state*/) const {
  return std::vector<uint32_t>(num_words_, 0);
}

bool DenyConstraintMachine::fill_allowed_bitmask(
    const ConstraintState& /*state*/,
    int32_t* row,
    size_t num_words) const {
  if (row == nullptr) {
    return false;
  }
  std::fill_n(row, num_words, 0);
  return true;
}

size_t DenyConstraintMachine::bitmask_num_words() const { return num_words_; }

uint64_t DenyConstraintMachine::fingerprint(
    const ConstraintState& /*state*/) const {
  return kDenyFingerprint;
}

ConstraintSnapshot DenyConstraintMachine::snapshot(
    const ConstraintState& /*state*/) const {
  // A deny row has nothing serializable: the request is failing through the
  // error channel, and the disabled snapshot keeps it out of the restore
  // path entirely.
  ConstraintSnapshot snapshot;
  snapshot.enabled = false;
  return snapshot;
}

std::string DenyConstraintMachine::error_detail(
    const ConstraintState& /*state*/) const {
  return "the constraint could not be restored; the row is denied";
}

}  // namespace xllm
