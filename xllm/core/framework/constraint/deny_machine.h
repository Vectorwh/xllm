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

// A machine for rows whose constraint could not be restored: the request is
// failed through the error channel, and the row decodes this step under an
// all-deny mask instead of an unconstrained one, so the invariant "a
// structured-output request never samples outside its grammar" holds even on
// the failing step.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/framework/constraint/state_machine.h"

namespace xllm {

// Type-erased marker payload: presence is the only information a deny state
// carries, so the payload itself is empty.
struct DenyConstraintPayload final {};

// Every operation denies: no token is acceptable, no state is valid, and the
// packed mask is all zeros at the model's vocabulary width. The width comes
// from the construction site (the worker knows the model vocab), so a batch
// whose every row failed still assembles at the right width.
class DenyConstraintMachine final
    : public ConstraintStateMachine,
      public std::enable_shared_from_this<DenyConstraintMachine> {
 public:
  // Construct through wrap_deny_machine only: states embed the machine via a
  // shared_ptr aliasing construction that needs shared_from_this, so an
  // instance not owned by shared_ptr throws on first use.
  friend std::shared_ptr<DenyConstraintMachine> wrap_deny_machine(
      size_t num_words);

 private:
  explicit DenyConstraintMachine(size_t num_words);

 public:
  ConstraintState initial_state(bool reasoning_phase) const override;
  ConstraintState restore(const ConstraintSnapshot& snapshot) const override;

  bool can_accept(const ConstraintState& state,
                  int32_t token_id) const override;
  bool is_valid(const ConstraintState& state) const override;
  bool is_complete(const ConstraintState& state) const override;
  // A deny state never completes and never terminates: every token is
  // rejected through the normal error channel instead.
  bool is_terminated(const ConstraintState& state) const override;
  bool advance(ConstraintState& state,
               int32_t token_id,
               HaltPolicy halt) const override;
  ConstraintState fork(const ConstraintState& state) const override;
  ConstraintState replay(const std::vector<int32_t>& committed_tokens,
                         bool reasoning_phase) const override;

  std::vector<uint32_t> allowed_bitmask(
      const ConstraintState& state) const override;
  bool fill_allowed_bitmask(const ConstraintState& state,
                            int32_t* row,
                            size_t num_words) const override;
  size_t bitmask_num_words() const override;
  uint64_t fingerprint(const ConstraintState& state) const override;

  ConstraintSnapshot snapshot(const ConstraintState& state) const override;
  std::string error_detail(const ConstraintState& state) const override;

 private:
  ConstraintState make_state() const;

  size_t num_words_;
};

inline std::shared_ptr<DenyConstraintMachine> wrap_deny_machine(
    size_t num_words) {
  // Plain new: make_shared's internal construction is not covered by the
  // friend declaration guarding the private constructor.
  return std::shared_ptr<DenyConstraintMachine>(
      new DenyConstraintMachine(num_words));
}

}  // namespace xllm
