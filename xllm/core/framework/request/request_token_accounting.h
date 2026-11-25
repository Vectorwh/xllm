/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace xllm {

struct OutputTokenProgress final {
  size_t index = 0;
  uint64_t generated_tokens = 0;
};

struct RequestTokenCounts final {
  bool enabled = false;
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
};

// Request-lifetime accounting: preemption/retry must not reset this ledger.
class RequestTokenAccounting final {
 public:
  uint64_t admit(uint64_t input_tokens, bool* first_admission = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (first_admission != nullptr) {
      *first_admission = !counts_.enabled;
    }
    if (counts_.enabled) {
      return 0;
    }
    counts_.enabled = true;
    counts_.input_tokens = input_tokens;
    return input_tokens;
  }

  // Decode continues an admitted request, never admitting its prompt again.
  // In streaming PD the first token belongs to Prefill's submission; Decode
  // may re-decode that token as byte context, but must not count it again.
  // Prefill owns index 0 only; Decode expands all additional sequences.
  void forward(bool stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (counts_.enabled) {
      return;
    }
    counts_.enabled = true;
    forwarded_stream_ = stream;
    if (stream) {
      // Context-only baseline, not a confirmed Prefill commit or counted
      // output.
      output_tokens_[0] = kPrefillHandoffTokens;
    }
  }

  // Progress is an immutable snapshot from response generation, not a read of
  // the sequence after the callback (which may already have advanced).
  // Preemption clears KV state, not the sequence's committed token history.
  uint64_t commit(const std::vector<OutputTokenProgress>& outputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!counts_.enabled) {
      return 0;
    }
    uint64_t delta = 0;
    for (const OutputTokenProgress& output : outputs) {
      uint64_t& previous = output_tokens_[output.index];
      if (output.generated_tokens <= previous) {
        continue;
      }
      delta += output.generated_tokens - previous;
      previous = output.generated_tokens;
    }
    counts_.output_tokens += delta;
    return delta;
  }

  bool enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return counts_.enabled;
  }

  RequestTokenCounts snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return counts_;
  }

  // This is proof from Prefill's response accounting, not its imported byte
  // context. It affects only full-length observation, not local Token/s.
  // FirstGeneration transports one token for sequence 0. A different count
  // cannot confirm that context baseline; callers must report the mismatch.
  bool confirm_upstream_output(uint64_t tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!forwarded_stream_) {
      return true;
    }
    if (tokens != kPrefillHandoffTokens) {
      return false;
    }
    upstream_output_tokens_ = tokens;
    return true;
  }

  std::optional<uint64_t> confirmed_output_tokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!counts_.enabled ||
        (forwarded_stream_ && !upstream_output_tokens_.has_value())) {
      return std::nullopt;
    }
    return counts_.output_tokens + upstream_output_tokens_.value_or(0);
  }

 private:
  static constexpr uint64_t kPrefillHandoffTokens = 1;
  mutable std::mutex mutex_;
  RequestTokenCounts counts_;
  bool forwarded_stream_ = false;
  std::optional<uint64_t> upstream_output_tokens_;
  // Sample-slot outputs use caller-supplied sample_id, not a dense sequence id.
  std::unordered_map<size_t, uint64_t> output_tokens_;
};

}  // namespace xllm
