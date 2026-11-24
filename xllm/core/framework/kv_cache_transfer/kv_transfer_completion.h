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

#pragma once

#include <folly/futures/Future.h>

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/kv_cache_transfer/kv_transfer_types.h"

namespace xllm {

class Device;
class InstanceRole;
struct ParallelArgs;
struct TransferKVInfo;

std::vector<std::string> canonical_transfer_request_ids(
    const std::vector<TransferKVInfo>& transfer_kv_infos);

std::vector<std::string> reduce_failed_request_ids(
    const std::unordered_set<std::string>& local_failed_request_ids,
    const std::vector<std::string>& canonical_request_ids,
    const ParallelArgs& parallel_args,
    const Device& device);

class KVTransferTracker;

// Owns asynchronous KV transfers until every transfer reaches a terminal
// state. Source KV blocks must not be released while this object is pending.
class KVTransferCompletion final {
 public:
  KVTransferCompletion();
  explicit KVTransferCompletion(std::chrono::milliseconds wait_timeout);
  ~KVTransferCompletion();

  KVTransferCompletion(const KVTransferCompletion&) = delete;
  KVTransferCompletion& operator=(const KVTransferCompletion&) = delete;
  KVTransferCompletion(KVTransferCompletion&&) = delete;
  KVTransferCompletion& operator=(KVTransferCompletion&&) = delete;

  void add(folly::SemiFuture<std::vector<KVTransferTaskResult>> future,
           std::vector<std::string> fallback_request_ids,
           std::function<void()> cancel = {});

  // Waits until all owned transfers finish and returns the failed request IDs.
  std::unordered_set<std::string> wait();

  // Request IDs whose transfer ended without a proof that the target buffers
  // stopped being accessed. Always a subset of the failed IDs.
  const std::unordered_set<std::string>& unsafe_request_ids() const {
    return unsafe_request_ids_;
  }

 private:
  // Aggregates the pending futures into one completion bounded by
  // collect_timeout. Returns the aggregated Try: hasValue means every transfer
  // finished; an empty Try means the collection timed out. Callers that only
  // need termination may discard the value.
  folly::Try<std::vector<folly::Try<std::vector<KVTransferTaskResult>>>>
  collect_pending(std::chrono::milliseconds collect_timeout);

  std::chrono::milliseconds wait_timeout_;
  std::unique_ptr<KVTransferTracker> tracker_;
  struct PendingTransfer {
    folly::SemiFuture<std::vector<KVTransferTaskResult>> future;
    std::vector<std::string> fallback_request_ids;
    // Unblocks a stuck transfer (e.g. aborts the layer synchronizer) so the
    // push thread stops reading source KV blocks, reports its keys as
    // failed, and completes its future. Empty when the transfer cannot be
    // cancelled; the destruction path then keeps its fail-safe.
    std::function<void()> cancel;
  };
  std::vector<PendingTransfer> futures_;
  std::unordered_set<std::string> failed_request_ids_;
  std::unordered_set<std::string> unsafe_request_ids_;
  bool waited_ = false;
};

// Waits for the owned PUSH transfers, reports every request whose transfer
// did not retire safely to its Decode peer, and returns the failed request IDs
// reduced across ranks. The instance role is identical across ranks; an empty
// local transfer list is not a safe gate. The report RPC completes before this
// call returns, so Decode has parked the affected target blocks before the
// caller's failure output lets the scheduler release those reservations.
std::vector<std::string> finalize_kv_push_failures(
    KVTransferCompletion& kv_transfers,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const std::string& kv_cache_transfer_mode,
    InstanceRole instance_role,
    const ParallelArgs& parallel_args,
    const Device& device);

// Tracks callbacks that retain KV block managers or blocks. A Completion
// token keeps one callback pending; releasing the last token unblocks wait().
// Destruction waits so owners can declare this as their last member and make
// callback lifetime a construction invariant instead of custom teardown code.
class KVTransferTracker final {
 private:
  class State;

 public:
  class Completion final {
   public:
    ~Completion();

    Completion(const Completion&) = delete;
    Completion& operator=(const Completion&) = delete;
    Completion(Completion&&) = delete;
    Completion& operator=(Completion&&) = delete;

   private:
    friend class KVTransferTracker;

    explicit Completion(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;
  };

  KVTransferTracker();
  ~KVTransferTracker();

  KVTransferTracker(const KVTransferTracker&) = delete;
  KVTransferTracker& operator=(const KVTransferTracker&) = delete;
  KVTransferTracker(KVTransferTracker&&) = delete;
  KVTransferTracker& operator=(KVTransferTracker&&) = delete;

  std::shared_ptr<Completion> track();
  bool has_pending() const;
  // Number of tracked transfers whose completion token is still held, i.e.
  // transfers whose push thread has not finished. Read before abandon():
  // abandon itself clears the count.
  size_t pending_count() const;
  void wait();
  bool wait_for(std::chrono::milliseconds timeout);
  // Clears the pending count so wait()/~KVTransferTracker() return without
  // waiting for transfers that will never finish. Late completions are
  // logged and ignored.
  void abandon();

 private:
  std::shared_ptr<State> state_;
};

}  // namespace xllm
