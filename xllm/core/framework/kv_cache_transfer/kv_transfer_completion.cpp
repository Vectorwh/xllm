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

#include "core/framework/kv_cache_transfer/kv_transfer_completion.h"

#include <folly/futures/HeapTimekeeper.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>

#include "common/metrics.h"

namespace xllm {
namespace {

constexpr std::chrono::seconds kKVTransferWaitTimeout{60};
constexpr std::chrono::seconds kKVTransferDrainTimeout{1};

folly::Timekeeper* kv_transfer_timekeeper() {
  static folly::HeapTimekeeper timekeeper;
  return &timekeeper;
}

}  // namespace

KVTransferCompletion::KVTransferCompletion()
    : KVTransferCompletion(kKVTransferWaitTimeout) {}

KVTransferCompletion::KVTransferCompletion(
    std::chrono::milliseconds wait_timeout)
    : wait_timeout_(wait_timeout),
      tracker_(std::make_unique<KVTransferTracker>()) {
  CHECK_GT(wait_timeout_.count(), 0) << "wait timeout must be positive";
}

KVTransferCompletion::~KVTransferCompletion() {
  if (waited_ || futures_.empty()) {
    return;
  }
  // A worker step left this object without calling wait(): an exception
  // unwound the step (observed on PD prefill: 32k-token inputs whose
  // in-flight chunk push was still reading source KV blocks when a later
  // forward threw), or an early return skipped wait_kv_push(). Aborting the
  // process here turned the recoverable per-request failure into a whole
  // prefill-group crash. Cancel the stuck transfers first: the cancel
  // callback aborts the layer synchronizer, which unblocks the push thread's
  // per-layer waits and makes it report its keys as failed and complete. A
  // push that respects the abort stops reading source KV blocks within at
  // most one more layer read, so the post-cancel drain uses the full wait
  // timeout (the same bound wait() uses) instead of the short drain floor.
  // Failed request IDs are not reduced here: the failure collective is
  // unsafe during stack unwinding and the leaving path already owns the
  // failure.
  LOG(ERROR) << futures_.size()
             << " KV transfer(s) still pending at destruction; an exception "
                "or early return skipped wait(), cancelling and draining";
  COUNTER_ADD(kv_transfer_destruction_pending_total,
              static_cast<double>(futures_.size()));
  bool all_cancelled = true;
  for (PendingTransfer& pending : futures_) {
    if (pending.cancel) {
      pending.cancel();
    } else {
      all_cancelled = false;
    }
  }
  const std::chrono::milliseconds cancel_drain_timeout =
      std::max(wait_timeout_,
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   kKVTransferDrainTimeout));
  (void)collect_pending(cancel_drain_timeout);
  if (!tracker_->wait_for(cancel_drain_timeout)) {
    if (all_cancelled) {
      // all_cancelled means every transfer owned a cancel hook; it is not
      // proof that the hook worked. A transfer that is still pending after
      // cancel plus a full wait-timeout drain ignored the abort and may
      // still be reading source KV blocks. Abandoning the tracker releases
      // those tokens anyway: this deliberately trades a bounded
      // use-after-release read on a request that already failed against
      // killing the whole worker group while this thread unwinds. The error
      // names the risk so the incident stays greppable; transfers that were
      // never cancellable keep the fail-safe below.
      LOG(ERROR) << "KV transfers still pending after cancel and a "
                 << cancel_drain_timeout.count()
                 << "ms drain; releasing their tracker tokens although a "
                    "push thread that ignored abort may still read source "
                    "KV blocks";
      // Count the accepted-risk outcome from the tracker (not futures_.size(),
      // which still includes transfers that completed during the drain) so
      // monitoring sees how often cancel delivery fails without reading logs.
      COUNTER_ADD(kv_transfer_destruction_abandoned_total,
                  static_cast<double>(tracker_->pending_count()));
      tracker_->abandon();
    } else {
      // An uncancellable transfer may still be reading source KV blocks;
      // releasing them here risks a use-after-release read, so keep the
      // original fail-safe.
      LOG(FATAL) << "uncancellable KV cache push did not finish after "
                    "timeout";
    }
  }
  futures_.clear();
}

void KVTransferCompletion::add(
    folly::SemiFuture<std::vector<KVTransferTaskResult>> future,
    std::vector<std::string> fallback_request_ids,
    std::function<void()> cancel) {
  std::shared_ptr<KVTransferTracker::Completion> transfer_completion =
      tracker_->track();
  future = std::move(future).deferEnsure(
      [transfer_completion = std::move(transfer_completion)]() mutable {
        transfer_completion.reset();
      });
  futures_.push_back(
      {std::move(future), std::move(fallback_request_ids), std::move(cancel)});
}

folly::Try<std::vector<folly::Try<std::vector<KVTransferTaskResult>>>>
KVTransferCompletion::collect_pending(
    std::chrono::milliseconds collect_timeout) {
  std::vector<folly::SemiFuture<std::vector<KVTransferTaskResult>>> futures;
  futures.reserve(futures_.size());
  for (PendingTransfer& pending : futures_) {
    futures.emplace_back(std::move(pending.future));
  }
  return std::move(folly::collectAll(std::move(futures))
                       .within(collect_timeout, kv_transfer_timekeeper()))
      .getTry();
}

std::unordered_set<std::string> KVTransferCompletion::wait() {
  if (waited_) {
    return failed_request_ids_;
  }
  if (futures_.empty()) {
    waited_ = true;
    return {};
  }

  folly::Try<std::vector<folly::Try<std::vector<KVTransferTaskResult>>>>
      completion_result = collect_pending(wait_timeout_);
  const bool timed_out = !completion_result.hasValue();
  if (!timed_out) {
    const std::vector<folly::Try<std::vector<KVTransferTaskResult>>>& results =
        completion_result.value();
    for (size_t index = 0; index < results.size(); ++index) {
      const folly::Try<std::vector<KVTransferTaskResult>>& result =
          results[index];
      if (!result.hasValue()) {
        LOG(ERROR) << "KV cache transfer future failed: "
                   << result.exception().what();
        failed_request_ids_.insert(futures_[index].fallback_request_ids.begin(),
                                   futures_[index].fallback_request_ids.end());
        // A future that threw never reported a batch status either, so it
        // cannot prove the destination buffers were retired.
        unsafe_request_ids_.insert(futures_[index].fallback_request_ids.begin(),
                                   futures_[index].fallback_request_ids.end());
        continue;
      }
      for (const KVTransferTaskResult& task : result.value()) {
        if (task.error_code != KVTransferErrorCode::NONE) {
          failed_request_ids_.insert(task.request_ids.begin(),
                                     task.request_ids.end());
          if (task.error_code == KVTransferErrorCode::UNSAFE) {
            unsafe_request_ids_.insert(task.request_ids.begin(),
                                       task.request_ids.end());
          }
        }
      }
    }
  }
  if (timed_out) {
    // A transfer that did not even report a batch status never proved that
    // the destination buffers were retired, so its requests are unsafe too.
    for (const PendingTransfer& pending : futures_) {
      failed_request_ids_.insert(pending.fallback_request_ids.begin(),
                                 pending.fallback_request_ids.end());
      unsafe_request_ids_.insert(pending.fallback_request_ids.begin(),
                                 pending.fallback_request_ids.end());
    }
  }
  // A timeout only changes the request result. The transfer continuation may
  // still be reading source KV blocks, so drain it before releasing futures.
  // Keep a short configured timeout from turning normal async cleanup into an
  // immediate fatal path, while still bounding a permanently stuck transfer.
  if (timed_out) {
    const std::chrono::milliseconds drain_timeout =
        std::max(wait_timeout_,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     kKVTransferDrainTimeout));
    if (!tracker_->wait_for(drain_timeout)) {
      LOG(FATAL) << "KV cache push did not finish after timeout";
    }
  } else {
    tracker_->wait();
  }
  futures_.clear();
  waited_ = true;
  return failed_request_ids_;
}

class KVTransferTracker::State final {
 public:
  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++pending_transfers_;
  }

  void finish() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Tolerate only the late completions of tokens that were alive when
      // abandon() cleared the count, and only up to that many: a finish
      // beyond the abandoned population is a track/finish accounting bug
      // that must abort loudly again.
      if (abandoned_pending_ > 0) {
        --abandoned_pending_;
        LOG(ERROR) << "KV transfer finished after its tracker was abandoned";
        return;
      }
      CHECK_GT(pending_transfers_, 0u);
      --pending_transfers_;
    }
    completion_cv_.notify_all();
  }

  void abandon() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // Record how many outstanding tokens may still finish late; further
      // finishes are accounting bugs (see finish()).
      abandoned_pending_ = pending_transfers_;
      pending_transfers_ = 0;
    }
    completion_cv_.notify_all();
  }

  bool has_pending() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_transfers_ > 0;
  }

  size_t pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_transfers_;
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    completion_cv_.wait(lock, [this]() { return pending_transfers_ == 0; });
  }

  bool wait_for(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return completion_cv_.wait_for(
        lock, timeout, [this]() { return pending_transfers_ == 0; });
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable completion_cv_;
  size_t pending_transfers_ = 0;
  size_t abandoned_pending_ = 0;
};

KVTransferTracker::Completion::Completion(std::shared_ptr<State> state)
    : state_(std::move(state)) {
  CHECK(state_ != nullptr);
  state_->start();
}

KVTransferTracker::Completion::~Completion() { state_->finish(); }

KVTransferTracker::KVTransferTracker() : state_(std::make_shared<State>()) {}

KVTransferTracker::~KVTransferTracker() { wait(); }

std::shared_ptr<KVTransferTracker::Completion> KVTransferTracker::track() {
  return std::shared_ptr<Completion>(new Completion(state_));
}

bool KVTransferTracker::has_pending() const { return state_->has_pending(); }

size_t KVTransferTracker::pending_count() const {
  return state_->pending_count();
}

void KVTransferTracker::wait() { state_->wait(); }

bool KVTransferTracker::wait_for(std::chrono::milliseconds timeout) {
  CHECK_GT(timeout.count(), 0) << "wait timeout must be positive";
  return state_->wait_for(timeout);
}

void KVTransferTracker::abandon() { state_->abandon(); }

}  // namespace xllm
