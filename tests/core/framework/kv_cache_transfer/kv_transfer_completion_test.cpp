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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/common/types.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/platform/device.h"

namespace xllm {
namespace {

using namespace std::chrono_literals;

class SimulatedAllGatherProcessGroup final : public ProcessGroup {
 public:
  explicit SimulatedAllGatherProcessGroup(
      bool mismatched_request_count = false,
      int64_t remote_second_bitmap_value = 1)
      : ProcessGroup(/*rank=*/0,
                     /*world_size=*/2,
                     torch::Device(torch::kCPU)),
        mismatched_request_count_(mismatched_request_count),
        remote_second_bitmap_value_(remote_second_bitmap_value) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    ++allgather_count_;
    inputs.push_back(input.clone());
    torch::Tensor remote_payload = input.clone();
    if (mismatched_request_count_) {
      remote_payload[0].add_(1);
    } else if (allgather_count_ == 2 && remote_payload.numel() > 1) {
      remote_payload[1].fill_(remote_second_bitmap_value_);
    }
    return torch::cat({input, remote_payload});
  }

  size_t allgather_count() const { return allgather_count_; }

  // Payloads passed to each collective, in call order.
  std::vector<torch::Tensor> inputs;

 private:
  bool mismatched_request_count_;
  int64_t remote_second_bitmap_value_;
  size_t allgather_count_ = 0;
};

ParallelArgs make_parallel_args(ProcessGroup* tp_group) {
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/2,
                             /*dp_size=*/1,
                             /*cp_size=*/1,
                             /*process_group=*/nullptr,
                             /*ep_size=*/1);
  parallel_args.tp_group_ = tp_group;
  return parallel_args;
}

TEST(KVTransferFailureReductionTest, MergesFailureBitmapAcrossTpRanks) {
  SimulatedAllGatherProcessGroup tp_group;
  ParallelArgs parallel_args = make_parallel_args(&tp_group);
  Device device(torch::Device(torch::kCPU));

  const KvPushFailureIds failures = reduce_failed_request_ids(
      {"request-a"}, {}, {"request-a", "request-b"}, parallel_args, device);
  EXPECT_EQ(failures.failed_request_ids,
            (std::vector<std::string>{"request-a", "request-b"}));
  EXPECT_TRUE(failures.unsafe_request_ids.empty());
  EXPECT_EQ(tp_group.allgather_count(), 2u);
}

TEST(KVTransferFailureReductionTest, MergesUnsafeBitmapAcrossTpRanks) {
  SimulatedAllGatherProcessGroup tp_group(
      /*mismatched_request_count=*/false, /*remote_second_bitmap_value=*/3);
  ParallelArgs parallel_args = make_parallel_args(&tp_group);
  Device device(torch::Device(torch::kCPU));

  const KvPushFailureIds failures =
      reduce_failed_request_ids({"request-a", "request-b"},
                                {"request-a"},
                                {"request-a", "request-b"},
                                parallel_args,
                                device);
  EXPECT_EQ(failures.failed_request_ids,
            (std::vector<std::string>{"request-a", "request-b"}));
  EXPECT_EQ(failures.unsafe_request_ids,
            (std::vector<std::string>{"request-a", "request-b"}));
  EXPECT_EQ(tp_group.allgather_count(), 2u);
}

TEST(KVTransferFailureReductionTest,
     TreatsOutOfRangeRankBitmapAsFailedAndUnsafe) {
  // The protocol only emits 0, 1 and 3; 2 (unsafe without failed) and values
  // outside 0-3 cannot be trusted. Treat them as failed-and-unsafe instead of
  // aborting the worker on the post-reduction invariant.
  for (const int64_t invalid_bits : {2, 5}) {
    SCOPED_TRACE(::testing::Message() << "invalid_bits=" << invalid_bits);
    SimulatedAllGatherProcessGroup tp_group(
        /*mismatched_request_count=*/false, invalid_bits);
    ParallelArgs parallel_args = make_parallel_args(&tp_group);
    Device device(torch::Device(torch::kCPU));

    const KvPushFailureIds failures = reduce_failed_request_ids(
        {"request-a"}, {}, {"request-a", "request-b"}, parallel_args, device);
    EXPECT_EQ(failures.failed_request_ids,
              (std::vector<std::string>{"request-a", "request-b"}));
    EXPECT_EQ(failures.unsafe_request_ids,
              (std::vector<std::string>{"request-b"}));
  }
}

TEST(KVTransferFailureReductionTest, TpReductionFeedsCpReduction) {
  // TP and CP form a mesh. The CP collective must carry the TP-reduced
  // bitmap, otherwise a diagonal rank's unsafe report never reaches the
  // driver. Simulate a TP peer that reports unsafe for request-b and a CP
  // peer that has no local failure; the CP input must already contain the TP
  // result.
  SimulatedAllGatherProcessGroup tp_group(
      /*mismatched_request_count=*/false, /*remote_second_bitmap_value=*/3);
  SimulatedAllGatherProcessGroup cp_group(
      /*mismatched_request_count=*/false, /*remote_second_bitmap_value=*/0);
  ParallelArgs parallel_args = make_parallel_args(&tp_group);
  parallel_args.cp_group_ = &cp_group;
  Device device(torch::Device(torch::kCPU));

  const KvPushFailureIds failures = reduce_failed_request_ids(
      {"request-a"}, {}, {"request-a", "request-b"}, parallel_args, device);
  ASSERT_EQ(cp_group.inputs.size(), 2u);
  EXPECT_EQ(cp_group.inputs[1][1].item<int64_t>(), 3);
  EXPECT_EQ(failures.failed_request_ids,
            (std::vector<std::string>{"request-a", "request-b"}));
  EXPECT_EQ(failures.unsafe_request_ids,
            (std::vector<std::string>{"request-b"}));
}

TEST(KVTransferFailureReductionTest,
     EmptyRequestSetUsesFixedMetadataCollective) {
  SimulatedAllGatherProcessGroup tp_group;
  ParallelArgs parallel_args = make_parallel_args(&tp_group);
  Device device(torch::Device(torch::kCPU));

  const KvPushFailureIds failures =
      reduce_failed_request_ids({}, {}, {}, parallel_args, device);
  EXPECT_TRUE(failures.failed_request_ids.empty());
  EXPECT_TRUE(failures.unsafe_request_ids.empty());
  EXPECT_EQ(tp_group.allgather_count(), 1u);
}

TEST(KVTransferFailureReductionTest, RejectsTpRankRequestCountMismatch) {
  SimulatedAllGatherProcessGroup tp_group(/*mismatched_request_count=*/true);
  ParallelArgs parallel_args = make_parallel_args(&tp_group);
  Device device(torch::Device(torch::kCPU));

  EXPECT_DEATH(
      {
        reduce_failed_request_ids({}, {}, {"request-a"}, parallel_args, device);
      },
      "request count differs across reduction ranks");
}

TEST(KVTransferFailureReductionTest, NonSendingRolesSkipEmptyCollectives) {
  for (const InstanceRole role :
       {InstanceRole::DEFAULT, InstanceRole::DECODE}) {
    SimulatedAllGatherProcessGroup tp_group;
    ParallelArgs parallel_args = make_parallel_args(&tp_group);
    Device device(torch::Device(torch::kCPU));
    KVTransferCompletion completion;

    const KvPushFailureIds failures = finalize_kv_push_failures(
        completion, {}, "PUSH", role, parallel_args, device);
    EXPECT_TRUE(failures.failed_request_ids.empty());
    EXPECT_TRUE(failures.unsafe_request_ids.empty());
    EXPECT_EQ(tp_group.allgather_count(), 0u);
  }
}

TEST(KVTransferFailureReductionTest, SendingRolesValidateEmptyTransferSets) {
  for (const InstanceRole role : {InstanceRole::PREFILL, InstanceRole::MIX}) {
    SimulatedAllGatherProcessGroup tp_group;
    ParallelArgs parallel_args = make_parallel_args(&tp_group);
    Device device(torch::Device(torch::kCPU));
    KVTransferCompletion completion;

    const KvPushFailureIds failures = finalize_kv_push_failures(
        completion, {}, "PUSH", role, parallel_args, device);
    EXPECT_TRUE(failures.failed_request_ids.empty());
    EXPECT_TRUE(failures.unsafe_request_ids.empty());
    EXPECT_EQ(tp_group.allgather_count(), 1u);
  }
}

TEST(KVTransferFailureReductionTest, FinalizeReturnsUnsafeSubset) {
  ParallelArgs parallel_args = make_parallel_args(/*tp_group=*/nullptr);
  Device device(torch::Device(torch::kCPU));
  KVTransferCompletion completion;
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  completion.add(promise.getSemiFuture(), {"request-a"});
  promise.setValue(std::vector<KVTransferTaskResult>{
      KVTransferTaskResult{{"request-a"}, KVTransferErrorCode::UNSAFE}});

  TransferKVInfo transfer;
  transfer.request_id = "request-a";
  const KvPushFailureIds failures =
      finalize_kv_push_failures(completion,
                                {transfer},
                                "PUSH",
                                InstanceRole::PREFILL,
                                parallel_args,
                                device);
  EXPECT_EQ(failures.failed_request_ids,
            (std::vector<std::string>{"request-a"}));
  EXPECT_EQ(failures.unsafe_request_ids,
            (std::vector<std::string>{"request-a"}));
}

TEST(KVTransferFailureReductionTest, NonSendingRolesRejectOutgoingTransfers) {
  SimulatedAllGatherProcessGroup tp_group;
  ParallelArgs parallel_args = make_parallel_args(&tp_group);
  Device device(torch::Device(torch::kCPU));
  KVTransferCompletion completion;
  TransferKVInfo transfer;
  transfer.request_id = "request-a";

  EXPECT_DEATH(
      {
        finalize_kv_push_failures(completion,
                                  {transfer},
                                  "PUSH",
                                  InstanceRole::DECODE,
                                  parallel_args,
                                  device);
      },
      "non-sending instance has outgoing KV transfer requests");
}

TEST(KVTransferCompletionTest, ReturnsMergedFailedRequestIds) {
  folly::Promise<std::vector<KVTransferTaskResult>> first_promise;
  folly::Promise<std::vector<KVTransferTaskResult>> second_promise;
  KVTransferCompletion completion;
  completion.add(first_promise.getSemiFuture(), {"request-a", "request-b"});
  completion.add(second_promise.getSemiFuture(), {"request-b", "request-c"});

  first_promise.setValue(std::vector<KVTransferTaskResult>{KVTransferTaskResult{
      {"request-a", "request-b"}, KVTransferErrorCode::FAILED}});
  second_promise.setValue(
      std::vector<KVTransferTaskResult>{KVTransferTaskResult{
          {"request-b", "request-c"}, KVTransferErrorCode::FAILED}});

  EXPECT_EQ(
      completion.wait(),
      (std::unordered_set<std::string>{"request-a", "request-b", "request-c"}));
}

TEST(KVTransferCompletionTest, TracksUnsafeRequestIds) {
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  KVTransferCompletion completion;
  completion.add(promise.getSemiFuture(), {"request-a"});
  promise.setValue(std::vector<KVTransferTaskResult>{
      KVTransferTaskResult{{"request-a"}, KVTransferErrorCode::UNSAFE}});

  EXPECT_EQ(completion.wait(), (std::unordered_set<std::string>{"request-a"}));
  EXPECT_EQ(completion.unsafe_request_ids(),
            (std::unordered_set<std::string>{"request-a"}));
}

TEST(KVTransferCompletionTest, WaitsForEveryTransfer) {
  folly::Promise<std::vector<KVTransferTaskResult>> first_promise;
  folly::Promise<std::vector<KVTransferTaskResult>> second_promise;
  KVTransferCompletion completion;
  completion.add(first_promise.getSemiFuture(), {"request-a"});
  completion.add(second_promise.getSemiFuture(), {"request-b"});

  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  std::future<std::unordered_set<std::string>> result =
      std::async(std::launch::async, [&]() {
        waiter_started.set_value();
        return completion.wait();
      });

  started.wait();
  first_promise.setValue(std::vector<KVTransferTaskResult>{});
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  second_promise.setValue(std::vector<KVTransferTaskResult>{});
  EXPECT_TRUE(result.get().empty());
}

TEST(KVTransferCompletionTest, IsolatesFailedTaskRequests) {
  folly::Promise<std::vector<KVTransferTaskResult>> success_promise;
  folly::Promise<std::vector<KVTransferTaskResult>> failure_promise;
  KVTransferCompletion completion;
  completion.add(success_promise.getSemiFuture(), {"request-success"});
  completion.add(failure_promise.getSemiFuture(),
                 {"request-failed-a", "request-failed-b"});

  success_promise.setValue(std::vector<KVTransferTaskResult>{
      KVTransferTaskResult{{"request-success"}, KVTransferErrorCode::NONE}});
  failure_promise.setValue(std::vector<KVTransferTaskResult>{
      KVTransferTaskResult{{"request-failed-a", "request-failed-b"},
                           KVTransferErrorCode::FAILED}});

  EXPECT_EQ(completion.wait(),
            (std::unordered_set<std::string>{"request-failed-a",
                                             "request-failed-b"}));
}

TEST(KVTransferCompletionTest, ReportsFutureExceptionForFallbackIds) {
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  KVTransferCompletion completion;
  completion.add(promise.getSemiFuture(), {"request-exception"});
  promise.setException(std::runtime_error("transfer failed"));

  EXPECT_EQ(completion.wait(),
            (std::unordered_set<std::string>{"request-exception"}));
}

TEST(KVTransferCompletionTest, DrainsPendingTransferAfterTimeout) {
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  KVTransferCompletion completion(1ms);
  completion.add(promise.getSemiFuture(), {"request-timeout"});

  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  std::future<std::unordered_set<std::string>> result =
      std::async(std::launch::async, [&]() {
        waiter_started.set_value();
        return completion.wait();
      });

  started.wait();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  promise.setValue(std::vector<KVTransferTaskResult>{});
  EXPECT_EQ(result.get(), (std::unordered_set<std::string>{"request-timeout"}));
}

TEST(KVTransferCompletionTest, DrainsPendingTransferAtDestruction) {
  // An exception unwinding the worker step skips wait(); the destructor must
  // drain the pending transfer instead of aborting the process. The transfer
  // completes while the destructor is draining, mirroring an in-flight push
  // that finishes after the step already failed.
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  auto setter = std::async(std::launch::async, [&promise]() {
    std::this_thread::sleep_for(50ms);
    promise.setValue(std::vector<KVTransferTaskResult>{});
  });
  {
    KVTransferCompletion completion;
    completion.add(promise.getSemiFuture(), {"request-pending"});
  }
  setter.get();
  SUCCEED();
}

TEST(KVTransferCompletionTest, DrainsCompletedTransferAtDestruction) {
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  {
    KVTransferCompletion completion;
    completion.add(promise.getSemiFuture(), {"request-pending"});
    promise.setValue(std::vector<KVTransferTaskResult>{
        KVTransferTaskResult{{"request-pending"}, KVTransferErrorCode::NONE}});
  }
  SUCCEED();
}

TEST(KVTransferCompletionTest, PropagatesExceptionAfterDestructionDrain) {
  // The unwinding exception must survive the destructor's drain: the drain
  // itself never throws, so the original failure reaches the handler.
  try {
    folly::Promise<std::vector<KVTransferTaskResult>> promise;
    KVTransferCompletion completion;
    completion.add(promise.getSemiFuture(), {"request-pending"});
    promise.setValue(std::vector<KVTransferTaskResult>{});
    throw std::runtime_error("forward failed");
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "forward failed");
  }
}

// A cancel hook that ignores abort is the extreme tail: the destructor
// cancels, drains the bounded window, then releases the leaked tracker
// token with abandon() instead of hanging the destroying thread.
TEST(KVTransferCompletionTest, AbandonsStuckTransferAtDestruction) {
  // Both counters are process-global and monotonic, so assert the delta of
  // this test's own completion rather than absolute values: --gtest_repeat
  // or a second abandon case in the same binary would otherwise fail.
  const double pending_before =
      COUNTER_VALUE(kv_transfer_destruction_pending_total);
  const double abandoned_before =
      COUNTER_VALUE(kv_transfer_destruction_abandoned_total);
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  {
    KVTransferCompletion completion(1ms);
    completion.add(promise.getSemiFuture(),
                   {"request-stuck"},
                   /*cancel=*/[] {});  // abort that unblocks nothing.
  }
  SUCCEED();  // Destructor cancels, drains 1s, abandons, and returns.
  // The accepted-risk abandon is counted, not just logged.
  EXPECT_GE(COUNTER_VALUE(kv_transfer_destruction_pending_total),
            pending_before + 1.0);
  EXPECT_EQ(COUNTER_VALUE(kv_transfer_destruction_abandoned_total),
            abandoned_before + 1.0);
}

// The cancel hook mirrors the layer-synchronizer abort contract: it unblocks
// the stuck push so the transfer completes during the bounded drain and no
// failure path is needed at all.
TEST(KVTransferCompletionTest, CancelUnblocksStuckTransferAtDestruction) {
  folly::Promise<std::vector<KVTransferTaskResult>> promise;
  auto* promise_ptr = &promise;
  {
    KVTransferCompletion completion(1ms);
    completion.add(promise.getSemiFuture(), {"request-stuck"}, [promise_ptr] {
      // Simulate abort() making the push thread report
      // failure and complete.
      promise_ptr->setValue(
          std::vector<KVTransferTaskResult>{KVTransferTaskResult{
              {"request-stuck"}, KVTransferErrorCode::FAILED}});
    });
  }
  SUCCEED();  // Destructor cancels, drain converges immediately.
}

// An uncancellable stuck transfer keeps the original fail-safe: the
// destructor would hit LOG(FATAL) after the bounded drain. This path is not
// death-tested here because folly's HeapTimekeeper thread does not survive
// the fork used by EXPECT_DEATH, so within() can never fire in the child.

TEST(KVTransferTrackerTest, WaitsForEveryTrackedTransfer) {
  KVTransferTracker tracker;
  std::shared_ptr<KVTransferTracker::Completion> first = tracker.track();
  std::shared_ptr<KVTransferTracker::Completion> second = tracker.track();

  std::promise<void> waiter_started;
  std::future<void> started = waiter_started.get_future();
  std::future<void> result = std::async(std::launch::async, [&]() {
    waiter_started.set_value();
    tracker.wait();
  });

  started.wait();
  first.reset();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  second.reset();
  EXPECT_EQ(result.wait_for(1s), std::future_status::ready);
}

TEST(KVTransferTrackerTest, ReportsPendingUntilEveryTransferFinishes) {
  KVTransferTracker tracker;
  EXPECT_FALSE(tracker.has_pending());

  std::shared_ptr<KVTransferTracker::Completion> first = tracker.track();
  std::shared_ptr<KVTransferTracker::Completion> second = tracker.track();
  EXPECT_TRUE(tracker.has_pending());

  first.reset();
  EXPECT_TRUE(tracker.has_pending());

  second.reset();
  EXPECT_FALSE(tracker.has_pending());
}

TEST(KVTransferTrackerTest, WaitForHasBoundedTimeout) {
  KVTransferTracker tracker;
  std::shared_ptr<KVTransferTracker::Completion> completion = tracker.track();

  EXPECT_FALSE(tracker.wait_for(1ms));
  completion.reset();
  EXPECT_TRUE(tracker.wait_for(1s));
}

TEST(KVTransferTrackerTest, DestructionWaitsForTrackedTransfer) {
  auto tracker = std::make_unique<KVTransferTracker>();
  std::shared_ptr<KVTransferTracker::Completion> completion = tracker->track();

  std::promise<void> destruction_started;
  std::future<void> started = destruction_started.get_future();
  std::future<void> result = std::async(
      std::launch::async,
      [tracker = std::move(tracker), &destruction_started]() mutable {
        destruction_started.set_value();
        tracker.reset();
      });

  started.wait();
  EXPECT_EQ(result.wait_for(50ms), std::future_status::timeout);
  completion.reset();
  EXPECT_EQ(result.wait_for(1s), std::future_status::ready);
}

}  // namespace
}  // namespace xllm
