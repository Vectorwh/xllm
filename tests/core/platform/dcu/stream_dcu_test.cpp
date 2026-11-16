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

#include <c10/hip/HIPStream.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <thread>

#include "core/platform/stream.h"

namespace {

class StreamDcuTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("stream_dcu_test");
    google::SetStderrLogging(google::INFO);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new StreamDcuTestEnvironment);

// check dcu device
constexpr int kDeviceId = 0;

bool HasHipDevice() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

void PrepareDeviceOrSkip() {
  if (!HasHipDevice()) {
    GTEST_SKIP() << "No DCU/HIP device available";
  }
  ASSERT_EQ(hipSetDevice(kDeviceId), hipSuccess);
  ASSERT_EQ(hipFree(nullptr), hipSuccess);
}

// xllm::Stream APIs under test: Stream(HIPStream), get_stream,
// set_stream_guard, synchronize
class StreamDcuTest : public ::testing::Test {
 protected:
  void SetUp() override { PrepareDeviceOrSkip(); }
};

TEST_F(StreamDcuTest, ConstructFromExistingCurrentStreamWorks) {
  constexpr size_t kNumBytes = 1 << 16;
  unsigned char* dev_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);

  c10::hip::HIPStream current = c10::hip::getCurrentHIPStream(kDeviceId);
  xllm::Stream wrapped(current);

  ASSERT_NE(wrapped.get_stream(), nullptr);
  EXPECT_EQ(*wrapped.get_stream(), current);

  auto guard = wrapped.set_stream_guard();
  EXPECT_EQ(*wrapped.get_stream(), c10::hip::getCurrentHIPStream(kDeviceId));

  ASSERT_EQ(
      hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, wrapped.get_stream()->stream()),
      hipSuccess);

  EXPECT_EQ(wrapped.synchronize(), 0);

  std::vector<unsigned char> host(kNumBytes);
  ASSERT_EQ(hipMemcpy(host.data(), dev_ptr, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x5A) << "Mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

// xllm::Stream APIs under test: Stream(), set_stream_guard, synchronize

// xllm::Stream APIs under test: Stream(), set_stream_guard, synchronize
TEST_F(StreamDcuTest, StreamGuardSwitchesAndRestoresCurrentStream) {
  const auto original = c10::hip::getCurrentHIPStream(kDeviceId);
  xllm::Stream first, second, third;
  {
    auto first_guard = first.set_stream_guard();
    EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId), *first.get_stream());
    {
      auto second_guard = second.set_stream_guard();
      EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId), *second.get_stream());
      {
        auto third_guard = third.set_stream_guard();
        EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId),
                  *third.get_stream());
      }
      EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId), *second.get_stream());
    }
    EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId), *first.get_stream());
  }
  EXPECT_EQ(c10::hip::getCurrentHIPStream(kDeviceId), original);
}

// xllm::Stream APIs under test: Stream(Stream&&), get_stream, synchronize
TEST_F(StreamDcuTest, MoveConstructPreservesPendingOperations) {
  constexpr size_t kNumBytes = 1 << 20;
  unsigned char* dev_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMemset(dev_ptr, 0, kNumBytes), hipSuccess);

  xllm::Stream s1;
  ASSERT_NE(s1.get_stream(), nullptr);
  c10::hip::HIPStream s1_stream = *s1.get_stream();

  ASSERT_EQ(hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, s1_stream.stream()),
            hipSuccess);

  xllm::Stream s2(std::move(s1));
  ASSERT_NE(s2.get_stream(), nullptr);
  EXPECT_EQ(*s2.get_stream(), s1_stream);

  EXPECT_EQ(s2.synchronize(), 0);

  std::vector<unsigned char> host(kNumBytes);
  ASSERT_EQ(hipMemcpy(host.data(), dev_ptr, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x5A) << "Data lost after move-construct at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);

  // Both stream objects leave scope exactly once.
}

// xllm::Stream APIs under test: Stream(), get_stream, wait_stream, synchronize
TEST_F(StreamDcuTest, WaitStreamEstablishesOrdering) {
  xllm::Stream producer;
  xllm::Stream consumer;

  hipStream_t producer_stream = producer.get_stream()->stream();
  hipStream_t consumer_stream = consumer.get_stream()->stream();

  ASSERT_NE(producer_stream, nullptr);
  ASSERT_NE(consumer_stream, nullptr);
  ASSERT_NE(producer_stream, consumer_stream);

  constexpr size_t kNumBytes = 1 << 20;
  unsigned char* dev_ptr = nullptr;
  unsigned char* host_ptr = nullptr;

  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);
  ASSERT_EQ(hipHostMalloc(reinterpret_cast<void**>(&host_ptr), kNumBytes, 0),
            hipSuccess);

  std::memset(host_ptr, 0, kNumBytes);

  ASSERT_EQ(hipMemset(dev_ptr, 0, kNumBytes), hipSuccess);
  std::atomic<bool> gate_open{false};
  ASSERT_EQ(hipLaunchHostFunc(
                producer_stream,
                [](void* data) {
                  auto* gate = static_cast<std::atomic<bool>*>(data);
                  while (!gate->load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                  }
                },
                &gate_open),
            hipSuccess);
  const hipError_t queued =
      hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, producer_stream);
  if (queued != hipSuccess) {
    gate_open.store(true, std::memory_order_release);
    EXPECT_EQ(producer.synchronize(), 0);
  }
  ASSERT_EQ(queued, hipSuccess);

  auto readback = std::async(std::launch::async, [&] {
    if (hipSetDevice(kDeviceId) != hipSuccess) {
      return false;
    }
    consumer.wait_stream(producer);
    if (hipMemcpyAsync(host_ptr,
                       dev_ptr,
                       kNumBytes,
                       hipMemcpyDeviceToHost,
                       consumer_stream) != hipSuccess) {
      return false;
    }
    return consumer.synchronize() == 0;
  });
  const auto before_release = readback.wait_for(std::chrono::milliseconds(100));
  gate_open.store(true, std::memory_order_release);
  EXPECT_EQ(before_release, std::future_status::timeout);
  EXPECT_TRUE(readback.get());
  EXPECT_EQ(producer.synchronize(), 0);

  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host_ptr[i], 0x5A) << "Mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
  ASSERT_EQ(hipHostFree(host_ptr), hipSuccess);
}

// xllm::Stream APIs under test: operator<<

// xllm::Stream APIs under test: set_stream_guard

// xllm::Stream APIs under test: wait_stream, synchronize
TEST_F(StreamDcuTest, WaitStreamSelfIsSafe) {
  constexpr size_t kNumBytes = 1 << 20;
  unsigned char* dev_ptr = nullptr;
  ASSERT_EQ(hipMalloc(&dev_ptr, kNumBytes), hipSuccess);
  ASSERT_EQ(hipMemset(dev_ptr, 0, kNumBytes), hipSuccess);

  xllm::Stream s;
  hipStream_t stream = s.get_stream()->stream();

  ASSERT_EQ(hipMemsetAsync(dev_ptr, 0x5A, kNumBytes, stream), hipSuccess);
  s.wait_stream(s);
  EXPECT_EQ(s.synchronize(), 0);

  std::vector<unsigned char> host(kNumBytes);
  ASSERT_EQ(hipMemcpy(host.data(), dev_ptr, kNumBytes, hipMemcpyDeviceToHost),
            hipSuccess);
  for (size_t i = 0; i < kNumBytes; ++i) {
    ASSERT_EQ(host[i], 0x5A) << "Data mismatch at byte " << i;
  }

  ASSERT_EQ(hipFree(dev_ptr), hipSuccess);
}

}  // namespace
