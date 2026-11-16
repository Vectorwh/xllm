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

#include "core/util/blocking_counter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>

#include "core/util/threadpool.h"

namespace xllm {
namespace {

TEST(BlockingCounterTest, MultiThreadTest) {
  BlockingCounter counter(5);
  std::atomic<int32_t> called{0};
  ThreadPool pool(4);
  for (int32_t index = 0; index < 4; ++index) {
    pool.schedule([&counter, &called] {
      ++called;
      counter.decrement_count();
    });
  }
  ++called;
  counter.decrement_count();
  EXPECT_TRUE(counter.wait_for(std::chrono::seconds(5)));
  EXPECT_EQ(called.load(), 5);
}

TEST(BlockingCounterTest, WaitTimeoutTest) {
  BlockingCounter counter(1);
  EXPECT_FALSE(counter.wait_for(std::chrono::milliseconds(1)));
  counter.decrement_count();
  EXPECT_TRUE(counter.wait_for(std::chrono::milliseconds(0)));
  counter.wait();
}

}  // namespace
}  // namespace xllm
