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

#include "core/util/threadpool.h"

#include <gtest/gtest.h>
#include <pthread.h>
#include <sched.h>

#include <array>
#include <atomic>
#include <set>
#include <thread>
#include <vector>

namespace xllm {
namespace {

TEST(ThreadPoolTest, ScheduleMultipleTasks) {
  std::vector<int32_t> completed;
  {
    ThreadPool pool(1);
    EXPECT_EQ(pool.schedule(nullptr), -1);
    pool.schedule([&completed] { completed.emplace_back(1); });
    pool.schedule([&completed] { completed.emplace_back(2); });
  }
  EXPECT_EQ(completed, (std::vector<int32_t>{1, 2}));
}

TEST(ThreadPoolTest, MultipleThreads) {
  std::array<std::thread::id, 8> workers;
  std::atomic<int32_t> completed{0};
  {
    ThreadPool pool(4);
    for (size_t index = 0; index < workers.size(); ++index) {
      pool.schedule([&workers, &completed, index] {
        workers[index] = std::this_thread::get_id();
        ++completed;
      });
    }
  }
  for (const auto& worker : workers) {
    EXPECT_NE(worker, std::thread::id{});
  }
  EXPECT_EQ(completed.load(), workers.size());
  const std::set<std::thread::id> distinct(workers.begin(), workers.end());
  EXPECT_GT(distinct.size(), 1);
}

TEST(ThreadPoolTest, CpuCoreBindingVerifyAffinity) {
  cpu_set_t allowed_cpus;
  CPU_ZERO(&allowed_cpus);
  ASSERT_EQ(pthread_getaffinity_np(
                pthread_self(), sizeof(allowed_cpus), &allowed_cpus),
            0);
  int32_t target_core = 0;
  while (target_core < CPU_SETSIZE && !CPU_ISSET(target_core, &allowed_cpus)) {
    ++target_core;
  }
  ASSERT_LT(target_core, CPU_SETSIZE);

  bool initialized = false;
  bool affinity_ok = false;
  {
    ThreadPool pool(
        1,
        [&initialized] { initialized = true; },
        std::vector<int32_t>{target_core});
    pool.schedule([&affinity_ok, target_core] {
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      affinity_ok =
          pthread_getaffinity_np(pthread_self(), sizeof(cpus), &cpus) == 0 &&
          CPU_ISSET(target_core, &cpus) && CPU_COUNT(&cpus) == 1;
    });
  }
  EXPECT_TRUE(initialized);
  EXPECT_TRUE(affinity_ok);
}

}  // namespace
}  // namespace xllm
