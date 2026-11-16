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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "core/platform/numa_utils.h"

namespace {

class NumaUtilsTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("numa_utils_test");
    google::SetStderrLogging(google::INFO);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new NumaUtilsTestEnvironment);

bool SaveCurrentThreadAffinity(cpu_set_t* cpu_set) {
  if (cpu_set == nullptr) {
    return false;
  }
  CPU_ZERO(cpu_set);
  return pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), cpu_set) ==
         0;
}

bool RestoreCurrentThreadAffinity(const cpu_set_t& cpu_set) {
  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) ==
         0;
}

bool CpuInList(int cpu, const std::vector<int32_t>& cpus) {
  return std::find(cpus.begin(), cpus.end(), cpu) != cpus.end();
}

bool HasHipDevice() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

// xllm::numa APIs under test: is_numa_available, get_num_numa_nodes
class NumaUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_TRUE(SaveCurrentThreadAffinity(&saved_)); }
  void TearDown() override {
    EXPECT_TRUE(RestoreCurrentThreadAffinity(saved_));
  }

 private:
  cpu_set_t saved_;
};

TEST_F(NumaUtilsTest, AvailabilityAndNodeCountAreConsistent) {
  const bool available = xllm::numa::is_numa_available();
  const int32_t num_nodes = xllm::numa::get_num_numa_nodes();

  if (!available) {
    EXPECT_EQ(num_nodes, -1);
  } else {
    EXPECT_GT(num_nodes, 0);
  }
}

// xllm::numa APIs under test: is_numa_available, get_current_numa_node,
// get_num_numa_nodes
TEST_F(NumaUtilsTest, CurrentNumaNodeMatchesAvailability) {
  const bool available = xllm::numa::is_numa_available();
  const int32_t current_node = xllm::numa::get_current_numa_node();

  if (!available) {
    EXPECT_EQ(current_node, -1);
  } else {
    EXPECT_GE(current_node, 0);
    EXPECT_LT(current_node, xllm::numa::get_num_numa_nodes());
  }
}

// xllm::numa APIs under test: get_numa_node_cpus,
// bind_thread_to_numa_node,bind_process_to_numa_node
TEST_F(NumaUtilsTest, InvalidNodeQueriesReturnEmptyOrError) {
  const bool available = xllm::numa::is_numa_available();

  if (!available) {
    GTEST_SKIP() << "NUMA is not available on this system";
  }

  EXPECT_TRUE(xllm::numa::get_numa_node_cpus(-1).empty());
  EXPECT_TRUE(
      xllm::numa::get_numa_node_cpus(xllm::numa::get_num_numa_nodes()).empty());

  EXPECT_NE(xllm::numa::bind_thread_to_numa_node(-1), 0);
  EXPECT_NE(xllm::numa::bind_process_to_numa_node(-1), 0);
  EXPECT_NE(
      xllm::numa::bind_thread_to_numa_node(xllm::numa::get_num_numa_nodes()),
      0);
  EXPECT_NE(
      xllm::numa::bind_process_to_numa_node(xllm::numa::get_num_numa_nodes()),
      0);
}

// xllm::numa APIs under test: get_numa_node_cpus, bind_thread_to_numa_node
TEST_F(NumaUtilsTest, ValidNodeCpuListLooksReasonable) {
  if (!xllm::numa::is_numa_available()) {
    GTEST_SKIP() << "NUMA is unavailable";
  }
  cpu_set_t allowed;
  ASSERT_TRUE(SaveCurrentThreadAffinity(&allowed));
  for (int32_t node = 0; node < xllm::numa::get_num_numa_nodes(); ++node) {
    const auto cpus = xllm::numa::get_numa_node_cpus(node);
    const std::unordered_set<int32_t> unique(cpus.begin(), cpus.end());
    EXPECT_EQ(unique.size(), cpus.size());
    for (int32_t cpu : cpus) {
      ASSERT_GE(cpu, 0);
      ASSERT_LT(cpu, CPU_SETSIZE);
      EXPECT_TRUE(CPU_ISSET(cpu, &allowed));
    }
  }
}

TEST_F(NumaUtilsTest, BuildCpuSetRespectsProcessAffinityConstraints) {
  const bool available = xllm::numa::is_numa_available();

  if (!available) {
    GTEST_SKIP() << "NUMA is not available on this system";
  }

  const int32_t num_nodes = xllm::numa::get_num_numa_nodes();
  if (num_nodes < 2) {
    GTEST_SKIP() << "Need >=2 NUMA nodes for intersection test";
  }

  std::vector<int32_t> node0_cpus = xllm::numa::get_numa_node_cpus(0);
  if (node0_cpus.empty()) {
    GTEST_SKIP() << "Node 0 has no CPUs";
  }

  // Save the original process affinity so we can restore it.
  cpu_set_t saved_affinity;
  CPU_ZERO(&saved_affinity);
  ASSERT_EQ(sched_getaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);

  // Restrict this process to a single CPU from node 0, then query node 1
  // (which should have no overlap with node 0's CPUs).
  cpu_set_t restricted;
  CPU_ZERO(&restricted);
  CPU_SET(node0_cpus.back(), &restricted);
  ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &restricted), 0);

  std::vector<int32_t> node1_cpus = xllm::numa::get_numa_node_cpus(1);
  EXPECT_TRUE(node1_cpus.empty())
      << "Node 1 CPUs should be empty under affinity restriction";

  EXPECT_NE(xllm::numa::bind_thread_to_numa_node(1), 0);

  ASSERT_EQ(sched_setaffinity(getpid(), sizeof(cpu_set_t), &saved_affinity), 0);
}

// xllm::numa APIs under test: bind_process_to_numa_node,
// bind_thread_to_numa_node, get_current_numa_node, get_numa_node_cpus,
// is_numa_available
//
// NOTE: bind_process_to_numa_node also calls numa_set_membind / numa_set_strict
// which are process-global memory policy settings with no standard undo API.
TEST_F(NumaUtilsTest, BindProcessToValidNodeSucceeds) {
  if (!xllm::numa::is_numa_available()) {
    GTEST_SKIP() << "NUMA is unavailable";
  }
  int32_t target = -1;
  std::vector<int32_t> expected;
  for (int32_t node = 0; node < xllm::numa::get_num_numa_nodes(); ++node) {
    auto cpus = xllm::numa::get_numa_node_cpus(node);
    if (!cpus.empty()) {
      target = node;
      expected = std::move(cpus);
      break;
    }
  }
  if (target < 0) {
    GTEST_SKIP() << "No allowed NUMA CPUs";
  }
  // NUMA memory policy has no portable undo; isolate it from later cases.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_EXIT(
      {
        auto bind = [&] {
          ASSERT_EQ(xllm::numa::bind_process_to_numa_node(target), 0);
          cpu_set_t actual;
          ASSERT_TRUE(SaveCurrentThreadAffinity(&actual));
          for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            EXPECT_EQ(CPU_ISSET(cpu, &actual) != 0, CpuInList(cpu, expected));
          }
        };
        bind();
        std::exit(::testing::Test::HasFailure() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS),
      "");
}

TEST_F(NumaUtilsTest, BindThreadToNumaNodeSucceeds) {
  if (!xllm::numa::is_numa_available()) {
    GTEST_SKIP() << "NUMA is unavailable";
  }
  int32_t target = -1;
  std::vector<int32_t> expected;
  for (int32_t node = 0; node < xllm::numa::get_num_numa_nodes(); ++node) {
    auto cpus = xllm::numa::get_numa_node_cpus(node);
    if (!cpus.empty()) {
      target = node;
      expected = std::move(cpus);
      break;
    }
  }
  if (target < 0) {
    GTEST_SKIP() << "No allowed NUMA CPUs";
  }
  ASSERT_EQ(xllm::numa::bind_thread_to_numa_node(target), 0);
  cpu_set_t actual;
  ASSERT_TRUE(SaveCurrentThreadAffinity(&actual));
  for (int32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    EXPECT_EQ(CPU_ISSET(cpu, &actual) != 0, CpuInList(cpu, expected));
  }
}

TEST_F(NumaUtilsTest, GetDeviceNumaNodeReturnsValidNodeOrMinusOneForHipDevice) {
  if (!HasHipDevice()) {
    GTEST_SKIP() << "No HIP/DCU device available";
  }

  const bool available = xllm::numa::is_numa_available();
  const int32_t num_numa_nodes = xllm::numa::get_num_numa_nodes();
  int device_count = 0;
  ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
  ASSERT_GT(device_count, 0);

  for (int device_id = 0; device_id < device_count; ++device_id) {
    const int32_t numa_node = xllm::numa::get_device_numa_node(device_id);

    if (!available) {
      EXPECT_EQ(numa_node, -1);
    } else {
      // Some environments may not expose sysfs; -1 is acceptable.
      if (numa_node != -1) {
        EXPECT_GE(numa_node, 0);
        EXPECT_LT(numa_node, num_numa_nodes);
      }
    }
  }
}

}  // namespace
