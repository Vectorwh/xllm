/* Copyright 2026 The xLLM Authors.

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

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

#include "core/platform/npu/npu_profiler.h"
#include "tests/npu_test_environment.h"

namespace {

namespace fs = std::filesystem;

class NpuProfilerTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    xllm::testing::init_npu_test_runtime();
    google::InitGoogleLogging("npu_profiler_test");
    google::SetStderrLogging(google::INFO);

    const int32_t ret = aclrtSetDevice(0);
    if (ret != 0) {
      LOG(ERROR) << "ACL set device id: 0 failed, ret:" << ret;
    }
    torch_npu::init_npu("npu:0");
  }

  void TearDown() override {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
    google::ShutdownGoogleLogging();
    xllm::testing::finalize_npu_test_runtime();
  }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new NpuProfilerTestEnvironment);

fs::path make_temp_dir(const std::string& tag) {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const int64_t ts =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  fs::path dir = fs::temp_directory_path() /
                 ("xllm_npu_profiler_test_" + tag + "_" +
                  std::to_string(::getpid()) + "_" + std::to_string(ts));
  fs::create_directories(dir);
  return dir;
}

}  // namespace

TEST(NpuProfilerTest, StartStopWritesAscendPtDir) {
  auto& profiler = xllm::NpuProfiler::get_instance();
  ASSERT_FALSE(profiler.is_running());

  fs::path dir = make_temp_dir("write");
  constexpr int32_t kRank = 7;
  ASSERT_TRUE(profiler.start(dir.string(), kRank));
  EXPECT_TRUE(profiler.is_running());
  // Second start() while already running must be a no-op success.
  EXPECT_TRUE(profiler.start(dir.string(), kRank));
  EXPECT_TRUE(profiler.is_running());

  // Give the CANN AclProfiler something to record so the trace has at least
  // one host-side op and one NPU-side kernel event.
  {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device("npu:0");
    auto a = torch::ones({16}, opts);
    auto b = a + 1.0f;
    (void)b.cpu();
  }

  ASSERT_TRUE(profiler.stop());
  EXPECT_FALSE(profiler.is_running());

  bool has_rank_trace = false;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("xllm_rank7_", 0) != 0) {
      continue;
    }
    const fs::path trace =
        entry.path() / "ASCEND_PROFILER_OUTPUT" / "trace_view.json";
    std::error_code error;
    const uintmax_t bytes = fs::file_size(trace, error);
    has_rank_trace |= !error && bytes > 0;
  }
  EXPECT_TRUE(has_rank_trace) << "no non-empty rank7 trace under " << dir;

  fs::remove_all(dir);
}
