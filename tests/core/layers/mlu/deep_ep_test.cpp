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

#include "layers/common/deep_ep.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <array>
#include <barrier>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "framework/parallel_state/parallel_args.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "tests/scoped_port_range.h"
#include "util/tensor_helper.h"

#if defined(USE_MLU)
#include "framework/parallel_state/mlu_process_group.h"
#elif defined(USE_CUDA)
#include "framework/parallel_state/cuda_process_group.h"
#endif

namespace xllm {
namespace layer {
namespace test {

// Special exit code definition for skipping test
constexpr int32_t EXIT_CODE_SKIP = 77;

// Helper function to create ProcessGroup
std::unique_ptr<xllm::ProcessGroup> create_test_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t port,
    const std::string& host,
    const torch::Device& device) {
  return xllm::create_process_group(rank,
                                    world_size,
                                    world_size,
                                    port,
                                    false,
                                    host,
                                    "deep_ep_test_group",
                                    device);
}

struct TestParams {
  int32_t rank;
  int32_t world_size;
  int32_t port;
  std::string host;
  int32_t device_index;
  int64_t token_size;
  int64_t max_tokens;
  int64_t num_experts;
  int64_t draft_experts;
};

namespace {
// ProcessGroupCNCL initializes one local device per clique. Threaded ranks
// exchange only bootstrap metadata on the host; DeepEP runs on both devices.
class ThreadRankGroup final : public ProcessGroup {
 public:
  ThreadRankGroup(int32_t rank,
                  const torch::Device& device,
                  std::barrier<>& ready,
                  std::array<torch::Tensor, 2>& inputs)
      : ProcessGroup(rank, inputs.size(), device),
        ready_(ready),
        inputs_(inputs) {}

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override {
    inputs_[rank()] = input.cpu();
    ready_.arrive_and_wait();
    for (size_t i = 0; i < inputs_.size(); ++i) {
      outputs.at(i).copy_(inputs_[i]);
    }
    ready_.arrive_and_wait();
  }

 private:
  std::barrier<>& ready_;
  std::array<torch::Tensor, 2>& inputs_;
};

void check_round_trip(DeepEPImpl& deep_ep,
                      int64_t num_experts,
                      const TestParams& params,
                      const torch::TensorOptions& options,
                      xllm::Device& xllm_device) {
  const int64_t hidden_dim = params.token_size / sizeof(float);
  constexpr int64_t kTokensPerDest = 2;
  const int64_t num_tokens_sent = params.world_size * kTokensPerDest;

  // Prepare Layout
  torch::Tensor token_count_slice = torch::zeros(
      {num_experts},
      torch::TensorOptions().dtype(torch::kInt32).device(options.device()));

  const int64_t num_experts_per_rank = num_experts / params.world_size;
  for (int32_t dest = 0; dest < params.world_size; ++dest) {
    token_count_slice[dest * num_experts_per_rank +
                      params.rank % num_experts_per_rank] = kTokensPerDest;
  }
  const float base_val = static_cast<float>(params.rank * num_tokens_sent);

  // Step 1: Prepare Input Data
  auto buffers = deep_ep.get_buffer();

  torch::Tensor src_data = torch::zeros(
      {num_tokens_sent, hidden_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  for (int64_t i = 0; i < num_tokens_sent; ++i) {
    src_data[i].fill_(base_val + i);
  }

  torch::Tensor src_data_bytes =
      src_data.to(options.device()).view({-1}).view(torch::kUInt8);
  CHECK(src_data_bytes.numel() <= buffers.dispatch_send_token_tensor.numel());
  buffers.dispatch_send_token_tensor.view({-1})
      .slice(0, 0, src_data_bytes.numel())
      .copy_(src_data_bytes);

  // Step 2: Dispatch Step
  deep_ep.dispatch_step(num_tokens_sent, token_count_slice);

  // Step 3: Process Dispatch Result
  int64_t max_recv_capacity = params.max_tokens * params.world_size;

  // Allocate Expert Input as Float
  torch::Tensor expert_input =
      torch::zeros({max_recv_capacity, hidden_dim}, options);

  // Convert to Int8 View to match DeepEP internal buffer dtype
  // DeepEP buffer is Int8, so we must pass a Int8 tensor to gather_split
  torch::Tensor expert_input_byte = view_as_dtype(expert_input, torch::kInt8);

  auto meta = deep_ep.process_dispatch_result(
      num_experts_per_rank, expert_input_byte, std::nullopt);

  // Validate count
  torch::Tensor valid_token_num_cpu = meta.token_sum.to(torch::kCPU);
  int64_t received_count = valid_token_num_cpu.item<int64_t>();
  CHECK_EQ(received_count, num_tokens_sent)
      << "Rank " << params.rank << ": Unexpected received token count "
      << received_count;

  // Step 4: Simulate Computation (on Float tensor)
  torch::Tensor expert_output =
      expert_input + static_cast<float>(params.rank + 1);

  // Step 5: Combine Pack
  // Convert to Int8 View to match DeepEP internal buffer dtype
  torch::Tensor expert_output_byte = view_as_dtype(expert_output, torch::kInt8);

  torch::Tensor combine_layout =
      deep_ep.combine_step_pack(expert_output_byte,
                                meta.gather_rank_index,
                                meta.token_sum,
                                params.token_size,
                                torch::kInt8);

  // Step 6: Combine Comm
  torch::Tensor final_output = deep_ep.combine_step_comm(
      combine_layout, num_tokens_sent, hidden_dim, torch::kFloat32);

  // Step 7: Verification
  xllm_device.synchronize_default_stream();

  torch::Tensor final_cpu = final_output.to(torch::kCPU);
  auto final_acc = final_cpu.accessor<float, 2>();

  for (int64_t i = 0; i < num_tokens_sent; ++i) {
    float expected_val =
        base_val + i + static_cast<float>(i / kTokensPerDest + 1);
    for (int64_t j = 0; j < hidden_dim; ++j) {
      float got = final_acc[i][j];
      CHECK(std::abs(got - expected_val) <= 1e-4)
          << "Rank " << params.rank << " Verification Failed at [" << i << "]["
          << j << "]: Expected " << expected_val << ", Got " << got;
    }
  }
}
}  // namespace

// Child process test function
int32_t run_deep_ep_test_child(TestParams params) {
  ::alarm(120);
  try {
    // 1. Check devices
    int32_t dev_count = xllm::Platform::device_count();
    if (dev_count < params.world_size) {
      LOG(WARNING) << "Rank " << params.rank
                   << ": Insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }
    params.device_index = params.rank % dev_count;

    // 2. Set device
    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    // 3. Create ProcessGroup
    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);

    CHECK(process_group) << "Rank " << params.rank
                         << ": Failed to create ProcessGroup";

    // 4. Create ParallelArgs
    ParallelArgs parallel_args(
        params.rank, params.world_size, process_group.get());
    parallel_args.moe_ep_group_ = process_group.get();
    parallel_args.ep_size_ = params.world_size;

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(device)
                       .requires_grad(false);

    auto get_instance = [&](int64_t num_experts) {
      return DeepEPManager::get_instance(params.token_size,
                                         params.token_size,
                                         params.max_tokens,
                                         num_experts,
                                         parallel_args,
                                         options);
    };
    auto target = get_instance(params.num_experts);
    auto draft = get_instance(params.draft_experts);
    auto target_again = get_instance(params.num_experts);
    CHECK_EQ(target.ptr().get(), target_again.ptr().get());
    check_round_trip(*target, params.num_experts, params, options, xllm_device);
    check_round_trip(
        *draft, params.draft_experts, params, options, xllm_device);
    check_round_trip(
        *target_again, params.num_experts, params, options, xllm_device);
    LOG(INFO) << "Rank " << params.rank << ": E2E Test Passed!";
    return 0;

  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

// Multi-process test fixture
class DeepEPMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    world_size_ = 2;
    port_ = xllm::tests::ScopedPortRange().release();
    host_ = "127.0.0.1";
    token_size_ = 128;
    max_tokens_ = 64;
    num_global_experts_ = 4;
    draft_experts_ = 2;
  }

  void run_test() {
    std::vector<pid_t> child_pids;
    child_pids.reserve(world_size_);

    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        TestParams params;
        params.rank = rank;
        params.world_size = world_size_;
        params.port = port_;
        params.host = host_;
        params.device_index = -1;
        params.token_size = token_size_;
        params.max_tokens = max_tokens_;
        params.num_experts = num_global_experts_;
        params.draft_experts = draft_experts_;

        int32_t exit_code = run_deep_ep_test_child(params);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        for (pid_t started : child_pids) {
          ::kill(started, SIGKILL);
        }
        for (pid_t started : child_pids) {
          int32_t status = 0;
          while (::waitpid(started, &status, 0) < 0 && errno == EINTR) {
          }
        }
        FAIL() << "Failed to fork rank " << rank;
      }
    }

    bool any_failed = false;
    int32_t skip_count = 0;

    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status = 0;
      pid_t waited;
      do {
        waited = ::waitpid(child_pids[i], &status, 0);
      } while (waited < 0 && errno == EINTR);
      if (waited == child_pids[i] && WIFEXITED(status)) {
        int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          ++skip_count;
        } else if (exit_code != 0) {
          any_failed = true;
          LOG(ERROR) << "Rank " << i << " failed with code " << exit_code;
        }
      } else {
        any_failed = true;
        LOG(ERROR) << "Rank " << i << " crashed (signal).";
      }
    }

    if (!any_failed && skip_count == world_size_) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    }
    ASSERT_FALSE(any_failed) << "DeepEP End-to-End Test Failed.";
    EXPECT_EQ(skip_count, 0)
        << "Only some ranks skipped; collective is invalid.";
  }

  int32_t world_size_;
  int32_t port_;
  std::string host_;
  int64_t token_size_;
  int64_t max_tokens_;
  int64_t num_global_experts_;
  int64_t draft_experts_;
};

TEST_F(DeepEPMultiDeviceTest, EndToEndFlow) { run_test(); }

TEST_F(DeepEPMultiDeviceTest, ThreadedRanksRecreateExpertGroup) {
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::alarm(120);
    if (xllm::Platform::device_count() < world_size_) {
      _exit(EXIT_CODE_SKIP);
    }
    std::barrier ready(world_size_);
    std::array<torch::Tensor, 2> exchange_info;
    auto run_rank = [&](int32_t rank) {
      xllm::Device device(rank);
      device.set_device();
      auto options =
          torch::TensorOptions().device(device.unwrap()).dtype(torch::kFloat32);
      for (int32_t generation = 0; generation < 2; ++generation) {
        TestParams params{rank,
                          world_size_,
                          port_,
                          host_,
                          rank,
                          token_size_,
                          max_tokens_,
                          num_global_experts_,
                          draft_experts_};
        auto group = std::make_unique<ThreadRankGroup>(
            rank, device.unwrap(), ready, exchange_info);
        std::weak_ptr<DeepEPImpl> previous;
        {
          ParallelArgs parallel(rank, world_size_, group.get());
          parallel.moe_ep_group_ = group.get();
          parallel.ep_size_ = world_size_;
          ready.arrive_and_wait();
          auto instance = DeepEPManager::get_instance(token_size_,
                                                      token_size_,
                                                      max_tokens_,
                                                      num_global_experts_,
                                                      parallel,
                                                      options);
          previous = instance.ptr();
          check_round_trip(
              *instance, num_global_experts_, params, options, device);
        }
        CHECK(!previous.expired());
        ready.arrive_and_wait();
        group.reset();
        CHECK(previous.expired()) << "DeepEP outlived its process group";
        ready.arrive_and_wait();
      }
    };
    std::thread peer(run_rank, 1);
    run_rank(0);
    peer.join();
    _exit(0);
  }
  int32_t status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  ASSERT_EQ(waited, child);
  ASSERT_TRUE(WIFEXITED(status))
      << "Child terminated with wait status " << status;
  if (WEXITSTATUS(status) == EXIT_CODE_SKIP) {
    GTEST_SKIP() << "Two devices required";
  }
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST_F(DeepEPMultiDeviceTest, TargetAndDraftShareExpertGroup) {
  world_size_ = 8;
  num_global_experts_ = 384;
  draft_experts_ = 128;
  run_test();
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
