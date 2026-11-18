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

#include "core/platform/device.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <future>
#include <latch>
#include <memory>

#include "core/platform/musa/musa_layer_synchronizer.h"
#include "core/platform/platform.h"
#include "util/scope_guard.h"

namespace xllm {
namespace {

TEST(MusaDeviceTest, InitializesNativeRuntimeBeforeTensorAllocation) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }

  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  // Registration and allocator initialization must also be safe to repeat.
  device.init_device_context();

  torch::Tensor tensor = torch::ones(
      {4},
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap()));
  EXPECT_TRUE(torch::equal(tensor.cpu(), torch::ones({4})));

  // FlashinferWorkspace also allocates pinned host memory immediately after
  // initializing its device buffers.
  torch::Tensor pinned = torch::empty(
      {16}, torch::TensorOptions().dtype(torch::kUInt8).pinned_memory(true));
  EXPECT_TRUE(pinned.is_pinned());
}

TEST(MusaDeviceTest, RecordedLayersCompleteAndAbortUnblocksMissingLayers) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  auto synchronizer = std::make_shared<MUSALayerSynchronizer>(/*num_layers=*/2);
  auto published_layer = std::async(std::launch::async, [&]() {
    return synchronizer->synchronize_layer(/*layer_index=*/0);
  });
  ScopeGuard publication_guard([&]() { synchronizer->abort(); });
  // An unpublished layer waits until its producer records an event.
  EXPECT_EQ(published_layer.wait_for(std::chrono::milliseconds(200)),
            std::future_status::timeout);
  // Concurrent producers must publish exactly one event for the same layer.
  std::latch record_start(/*expected=*/3);
  auto record_layer = [&]() {
    record_start.arrive_and_wait();
    return synchronizer->record_current(/*layer_index=*/0,
                                        /*device_index=*/0);
  };
  auto first_record = std::async(std::launch::async, record_layer);
  auto second_record = std::async(std::launch::async, record_layer);
  record_start.arrive_and_wait();
  EXPECT_NE(first_record.get(), second_record.get());
  const std::future_status published =
      published_layer.wait_for(std::chrono::seconds(1));
  ASSERT_EQ(published, std::future_status::ready);
  EXPECT_TRUE(published_layer.get());
  publication_guard.dismiss();
  EXPECT_TRUE(synchronizer->synchronize_layer(/*layer_index=*/0));
  EXPECT_FALSE(synchronizer->record_current(/*layer_index=*/0,
                                            /*device_index=*/0));

  auto missing_layer = std::async(std::launch::async, [&]() {
    return synchronizer->synchronize_layer(/*layer_index=*/1);
  });
  synchronizer->abort();
  const std::future_status missing =
      missing_layer.wait_for(std::chrono::seconds(1));
  if (missing != std::future_status::ready) {
    synchronizer->abort();
  }
  EXPECT_EQ(missing, std::future_status::ready);
  EXPECT_FALSE(missing_layer.get());
  EXPECT_FALSE(synchronizer->record_current(/*layer_index=*/1,
                                            /*device_index=*/0));
  EXPECT_FALSE(synchronizer->synchronize_layer(/*layer_index=*/0));

  // Cancellation must release unpublished layers without a local watchdog.
  auto cancelled = std::make_shared<MUSALayerSynchronizer>(/*num_layers=*/1);
  auto unpublished_layer = std::async(std::launch::async, [&]() {
    return cancelled->synchronize_layer(/*layer_index=*/0);
  });
  cancelled->abort();
  EXPECT_EQ(unpublished_layer.wait_for(std::chrono::seconds(1)),
            std::future_status::ready);
  EXPECT_FALSE(unpublished_layer.get());
}

}  // namespace
}  // namespace xllm
