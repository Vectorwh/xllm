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

#include "atb_workspace.h"

#include <acl/acl.h>

#include <algorithm>
#include <limits>

#include "atb_buffer.h"
#include "common/device_monitor.h"

namespace xllm {

thread_local AtbGraphWorkspace* AtbGraphWorkspace::current_ = nullptr;

AtbGraphWorkspace::Scope::Scope(AtbGraphWorkspace& workspace)
    : previous_(current_) {
  current_ = &workspace;
}

AtbGraphWorkspace::Scope::~Scope() { current_ = previous_; }

AtbGraphWorkspace* AtbGraphWorkspace::current() { return current_; }

void* AtbGraphWorkspace::get_buffer(uint64_t size_bytes, int32_t device_id) {
  CHECK_EQ(device_id, device_.index());
  CHECK_LE(size_bytes,
           static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
  if (size_bytes == 0) {
    return nullptr;
  }
  if (size_bytes > capacity_bytes_) {
    uint64_t capacity = std::max(uint64_t{1}, capacity_bytes_);
    while (capacity < size_bytes) {
      CHECK_LE(capacity,
               static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2);
      capacity *= 2;
    }
    // Captured nodes retain their old addresses. Doubling bounds all retained
    // generations below twice the newest capacity without synchronizing.
    buffers_.emplace_back(torch::empty(
        {static_cast<int64_t>(capacity)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device_)));
    capacity_bytes_ = capacity;
  }
  return buffers_.back().data_ptr();
}

AtbWorkspace::AtbWorkspace(at::Device device) {
  int32_t device_id = device.index();
  auto it = buffer_map_.find(device_id);
  if (it == buffer_map_.end()) {
    buffer_map_[device_id] = std::make_unique<AtbBuffer>(1, device);
  }
}

AtbWorkspace::~AtbWorkspace() = default;

void* AtbWorkspace::get_workspace_buffer(uint64_t bufferSize) {
  int32_t device_id = 0;
  aclrtGetDevice(&device_id);

  auto it = buffer_map_.find(device_id);
  if (it == buffer_map_.end()) {
    LOG(FATAL) << " Fail to find device_id in buffer_map_ : " << device_id;
  }
  DeviceMonitor::get_instance().monitor_buffer(device_id, bufferSize);

  if (AtbGraphWorkspace* graph_workspace = AtbGraphWorkspace::current()) {
    return graph_workspace->get_buffer(bufferSize, device_id);
  }
  return it->second->get_buffer(bufferSize);
}

}  // namespace xllm
