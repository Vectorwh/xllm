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

#include "core/platform/musa/musa_layer_synchronizer.h"

#include <glog/logging.h>
#include <torch_musa/csrc/core/MUSAStream.h>

#include <limits>
#include <thread>
#include <utility>

namespace xllm {
namespace {

size_t checked_layer_count(int64_t num_layers) {
  CHECK_GT(num_layers, 0);
  CHECK_LE(num_layers, std::numeric_limits<uint32_t>::max());
  return static_cast<size_t>(num_layers);
}

}  // namespace

MUSALayerSynchronizer::MUSALayerSynchronizer(int64_t num_layers)
    : events_(checked_layer_count(num_layers)),
      event_record_flags_(static_cast<size_t>(num_layers)) {
  for (std::atomic<bool>& flag : event_record_flags_) {
    flag.store(false, std::memory_order_relaxed);
  }
}

bool MUSALayerSynchronizer::valid_index(int64_t layer_index) const {
  if (layer_index < 0 || static_cast<size_t>(layer_index) >= events_.size()) {
    LOG(ERROR) << "MUSA layer event index out of bounds, layer=" << layer_index
               << ", size=" << events_.size();
    return false;
  }
  return true;
}

StreamEventPtr MUSALayerSynchronizer::wait_for_recorded_event(
    int64_t layer_index) {
  if (!valid_index(layer_index)) {
    return nullptr;
  }
  while (!event_record_flags_[layer_index].load(std::memory_order_acquire)) {
    if (aborted_.load(std::memory_order_acquire)) {
      return nullptr;
    }
    std::this_thread::yield();
  }
  if (aborted_.load(std::memory_order_acquire) ||
      events_[layer_index] == nullptr) {
    return nullptr;
  }
  return events_[layer_index];
}

bool MUSALayerSynchronizer::synchronize_layer(int64_t layer_index) {
  const StreamEventPtr event = wait_for_recorded_event(layer_index);
  if (event == nullptr) {
    return false;
  }
  // RDMA reads HBM from a host thread. The producer kernels must be complete
  // before their cache addresses can be handed to the transfer engine.
  if (!event->synchronize()) {
    LOG(ERROR) << "MUSA layer event synchronize failed, layer=" << layer_index;
    this->abort();
    return false;
  }
  return !aborted_.load(std::memory_order_acquire);
}

bool MUSALayerSynchronizer::record_stream(int64_t layer_index, Stream* stream) {
  // Serialize writers without publishing readiness before the event exists.
  std::lock_guard<std::mutex> lock(record_mutex_);
  if (!valid_index(layer_index) || stream == nullptr ||
      aborted_.load(std::memory_order_acquire)) {
    return false;
  }
  if (event_record_flags_[layer_index].load(std::memory_order_acquire)) {
    LOG(ERROR) << "MUSA layer event recorded twice, layer=" << layer_index;
    return false;
  }
  StreamEventPtr event = stream->record_event();
  if (event == nullptr) {
    return false;
  }
  events_[layer_index] = std::move(event);
  event_record_flags_[layer_index].store(true, std::memory_order_release);
  return true;
}

bool MUSALayerSynchronizer::record_current(int64_t layer_index,
                                           int32_t device_index) {
  // Device/stream exceptions propagate to the producer's ownership guard.
  // Do not turn an unexpected device failure into a recoverable record miss.
  Stream current_stream(c10::musa::getCurrentMUSAStream(device_index));
  return record_stream(layer_index, &current_stream);
}

void MUSALayerSynchronizer::abort() {
  aborted_.store(true, std::memory_order_release);
}

}  // namespace xllm
