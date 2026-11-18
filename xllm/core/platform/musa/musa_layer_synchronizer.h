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

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "core/platform/stream.h"

namespace xllm {

// One instance belongs to one forward. A host-side Mooncake writer must wait
// for the producer event, not merely enqueue a device-stream dependency.
class MUSALayerSynchronizer final {
 public:
  explicit MUSALayerSynchronizer(int64_t num_layers);

  bool synchronize_layer(int64_t layer_index);
  bool record_current(int64_t layer_index, int32_t device_index);
  void abort();

 private:
  bool valid_index(int64_t layer_index) const;
  StreamEventPtr wait_for_recorded_event(int64_t layer_index);
  bool record_stream(int64_t layer_index, Stream* stream);

  std::mutex record_mutex_;
  std::vector<StreamEventPtr> events_;
  std::vector<std::atomic<bool>> event_record_flags_;
  std::atomic<bool> aborted_{false};
};

}  // namespace xllm
