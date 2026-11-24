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

#include "framework/lora/dit_lora_context.h"

#include <glog/logging.h>

#include "framework/lora/dit_lora_runtime.h"

namespace xllm::dit {
namespace {
thread_local std::span<const uint64_t> active_adapter_ids;
}  // namespace

std::span<const uint64_t> active_dit_lora_adapter_ids() {
  return active_adapter_ids;
}

DiTLoRAScope::DiTLoRAScope(std::span<const uint64_t> adapter_ids)
    : previous_adapter_ids_(active_adapter_ids) {
  if (adapter_ids.size() == 1 && adapter_ids.front() == 0) {
    adapter_ids = {};
  }
  uint64_t previous_id = 0;
  for (const uint64_t adapter_id : adapter_ids) {
    CHECK_GT(adapter_id, previous_id)
        << "Expected sorted, unique DiT adapter IDs";
    CHECK(DiTLoRARuntime::instance().contains(adapter_id))
        << "DiT adapter " << adapter_id << " is not installed on this worker";
    previous_id = adapter_id;
  }
  active_adapter_ids = adapter_ids;
}

DiTLoRAScope::~DiTLoRAScope() { active_adapter_ids = previous_adapter_ids_; }

}  // namespace xllm::dit
