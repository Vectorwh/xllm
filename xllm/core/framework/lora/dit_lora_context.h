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

// Thread-local LoRA routing for DiT pipelines.
//
// All requests in a DiT batch share the same list of active adapters.
// DiTWorkerImpl::step owns the input and scopes the selection for the
// whole executor forward, including every denoising step.

#pragma once

#include <cstdint>
#include <span>

namespace xllm::dit {

std::span<const uint64_t> active_dit_lora_adapter_ids();

// The caller owns the ID list for the lifetime of the scope.
class DiTLoRAScope final {
 public:
  explicit DiTLoRAScope(std::span<const uint64_t> adapter_ids);
  ~DiTLoRAScope();
  DiTLoRAScope(const DiTLoRAScope&) = delete;
  DiTLoRAScope& operator=(const DiTLoRAScope&) = delete;

 private:
  std::span<const uint64_t> previous_adapter_ids_;
};

}  // namespace xllm::dit
