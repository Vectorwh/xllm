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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "atb_buffer.h"

namespace xllm {

// Temporary capture workaround for models using the ATB workspace API.
// Remove this adapter once those model paths manage capture-safe workspace
// lifetimes. Until then, share scratch within one serial executor slot and
// retain each allocation to preserve addresses in all captured graphs.
class AtbGraphWorkspace final {
 public:
  class Scope final {
   public:
    explicit Scope(AtbGraphWorkspace& workspace);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    AtbGraphWorkspace* previous_;
  };

  explicit AtbGraphWorkspace(torch::Device device) : device_(device) {}
  AtbGraphWorkspace(const AtbGraphWorkspace&) = delete;
  AtbGraphWorkspace& operator=(const AtbGraphWorkspace&) = delete;

  static AtbGraphWorkspace* current();
  void* get_buffer(uint64_t size_bytes, int32_t device_id);

 private:
  static thread_local AtbGraphWorkspace* current_;
  torch::Device device_;
  uint64_t capacity_bytes_ = 0;
  std::vector<torch::Tensor> buffers_;
};

class AtbWorkspace {
 public:
  AtbWorkspace() = default;

  AtbWorkspace(at::Device device);

  ~AtbWorkspace();

  AtbWorkspace(const AtbWorkspace&) = delete;

  AtbWorkspace& operator=(const AtbWorkspace&) = delete;

  AtbWorkspace(AtbWorkspace&&) = default;

  AtbWorkspace& operator=(AtbWorkspace&&) = default;

  void* get_workspace_buffer(uint64_t bufferSize);

 private:
  std::map<int32_t, std::unique_ptr<AtbBuffer>> buffer_map_;
};

}  // namespace xllm
