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

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <vector>

namespace xllm {

// Shared upper bound (bytes) of the custom all-reduce staging buffer.
inline constexpr int64_t kMusaCustomArMaxBytes = 8 * 1024 * 1024;

class ProcessGroup;

class MusaCustomAllReduce final {
 public:
  MusaCustomAllReduce(int32_t rank,
                      int32_t world_size,
                      std::string group_name,
                      const torch::Device& device,
                      c10::intrusive_ptr<c10d::Store> store);
  ~MusaCustomAllReduce();

  MusaCustomAllReduce(const MusaCustomAllReduce&) = delete;
  MusaCustomAllReduce& operator=(const MusaCustomAllReduce&) = delete;

  bool enabled() const { return enabled_; }

  std::optional<torch::Tensor> allreduce(
      const torch::Tensor& input,
      const std::optional<torch::Tensor>& output = std::nullopt);

  // Optional out-of-place reduction for the generic TP caller. Selection is
  // fixed at group initialization; unsupported tensors retain the MCCL path.
  std::optional<torch::Tensor> allreduce_tp(const torch::Tensor& input);

 private:
  bool should_use(const torch::Tensor& input) const;
  // Both ranks must slice identically; each slice fits the IPC buffer.
  std::optional<torch::Tensor> allreduce_in_slices(const torch::Tensor& input);
  void init();
  // CP-only: init the IPC buffers, then agree on the effective backend across
  // all cp_group ranks via the store and fail fast on any split (a split
  // deadlocks the CP collective silently).
  void init_cp_group_consistently();
  std::vector<int64_t> create_shared_buffer(int64_t size_in_bytes,
                                            const std::string& name);
  std::vector<uint8_t> tensor_to_bytes(const torch::Tensor& tensor) const;
  torch::Tensor bytes_to_tensor(const std::vector<uint8_t>& bytes) const;

  int32_t rank_ = 0;
  int32_t world_size_ = 0;
  std::string group_name_;
  torch::Device device_;
  c10::intrusive_ptr<c10d::Store> store_;

  bool enabled_ = false;
  bool enable_tp_allreduce_ = false;
  int64_t max_size_bytes_ = kMusaCustomArMaxBytes;
  int64_t custom_ar_handle_ = 0;
  std::vector<int64_t> meta_ptrs_;
  std::vector<int64_t> buffer_ptrs_;
  torch::Tensor rank_data_;
};

bool musa_custom_allreduce_enabled(ProcessGroup* process_group);

std::optional<torch::Tensor> musa_allreduce_custom(
    ProcessGroup* process_group,
    const torch::Tensor& input,
    const std::optional<torch::Tensor>& output = std::nullopt);

// Only tp_group participates; world/CP callers retain their existing policy.
std::optional<torch::Tensor> musa_allreduce_tp_custom(
    ProcessGroup* process_group,
    const torch::Tensor& input);

}  // namespace xllm
