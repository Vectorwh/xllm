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

#include "framework/lora/dit_lora_runtime.h"

#include <glog/logging.h>

#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

#include "framework/lora/adapter_loader.h"

namespace xllm::dit {

DiTLoRARuntime& DiTLoRARuntime::instance() {
  static DiTLoRARuntime runtime;
  return runtime;
}

std::vector<LoRARequest> DiTLoRARuntime::register_adapters(
    const LoRAConfig& config,
    const std::string& base_model_name,
    LoRARegistry& registry) {
  if (!config.enable_lora) {
    return {};
  }
  CHECK(!config.allow_runtime_lora_updating)
      << "DiT LoRA requires --allow_runtime_lora_updating=false";
  CHECK(!base_model_name.empty());
  CHECK_GT(config.max_loras, 0);
  CHECK_LE(config.lora_modules.size(), static_cast<size_t>(config.max_loras));

  std::unordered_set<std::string> names;
  names.reserve(config.lora_modules.size());
  std::vector<LoRARequest> adapters;
  adapters.reserve(config.lora_modules.size());
  // All ranks must supply lora_modules in the same order for consistent IDs.
  for (const auto& [name, path] : config.lora_modules) {
    CHECK(!name.empty() && !path.empty()) << "Empty DiT LoRA name or path";
    CHECK_NE(name, base_model_name) << "LoRA alias collides with base model";
    CHECK(names.emplace(name).second) << "Duplicate DiT LoRA alias: " << name;
    adapters.push_back({name,
                        static_cast<uint64_t>(adapters.size()) + 1,
                        path,
                        base_model_name});
    const LoRARequest& adapter = adapters.back();
    const auto existing = registry.lookup(adapter.lora_name);
    if (existing.has_value()) {
      CHECK_EQ(existing->base_model_name, base_model_name)
          << "DiT LoRA base model conflict: " << adapter.lora_name;
    }
    // The registry rejects conflicting paths or IDs rather than replacing
    // existing entries. ID 0 remains reserved for base-model requests.
    const auto registered_id = registry.register_adapter(adapter);
    CHECK(registered_id.has_value())
        << "Cannot register DiT LoRA: " << adapter.lora_name;
    CHECK_EQ(*registered_id, adapter.lora_int_id);
    LOG(INFO) << "DiT static adapter: name=" << adapter.lora_name
              << ", id=" << adapter.lora_int_id << ", base=" << base_model_name
              << ", path=" << adapter.lora_path;
  }
  return adapters;
}

namespace {

torch::Tensor shard_output_rows(const torch::Tensor& tensor,
                                const DiTLoRAOutputLayout& layout,
                                int64_t tp_rank,
                                int64_t tp_size) {
  CHECK_EQ(tensor.dim(), 2);
  CHECK_GT(tp_size, 0);
  CHECK_GE(tp_rank, 0);
  CHECK_LT(tp_rank, tp_size);
  if (layout.partition_sizes.empty()) {
    CHECK(layout.partition_order.empty());
    CHECK_EQ(tensor.size(0) % tp_size, 0);
    const int64_t rows = tensor.size(0) / tp_size;
    return tensor.narrow(/*dim=*/0, tp_rank * rows, rows).contiguous();
  }
  CHECK_EQ(layout.partition_sizes.size(), layout.partition_order.size());
  CHECK_EQ(std::accumulate(layout.partition_sizes.begin(),
                           layout.partition_sizes.end(),
                           int64_t{0}),
           tensor.size(0));
  std::vector<int64_t> offsets;
  offsets.reserve(layout.partition_sizes.size());
  int64_t offset = 0;
  for (int64_t size : layout.partition_sizes) {
    CHECK_GT(size, 0);
    CHECK_EQ(size % tp_size, 0);
    offsets.emplace_back(offset);
    offset += size;
  }
  std::vector<bool> visited(offsets.size(), false);
  std::vector<torch::Tensor> pieces;
  pieces.reserve(offsets.size());
  for (int64_t index : layout.partition_order) {
    CHECK_GE(index, 0);
    CHECK_LT(static_cast<size_t>(index), offsets.size());
    CHECK(!visited[index]);
    visited[index] = true;
    const int64_t rows = layout.partition_sizes[index] / tp_size;
    pieces.emplace_back(
        tensor.narrow(/*dim=*/0, offsets[index] + tp_rank * rows, rows));
  }
  return torch::cat(pieces, /*dim=*/0).contiguous();
}

}  // namespace

bool DiTLoRARuntime::load_adapters(const LoRAConfig& config,
                                   const std::vector<LoRARequest>& requests,
                                   torch::Device device,
                                   const Targets& targets) {
  CHECK(installed_adapter_ids_.empty());
  if (!requests.empty() && targets.empty()) {
    LOG(ERROR) << "This DiT model has no LoRA targets";
    return false;
  }

  struct PendingProjection {
    // The caller's target map outlives all pending installations.
    const Target* target;
    torch::Tensor a;
    torch::Tensor b;
  };
  struct PendingAdapter {
    uint64_t id;
    float scaling;
    std::vector<PendingProjection> projections;
  };
  std::vector<PendingAdapter> pending_adapters;
  pending_adapters.reserve(requests.size());
  std::unordered_set<uint64_t> adapter_ids;
  adapter_ids.reserve(requests.size());
  LoRAAdapterLoader loader(config);
  for (const LoRARequest& request : requests) {
    CHECK_GT(request.lora_int_id, 0);
    CHECK(adapter_ids.emplace(request.lora_int_id).second);
    const auto adapter = loader.load_dit(request);
    if (!adapter) {
      return false;
    }
    PendingAdapter pending{request.lora_int_id, adapter->scaling, {}};
    pending.projections.reserve(adapter->target_modules.size());
    int64_t device_bytes = 0;
    for (const std::string& name : adapter->target_modules) {
      const auto found = targets.find(name);
      if (found == targets.end()) {
        LOG(ERROR) << "Unbound DiT LoRA target " << name << " in "
                   << request.lora_path;
        return false;
      }
      const Target& target = found->second;
      const int64_t tp_size = target.tp_size;
      const int64_t tp_rank = target.tp_rank;
      const bool column = target.parallel_type == ParallelType::COLUMN;
      const int64_t global_input =
          target.local_input_size * (column ? 1 : tp_size);
      const int64_t global_output =
          target.local_output_size * (column ? tp_size : 1);
      torch::Tensor a = adapter->tensors.at(name + "#A");
      torch::Tensor b = adapter->tensors.at(name + "#B");
      if (a.size(1) != global_input || b.size(0) != global_output) {
        LOG(ERROR) << "DiT LoRA shape mismatch at " << name << " in "
                   << request.lora_path << ": A=" << a.sizes()
                   << ", B=" << b.sizes() << ", expected input=" << global_input
                   << ", output=" << global_output;
        return false;
      }
      if (column) {
        b = shard_output_rows(b, target.layout, tp_rank, tp_size);
      } else {
        a = a.narrow(/*dim=*/1,
                     tp_rank * target.local_input_size,
                     target.local_input_size)
                .contiguous();
      }
      // Rolling base blocks may currently be on CPU. Adapter placement follows
      // the worker's execution device, not the transient base weight device.
      const torch::TensorOptions options =
          torch::TensorOptions().device(device).dtype(target.dtype);
      a = a.to(options).contiguous();
      b = b.to(options).contiguous();
      device_bytes += a.nbytes() + b.nbytes();
      VLOG(1) << "DiT LoRA " << request.lora_name << ": target=" << name
              << ", TP=" << tp_rank << "/" << tp_size << ", device=" << device
              << ", A=" << a.sizes() << ", B=" << b.sizes();
      pending.projections.push_back({&target, std::move(a), std::move(b)});
    }
    pending_adapters.push_back(std::move(pending));
    LOG(INFO) << "Prepared DiT LoRA '" << request.lora_name
              << "' id=" << request.lora_int_id << ", device=" << device
              << ", bound_targets=" << adapter->target_modules.size()
              << ", rank=" << adapter->r << ", scaling=" << adapter->scaling
              << ", device_bytes=" << device_bytes;
  }

  for (PendingAdapter& pending : pending_adapters) {
    for (PendingProjection& proj : pending.projections) {
      proj.target->install_weights(
          pending.id, std::move(proj.a), std::move(proj.b), pending.scaling);
    }
  }
  installed_adapter_ids_ = std::move(adapter_ids);
  return true;
}

bool DiTLoRARuntime::contains(uint64_t adapter_id) const {
  return installed_adapter_ids_.contains(adapter_id);
}

}  // namespace xllm::dit
