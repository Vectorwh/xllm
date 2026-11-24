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

// Registers static adapters and installs their TP-local weights at startup.
// The model's linear modules own the installed tensors.

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "framework/lora/lora_config.h"
#include "framework/lora/lora_registry.h"
#include "framework/lora/lora_request.h"

namespace xllm::dit {

// Partition sizes describe checkpoint output rows; order describes compute
// layout. Each partition is TP-sharded before concatenation in output order.
struct DiTLoRAOutputLayout {
  std::vector<int64_t> partition_sizes;
  std::vector<int64_t> partition_order;
};

class DiTLoRARuntime final {
 public:
  static DiTLoRARuntime& instance();

  // Every serving process (engine and each worker) must receive the same
  // --lora_modules in the same order: IDs are assigned by input position
  // (1-based; 0 is reserved for base-model requests), so identical inputs
  // yield identical name->id bindings without any cross-process traffic.
  // Repeated registration in the engine/worker is idempotent. Static: the
  // engine process registers too, and it never init()s a device.
  static std::vector<LoRARequest> register_adapters(
      const LoRAConfig& config,
      const std::string& base_model_name,
      LoRARegistry& registry);

  enum class ParallelType { COLUMN, ROW };
  struct Target {
    ParallelType parallel_type;
    int64_t local_input_size;
    int64_t local_output_size;
    torch::ScalarType dtype;
    // TP topology captured at registration; forward-time reduction uses the
    // wrapper's own process group, never these numbers.
    int64_t tp_rank;
    int64_t tp_size;
    DiTLoRAOutputLayout layout;
    std::function<void(uint64_t, torch::Tensor, torch::Tensor, float)>
        install_weights;
  };
  using Targets = std::unordered_map<std::string, Target>;

  // Prepare the adapters on the worker device before publishing their
  // weights. Target callbacks are used only during this startup call.
  bool load_adapters(const LoRAConfig& config,
                     const std::vector<LoRARequest>& requests,
                     torch::Device device,
                     const Targets& targets);

  bool contains(uint64_t adapter_id) const;

 private:
  DiTLoRARuntime() = default;

  std::unordered_set<uint64_t> installed_adapter_ids_;
};

}  // namespace xllm::dit

namespace xllm::lora {

using SuffixAlias = std::pair<std::string, std::string>;

namespace detail {

inline std::string apply_suffix_aliases(
    std::string path,
    const std::vector<SuffixAlias>& aliases) {
  for (const auto& [suffix, replacement] : aliases) {
    if (path.size() < suffix.size()) {
      continue;
    }
    const std::size_t start = path.size() - suffix.size();
    if (path.compare(start, suffix.size(), suffix) != 0) {
      continue;
    }
    if (start != 0 && path[start - 1] != '.') {
      continue;
    }
    path.replace(start, suffix.size(), replacement);
    break;
  }
  return path;
}

}  // namespace detail

template <typename... Wrappers, typename Visitor>
void visit_lora_projectors(torch::nn::Module& root,
                           std::string root_prefix,
                           const std::vector<SuffixAlias>& suffix_aliases,
                           Visitor&& visitor) {
  // LibTorch inserts the separator between the prefix and each child key.
  while (!root_prefix.empty() && root_prefix.back() == '.') {
    root_prefix.pop_back();
  }
  for (const auto& item :
       root.named_modules(root_prefix, /*include_self=*/false)) {
    ([&item, &suffix_aliases, &visitor] {
      auto* wrapper = dynamic_cast<Wrappers*>(item.value().get());
      if (wrapper == nullptr) {
        return false;
      }
      std::string proj_name =
          detail::apply_suffix_aliases(item.key(), suffix_aliases);
      VLOG(1) << "[LoRA walker] found " << proj_name;
      visitor(std::move(proj_name), *wrapper);
      return true;
    }() ||
     ...);
  }
}

}  // namespace xllm::lora
