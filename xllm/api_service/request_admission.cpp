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

#include "api_service/request_admission.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_split.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <utility>

#include "core/framework/config/service_config.h"
#include "core/framework/lora/lora_runtime.h"

namespace xllm::api_service_internal {

Status rate_limiter_rejection_status(const RateLimiter& rate_limiter) {
  if (rate_limiter.is_sleeping()) {
    return {StatusCode::UNAVAILABLE, "Model is currently in sleep state."};
  }
  return {StatusCode::RESOURCE_EXHAUSTED,
          "The number of concurrent requests has reached the limit.",
          FailureReason::CONCURRENCY_LIMIT_REACHED};
}

Status model_not_supported_status() {
  return {StatusCode::UNKNOWN,
          "Model not supported",
          FailureReason::MODEL_NOT_FOUND};
}

Status admit_rpc_request(const std::string& model,
                         const absl::flat_hash_set<std::string>& models,
                         RateLimiter* rate_limiter,
                         bool supports_lora,
                         int64_t num_requests) {
  // Only services that resolve and pin an adapter may admit its name.
  // Otherwise a request can pass admission and silently execute the base.
  if (!models.contains(model)) {
    const bool is_lora_adapter =
        supports_lora && LoRARuntime::instance().enabled() &&
        LoRARuntime::instance().registry().lookup(model).has_value();
    if (!is_lora_adapter) {
      return model_not_supported_status();
    }
  }

  CHECK_GT(num_requests, 0);
  const int32_t max = ServiceConfig::get_instance().max_concurrent_requests();
  // A batch that cannot fit even on an idle server will not benefit from
  // retrying. Reserve 429 for temporary occupancy by other requests.
  if (max > 0 && num_requests > max) {
    return Status::invalid_parameter(
        "documents count exceeds max_concurrent_requests", "documents");
  }
  if (num_requests > std::numeric_limits<int32_t>::max()) {
    return Status::invalid_parameter(
        "documents count exceeds supported request count", "documents");
  }
  CHECK(rate_limiter != nullptr);
  if (rate_limiter->is_limited(static_cast<int32_t>(num_requests))) {
    return rate_limiter_rejection_status(*rate_limiter);
  }
  return {};
}

Status admit_dit_request(const std::string& model,
                         const std::string& base_model,
                         const absl::flat_hash_set<std::string>& models,
                         RateLimiter* rate_limiter,
                         std::vector<uint64_t>& adapter_ids) {
  const LoRARuntime& runtime = LoRARuntime::instance();
  std::vector<uint64_t> resolved_ids;
  if (!models.contains(model) && runtime.enabled()) {
    // DiT registers all static adapters before serving requests.
    const std::vector<std::string> names = absl::StrSplit(model, ',');
    resolved_ids.reserve(names.size());
    for (const std::string& name : names) {
      const auto adapter = runtime.registry().lookup(
          std::string(absl::StripAsciiWhitespace(name)));
      if (!adapter.has_value() || adapter->base_model_name != base_model) {
        return model_not_supported_status();
      }
      resolved_ids.emplace_back(adapter->lora_int_id);
    }
    std::sort(resolved_ids.begin(), resolved_ids.end());
    if (std::adjacent_find(resolved_ids.begin(), resolved_ids.end()) !=
        resolved_ids.end()) {
      return Status::invalid_parameter("Duplicate LoRA adapter", "model");
    }
  }

  const Status status = admit_rpc_request(
      resolved_ids.empty() ? model : base_model, models, rate_limiter);
  if (status.ok()) {
    adapter_ids = resolved_ids.empty() ? std::vector<uint64_t>{0}
                                       : std::move(resolved_ids);
  }
  return status;
}

}  // namespace xllm::api_service_internal
