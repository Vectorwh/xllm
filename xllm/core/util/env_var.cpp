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

#include "env_var.h"

#include <glog/logging.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace xllm {
namespace util {

bool get_bool_env(const std::string& key, bool defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  std::string strVal(val);
  return (strVal == "1" || strVal == "true" || strVal == "TRUE" ||
          strVal == "True");
}

int64_t get_int_env(const std::string& key, int64_t defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  // Use strtol for proper error handling
  char* endptr;
  int64_t result = std::strtol(val, &endptr, 10);
  // Check if conversion was successful (endptr points to end of string or valid
  // terminator)
  if (endptr == val || *endptr != '\0') {
    return defaultValue;
  }
  // Check for overflow/underflow
  if (result < INT64_MIN || result > INT64_MAX) {
    return defaultValue;
  }
  return result;
}

std::string get_string_env(const std::string& name) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    LOG(FATAL) << "Environment variable " << name.c_str() << " is not set";
  }
  return std::string(val);
}

std::optional<std::string> get_optional_string_env(const std::string& name) {
  const char* val = std::getenv(name.c_str());
  if (val == nullptr) {
    return std::nullopt;
  }
  return std::string(val);
}

double get_double_env(const std::string& key, double defaultValue = -1) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  char* endptr = nullptr;
  double result = std::strtod(val, &endptr);
  // Check if conversion was successful (endptr points to end of string or valid
  // terminator)
  if (endptr == val || *endptr != '\0') {
    LOG(WARNING) << "Invalid value for " << key << ": " << val
                 << ". Must be a valid double. Using default " << defaultValue;
    return defaultValue;
  }
  return result;
}

int64_t get_process_group_test_timeout_seconds() {
  // Default timeout is 4 seconds, but can be overridden via environment
  // variable to accommodate multi-node multi-device communication scenarios
  // where network latency may require a longer timeout period.
  constexpr int64_t kDefaultTimeoutSeconds = 4;
  constexpr const char* kTimeoutEnvVar =
      "XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS";
  return get_int_env(kTimeoutEnvVar, kDefaultTimeoutSeconds);
}

const std::array<double, 2>& get_musa_pcp_costs() {
  static const std::array<double, 2> costs = [] {
    const auto read_cost = [](const std::string& name, double default_value) {
      const std::optional<std::string> value = get_optional_string_env(name);
      if (!value.has_value()) {
        return default_value;
      }
      char* end = nullptr;
      errno = 0;
      const double cost = std::strtod(value->c_str(), &end);
      CHECK(end != value->c_str() && end == value->c_str() + value->size() &&
            errno != ERANGE && std::isfinite(cost) && cost >= 0)
          << name << " must be a finite non-negative double, got: " << *value;
      // Canonicalize signed zero for the bytewise CP consensus check.
      return cost == 0 ? 0.0 : cost;
    };
    // Defaults measured for Qwen3.8 FP8 on MTT S5000.
    const std::array<double, 2> values = {
        read_cost("XLLM_MUSA_PCP_COST_PAIR", /*default_value=*/9.4e-6),
        read_cost("XLLM_MUSA_PCP_COST_OTHER_KV_ROW", /*default_value=*/0.04)};
    LOG(INFO) << "[MUSA_PCP_COST] attention_per_pair=" << values[0]
              << " other_rank_kv_per_row=" << values[1];
    return values;
  }();
  return costs;
}

bool store_prefetch_stats_enabled() {
  static const bool enabled =
      get_bool_env("XLLM_STORE_PREFETCH_STATS", /*defaultValue=*/false);
  return enabled;
}

}  // namespace util
}  // namespace xllm
