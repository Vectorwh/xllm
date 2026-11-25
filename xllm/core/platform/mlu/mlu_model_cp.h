/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <string>
#include <unordered_map>

namespace xllm::mlu {

struct ModelCpCapability {
  bool supports_vlm;
};

inline const std::unordered_map<std::string, ModelCpCapability>&
model_cp_capabilities() {
  static const std::unordered_map<std::string, ModelCpCapability> kCapabilities{
      {"deepseek_v41", {.supports_vlm = true}},
      {"deepseek_v4", {.supports_vlm = false}},
      {"glm5_next", {.supports_vlm = true}},
      {"glm_moe_dsa", {.supports_vlm = false}},
  };
  return kCapabilities;
}

inline bool supports_vlm_cp(const std::string& model_type) {
  const auto& capabilities = model_cp_capabilities();
  const auto it = capabilities.find(model_type);
  return it != capabilities.end() && it->second.supports_vlm;
}

}  // namespace xllm::mlu
