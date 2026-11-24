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

#include "adapter_loader.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "framework/state_dict/state_dict.h"

namespace xllm {

namespace {
// Fixed prefix HuggingFace PEFT prepends to every weight in
// adapter_model.safetensors. We strip it and then further rewrite the
// remainder into an xllm-friendly key.
constexpr const char* kHfPrefix = "base_model.model.";
constexpr const char* kLoraATag = ".lora_A.weight";
constexpr const char* kLoraBTag = ".lora_B.weight";

bool contains(const std::vector<std::string>& v, const std::string& item) {
  return std::find(v.begin(), v.end(), item) != v.end();
}
}  // namespace

std::optional<LoRAAdapter> LoRAAdapterLoader::load(const LoRARequest& req) {
  namespace fs = std::filesystem;
  const fs::path root = req.lora_path;
  const fs::path config_path = root / "adapter_config.json";
  const fs::path weights_path = root / "adapter_model.safetensors";

  if (!fs::exists(config_path)) {
    LOG(ERROR) << "[LoRAAdapterLoader] '" << req.lora_name
               << "': missing adapter_config.json at " << config_path;
    return std::nullopt;
  }
  if (!fs::exists(weights_path)) {
    LOG(ERROR) << "[LoRAAdapterLoader] '" << req.lora_name
               << "': missing adapter_model.safetensors at " << weights_path;
    return std::nullopt;
  }

  LoRAAdapter adapter;
  adapter.request = req;

  if (!parse_config(config_path.string(), req.base_model_name, &adapter)) {
    return std::nullopt;
  }
  if (!load_weights(weights_path.string(), &adapter)) {
    return std::nullopt;
  }
  for (const auto& [key, tensor] : adapter.tensors) {
    if (key.find(".experts.") != std::string::npos) {
      LOG(ERROR) << "[LoRAAdapterLoader] '" << req.lora_name
                 << "': MoE expert LoRA forward is not implemented; "
                    "refusing an adapter that would silently ignore weights";
      return std::nullopt;
    }
  }

  LOG(INFO) << "[LoRAAdapterLoader] loaded '" << req.lora_name
            << "' r=" << adapter.r << " scaling=" << adapter.scaling
            << " target_modules=" << adapter.target_modules.size()
            << " tensors=" << adapter.tensors.size();
  return adapter;
}

bool LoRAAdapterLoader::parse_config(
    const std::string& path,
    const std::string& expected_base_model_name,
    LoRAAdapter* out) const {
  std::ifstream f(path);
  if (!f) {
    LOG(ERROR) << "[LoRAAdapterLoader] cannot open " << path;
    return false;
  }
  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path
               << ": parse error: " << e.what();
    return false;
  }

  try {
    // Reject non-LoRA PEFT variants explicitly. DoRA and IA3 need extra
    // machinery we don't have yet.
    if (j.value("peft_type", std::string("LORA")) != "LORA") {
      LOG(ERROR) << "[LoRAAdapterLoader] " << path
                 << ": peft_type != LORA (got '" << j.value("peft_type", "?")
                 << "'), refusing";
      return false;
    }
    if (j.value("bias", std::string("none")) != "none") {
      LOG(ERROR) << "[LoRAAdapterLoader] " << path
                 << ": bias != none, refusing";
      return false;
    }
    if (j.value("use_dora", false)) {
      LOG(ERROR) << "[LoRAAdapterLoader] " << path
                 << ": use_dora=true, refusing";
      return false;
    }

    const int32_t r = j.value("r", 0);
    const int32_t lora_alpha = j.value("lora_alpha", 0);
    if (r <= 0) {
      LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": r=" << r
                 << " must be > 0";
      return false;
    }
    if (r > config_.max_lora_rank) {
      LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": r=" << r
                 << " exceeds max_lora_rank=" << config_.max_lora_rank;
      return false;
    }
    out->r = r;
    out->scaling = lora_alpha > 0 ? static_cast<float>(lora_alpha) / r : 1.0f;

    // target_modules whitelist check.
    if (j.contains("target_modules")) {
      if (!j["target_modules"].is_array()) {
        LOG(ERROR) << "[LoRAAdapterLoader] " << path
                   << ": target_modules must be an array of strings";
        return false;
      }
      for (const auto& m : j["target_modules"]) {
        const std::string mod = m.get<std::string>();
        if (!contains(config_.lora_target_modules, mod)) {
          LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": target_modules "
                     << "contains '" << mod
                     << "' which is not in --lora-target-modules whitelist";
          return false;
        }
        out->target_modules.push_back(mod);
      }
    }

    // Optional base-model check: only enforced if the caller passed one.
    if (!expected_base_model_name.empty() &&
        j.contains("base_model_name_or_path")) {
      const std::string got = j["base_model_name_or_path"].get<std::string>();
      if (got != expected_base_model_name) {
        LOG(ERROR) << "[LoRAAdapterLoader] " << path
                   << ": base_model_name_or_path='" << got
                   << "' does not match expected='" << expected_base_model_name
                   << "'";
        return false;
      }
    }

    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path
               << ": invalid adapter_config.json fields: " << e.what();
    return false;
  }
}

std::optional<std::string> LoRAAdapterLoader::canonicalize_weight_name(
    const std::string& raw) const {
  // Strip HuggingFace fixed prefix.
  std::string s;
  if (raw.rfind(kHfPrefix, 0) == 0) {
    s = raw.substr(std::string(kHfPrefix).size());
  } else {
    // Some adapters skip the prefix. Tolerate it silently.
    s = raw;
  }

  // Look for the .lora_A.weight / .lora_B.weight suffix.
  const auto a_pos = s.rfind(kLoraATag);
  const auto b_pos = s.rfind(kLoraBTag);
  const bool is_a = a_pos != std::string::npos &&
                    a_pos + std::string(kLoraATag).size() == s.size();
  const bool is_b = b_pos != std::string::npos &&
                    b_pos + std::string(kLoraBTag).size() == s.size();
  if (!is_a && !is_b) {
    return std::nullopt;  // Not a LoRA weight (could be embedding LoRA which
                          // we skip in v1).
  }

  // Canonical form: "<subkey>#A" or "<subkey>#B" where subkey is the
  // dot-path of the module (e.g. "layers.5.self_attn.q_proj"). Using '#'
  // as the separator keeps subkey addressable via ordinary regex splits.
  const auto pos = is_a ? a_pos : b_pos;
  const std::string subkey = s.substr(0, pos);
  return subkey + (is_a ? "#A" : "#B");
}

bool LoRAAdapterLoader::load_weights(const std::string& safetensors_path,
                                     LoRAAdapter* out) const {
  auto sd = StateDictFromSafeTensor::load(safetensors_path);
  if (!sd) {
    LOG(ERROR) << "[LoRAAdapterLoader] safetensors load failed: "
               << safetensors_path;
    return false;
  }

  int32_t skipped = 0;
  int32_t taken = 0;
  for (const auto& [raw_name, tensor] : *sd) {
    auto canon = canonicalize_weight_name(raw_name);
    if (!canon) {
      ++skipped;
      LOG_EVERY_N(WARNING, 32)
          << "[LoRAAdapterLoader] skip unrecognised tensor '" << raw_name
          << "' in " << safetensors_path;
      continue;
    }
    // Sanity check: A is [r, ?], B is [?, r].
    const bool is_a =
        canon->size() >= 2 && canon->substr(canon->size() - 2) == "#A";
    const auto sizes = tensor.sizes();
    if (sizes.size() != 2) {
      LOG(ERROR) << "[LoRAAdapterLoader] '" << raw_name
                 << "' expected 2D tensor, got sizes.size()=" << sizes.size();
      return false;
    }
    if (is_a && sizes[0] != out->r) {
      LOG(ERROR) << "[LoRAAdapterLoader] '" << raw_name
                 << "' first dim=" << sizes[0] << " != r=" << out->r;
      return false;
    }
    if (!is_a && sizes[1] != out->r) {
      LOG(ERROR) << "[LoRAAdapterLoader] '" << raw_name
                 << "' second dim=" << sizes[1] << " != r=" << out->r;
      return false;
    }
    // MMAP_CLONE_APPLIED: `tensor` is at::from_blob(mmap_ptr,...) — the mmap
    // is owned by `sd` which is a local unique_ptr. Once this function
    // returns and sd destructs, the mmap is unmapped and `tensor` points
    // into freed memory. Materialize a heap-allocated copy right now.
    auto owned = torch::empty(
        tensor.sizes(),
        torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU));
    owned.copy_(tensor);
    out->tensors.emplace(*canon, std::move(owned));
    ++taken;
  }

  if (taken == 0) {
    LOG(ERROR) << "[LoRAAdapterLoader] no LoRA weights found in "
               << safetensors_path;
    return false;
  }
  LOG(INFO) << "[LoRAAdapterLoader] " << safetensors_path << " taken=" << taken
            << " skipped=" << skipped;
  return true;
}

std::optional<LoRAAdapter> LoRAAdapterLoader::load_dit(const LoRARequest& req) {
  namespace fs = std::filesystem;
  try {
    // 1. Resolve the safetensors file.
    fs::path weights_path(req.lora_path);
    if (fs::is_directory(weights_path)) {
      std::vector<fs::path> candidates;
      for (const auto& entry : fs::directory_iterator(weights_path)) {
        if (entry.is_regular_file() &&
            entry.path().extension() == ".safetensors") {
          candidates.emplace_back(entry.path());
        }
      }
      if (candidates.size() != 1) {
        LOG(ERROR)
            << "DiT LoRA directory requires exactly one safetensors file: "
            << weights_path;
        return std::nullopt;
      }
      weights_path = candidates.front();
    } else if (!fs::is_regular_file(weights_path) ||
               weights_path.extension() != ".safetensors") {
      LOG(ERROR) << "Invalid DiT LoRA file: " << weights_path;
      return std::nullopt;
    }

    // 2. Read and validate the safetensors header and metadata.
    std::ifstream stream(weights_path, std::ios::binary);
    std::array<unsigned char, 8> length_bytes{};
    stream.read(reinterpret_cast<char*>(length_bytes.data()),
                length_bytes.size());
    uint64_t header_size = 0;
    for (size_t i = 0; i < length_bytes.size(); ++i) {
      header_size |= static_cast<uint64_t>(length_bytes[i]) << (8 * i);
    }
    constexpr uint64_t kMaxHeaderBytes = 100 * 1024 * 1024;
    const uint64_t file_size = fs::file_size(weights_path);
    if (!stream || file_size < 8 || header_size == 0 ||
        header_size > kMaxHeaderBytes || header_size > file_size - 8) {
      LOG(ERROR) << "Invalid DiT LoRA safetensors header: " << weights_path;
      return std::nullopt;
    }
    std::string header(header_size, '\0');
    stream.read(header.data(), static_cast<std::streamsize>(header_size));
    if (!stream) {
      LOG(ERROR) << "Truncated DiT LoRA header: " << weights_path;
      return std::nullopt;
    }
    const nlohmann::json metadata = nlohmann::json::parse(header).value(
        "__metadata__", nlohmann::json::object());
    if (!metadata.is_object()) {
      LOG(ERROR) << "Invalid DiT LoRA metadata: " << weights_path;
      return std::nullopt;
    }

    // 3. Load A/B tensors, infer rank, and retain CPU-owned copies.
    LoRAAdapter adapter;
    adapter.request = req;
    auto state_dict = StateDictFromSafeTensor::load(weights_path.string());
    if (!state_dict) {
      return std::nullopt;
    }
    constexpr std::string_view kAlphaSuffix = ".alpha";
    constexpr std::array<std::string_view, 3> kPrefixes = {
        kHfPrefix, "diffusion_model.", "transformer."};
    constexpr std::array<std::pair<std::string_view, std::string_view>, 4>
        kSuffixAliases = {{{".lora_down.weight", kLoraATag},
                           {".lora_up.weight", kLoraBTag},
                           {".lora_A.default.weight", kLoraATag},
                           {".lora_B.default.weight", kLoraBTag}}};
    std::unordered_map<std::string, double> projection_alphas;
    for (const auto& [raw_name, tensor] : *state_dict) {
      std::string normalized = raw_name;
      for (const std::string_view prefix : kPrefixes) {
        if (normalized.starts_with(prefix)) {
          normalized.erase(/*index=*/0, prefix.size());
        }
      }
      if (normalized.ends_with(kAlphaSuffix)) {
        if (tensor.numel() != 1) {
          LOG(ERROR) << "Expected scalar alpha for " << raw_name << " in "
                     << weights_path;
          return std::nullopt;
        }
        normalized.resize(normalized.size() - kAlphaSuffix.size());
        const double alpha = tensor.item<double>();
        if (!std::isfinite(alpha) || alpha < 0 ||
            !projection_alphas.try_emplace(std::move(normalized), alpha)
                 .second) {
          LOG(ERROR) << "Invalid or duplicate alpha for " << raw_name << " in "
                     << weights_path;
          return std::nullopt;
        }
        continue;
      }
      for (const auto& [suffix, canonical] : kSuffixAliases) {
        if (normalized.ends_with(suffix)) {
          normalized.replace(
              normalized.size() - suffix.size(), suffix.size(), canonical);
          break;
        }
      }
      const auto name = canonicalize_weight_name(normalized);
      if (!name || tensor.dim() != 2 || !tensor.is_floating_point()) {
        LOG(ERROR) << "Unsupported DiT LoRA tensor " << raw_name << " in "
                   << weights_path;
        return std::nullopt;
      }
      const int64_t rank =
          name->back() == 'A' ? tensor.size(0) : tensor.size(1);
      if (rank <= 0 || rank > config_.max_lora_rank ||
          (adapter.r != 0 && adapter.r != rank)) {
        LOG(ERROR) << "Invalid/inconsistent DiT LoRA rank=" << rank << " for "
                   << raw_name << " in " << weights_path
                   << ", max_lora_rank=" << config_.max_lora_rank;
        return std::nullopt;
      }
      adapter.r = static_cast<int32_t>(rank);
      if (!adapter.tensors.emplace(*name, tensor.clone()).second) {
        LOG(ERROR) << "Duplicate DiT LoRA tensor " << *name << " in "
                   << weights_path;
        return std::nullopt;
      }
    }
    if (adapter.tensors.empty()) {
      LOG(ERROR) << "No DiT LoRA tensors: " << weights_path;
      return std::nullopt;
    }

    // 4. Validate A/B pairs and collect projection targets.
    adapter.target_modules.reserve(adapter.tensors.size() / 2);
    for (const auto& entry : adapter.tensors) {
      const std::string& name = entry.first;
      const std::string target = name.substr(0, name.size() - 2);
      const bool is_a = name.back() == 'A';
      if (!adapter.tensors.contains(target + (is_a ? "#B" : "#A"))) {
        LOG(ERROR) << "Incomplete DiT LoRA A/B pair " << target << " in "
                   << weights_path;
        return std::nullopt;
      }
      if (is_a) {
        adapter.target_modules.emplace_back(target);
      }
    }
    for (const auto& entry : projection_alphas) {
      if (!adapter.tensors.contains(entry.first + "#A")) {
        LOG(ERROR) << "Alpha without a DiT LoRA target " << entry.first
                   << " in " << weights_path;
        return std::nullopt;
      }
    }

    // 5. Resolve alpha and fold per-projection scales into B.
    double metadata_alpha = adapter.r;
    if (metadata.contains("alpha")) {
      const nlohmann::json parsed =
          nlohmann::json::parse(metadata.at("alpha").get<std::string>());
      if (!parsed.is_number()) {
        LOG(ERROR) << "Invalid DiT LoRA metadata alpha in " << weights_path;
        return std::nullopt;
      }
      metadata_alpha = parsed.get<double>();
    }
    const double default_scale = metadata_alpha / adapter.r;
    if (metadata_alpha < 0 ||
        !std::isfinite(static_cast<float>(default_scale))) {
      LOG(ERROR) << "Invalid DiT LoRA alpha in " << weights_path;
      return std::nullopt;
    }

    const bool fold_scales = !projection_alphas.empty();
    adapter.scaling = fold_scales ? 1.0f : static_cast<float>(default_scale);
    if (fold_scales) {
      for (const std::string& target : adapter.target_modules) {
        const auto alpha = projection_alphas.find(target);
        const double scale = alpha == projection_alphas.end()
                                 ? default_scale
                                 : alpha->second / adapter.r;
        if (!std::isfinite(static_cast<float>(scale))) {
          LOG(ERROR) << "Invalid DiT LoRA alpha for " << target << " in "
                     << weights_path;
          return std::nullopt;
        }
        if (scale != 1.0) {
          adapter.tensors.at(target + "#B").mul_(scale);
        }
      }
    }
    LOG(INFO) << "DiT LoRA '" << req.lora_name << "' rank=" << adapter.r
              << ", default_alpha=" << metadata_alpha
              << ", scaling=" << adapter.scaling
              << ", folded_scales=" << fold_scales
              << ", targets=" << adapter.target_modules.size();
    return adapter;
  } catch (const nlohmann::json::exception& error) {
    LOG(ERROR) << "Invalid DiT LoRA JSON for " << req.lora_path << ": "
               << error.what();
    return std::nullopt;
  } catch (const fs::filesystem_error& error) {
    LOG(ERROR) << "Invalid DiT LoRA path " << req.lora_path << ": "
               << error.what();
    return std::nullopt;
  }
}

}  // namespace xllm
