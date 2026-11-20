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

#include "models/llm/mlu/deepseek_v41_dspark_quant_args.h"

#include <array>

namespace xllm::mlu::model {
namespace {
using Shards = std::vector<std::unique_ptr<StateDict>>;

// Only inspect headers of the same mmap-backed tensors the loader will consume.
torch::Tensor find_tensor(const Shards& shards, const std::string& key) {
  const std::string hf_key = "model." + key;
  for (const auto& shard : shards) {
    if (shard->has(key)) {
      return shard->get_tensor(key);
    }
    if (shard->has(hf_key)) {
      return shard->get_tensor(hf_key);
    }
  }
  return {};
}

void check_tensor(const torch::Tensor& tensor,
                  const std::string& key,
                  torch::ScalarType dtype,
                  torch::IntArrayRef shape) {
  CHECK(tensor.defined()) << "Missing " << key;
  CHECK(tensor.scalar_type() == dtype && tensor.sizes() == shape)
      << key << ": expected " << dtype << " " << shape << ", got "
      << tensor.scalar_type() << " " << tensor.sizes();
}

bool is_quantized(const Shards& shards,
                  const std::string& module,
                  int64_t in,
                  int64_t out,
                  bool packed = false) {
  auto weight = find_tensor(shards, module + ".weight");
  auto scale = find_tensor(shards, module + ".weight_scale");
  auto smooth = find_tensor(shards, module + ".smooth");
  CHECK(weight.defined()) << "Missing " << module << ".weight";
  if (weight.scalar_type() == torch::kBFloat16) {
    check_tensor(weight, module + ".weight", torch::kBFloat16, {out, in});
    CHECK(!scale.defined() && !smooth.defined())
        << "Unexpected quantization tensors for BF16 " << module;
    return false;
  }
  check_tensor(
      weight, module + ".weight", torch::kInt8, {out, packed ? in / 2 : in});
  check_tensor(scale,
               module + ".weight_scale",
               torch::kBFloat16,
               {out, packed ? in / 128 : 1});
  check_tensor(smooth, module + ".smooth", torch::kBFloat16, {in});
  return true;
}
}  // namespace

QuantArgs build_dspark_decoder_quant_args(const ModelArgs& args,
                                          const QuantArgs& checkpoint,
                                          const Shards& shards) {
  if (!checkpoint.is_compressed_tensors_w8a8_dynamic()) {
    return {};
  }
  CompressedQuantGroup dense{{}, 8, 0, true};
  CompressedQuantGroup routed{{}, 4, 128, true};
  const auto check_scheme = [&](const std::string& module, bool packed) {
    const auto scheme = checkpoint.module_quant_scheme(module);
    CHECK(scheme && scheme->bits == (packed ? 4 : 8) &&
          scheme->group_size == (packed ? 128 : 0) && scheme->preserve_smooth)
        << "Incompatible compressed-tensors scheme for " << module;
  };
  auto add_dense = [&](const std::string& module, int64_t in, int64_t out) {
    if (is_quantized(shards, module, in, out)) {
      check_scheme(module, false);
      dense.targets.emplace_back(module);
    }
  };
  const int64_t heads = args.n_heads() * args.head_dim();
  const int64_t output_rank = args.o_groups() * args.o_lora_rank();
  for (int32_t id = 0; id < args.n_layers(); ++id) {
    const std::string layer = "mtp." + std::to_string(id);
    add_dense(layer + ".attn.wo_a", heads / args.o_groups(), output_rank);
    add_dense(layer + ".attn.wq_b", args.q_lora_rank(), heads);
    add_dense(layer + ".attn.wo_b", output_rank, args.hidden_size());
    std::optional<bool> packed;
    for (int64_t expert = -1; expert < args.n_routed_experts(); ++expert) {
      const bool shared = expert == -1;
      if (shared && args.n_shared_experts() == 0) {
        continue;
      }
      const std::string prefix =
          layer + ".ffn." +
          (shared ? "shared_experts."
                  : "experts." + std::to_string(expert) + ".");
      const int64_t width =
          args.moe_intermediate_size() * (shared ? args.n_shared_experts() : 1);
      for (const auto& [alias, canonical] :
           std::array<std::pair<std::string, std::string>, 3>{
               {{"w1", "gate_proj"}, {"w3", "up_proj"}, {"w2", "down_proj"}}}) {
        const bool down = alias == "w2";
        const std::string module =
            find_tensor(shards, prefix + alias + ".weight").defined()
                ? prefix + alias
                : prefix + canonical;
        const bool integer = is_quantized(shards,
                                          module,
                                          down ? width : args.hidden_size(),
                                          down ? args.hidden_size() : width,
                                          !shared);
        if (integer) {
          check_scheme(checkpoint.module_quant_scheme(module)
                           ? module
                           : prefix + canonical,
                       !shared);
        }
        if (shared) {
          if (integer) {
            dense.targets.emplace_back(prefix + canonical);
          }
        } else {
          CHECK(!packed.has_value() || *packed == integer)
              << "Inconsistent fused expert storage: " << module;
          packed = integer;
        }
      }
    }
    if (packed.value_or(false)) {
      routed.targets.emplace_back(layer + ".ffn.experts.*");
    }
  }
  if (dense.targets.empty() && routed.targets.empty()) {
    return {};
  }
  QuantArgs quant = checkpoint;
  quant.compressed_groups().clear();
  quant.ignored_modules().clear();
  if (!dense.targets.empty()) {
    quant.compressed_groups().emplace_back(std::move(dense));
  }
  if (!routed.targets.empty()) {
    quant.compressed_groups().emplace_back(std::move(routed));
  }
  return quant;
}
}  // namespace xllm::mlu::model
