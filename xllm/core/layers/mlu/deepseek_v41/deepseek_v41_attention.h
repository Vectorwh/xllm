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

#pragma once

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "layers/common/linear.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_indexer.h"

namespace xllm::layer {
class DeepseekV41AttentionImpl final : public torch::nn::Module {
 public:
  DeepseekV41AttentionImpl(const ModelContext& context,
                           int32_t layer_id,
                           const std::string& layer_stack = "layers");
  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  void write_context_kv(const torch::Tensor& hidden,
                        const torch::Tensor& positions,
                        const torch::Tensor& rope,
                        const torch::Tensor& cache_slots,
                        KVCache& cache);
  torch::Tensor forward(
      const torch::Tensor& x,
      const torch::Tensor& positions,
      const torch::Tensor& rope,
      std::vector<KVCache>& caches,
      const ModelInputParams& input,
      Csa2Context& shared,
      const mlu_v4_cp::DeepseekV4CpContext* cp_context = nullptr);

 private:
  void load_output_weights(const StateDict& state);
  torch::Tensor project_query(const torch::Tensor& qr,
                              const torch::Tensor& positions,
                              const torch::Tensor& rope);
  torch::Tensor project_output(const torch::Tensor& output,
                               const torch::Tensor& positions,
                               const torch::Tensor& rope);
  ModelArgs args_;
  int32_t layer_id_;
  std::string attn_prefix_;
  int32_t source_ = -1;
  int32_t ratio_ = 0;
  bool index_source_ = false;
  ProcessGroup* tp_group_;
  int64_t rank_;
  int64_t tp_size_;
  torch::TensorOptions options_;
  std::unordered_map<std::string, torch::Tensor> weights_;
  ColumnParallelLinear wq_b_{nullptr};
  DeepseekV41Indexer indexer_{nullptr};
};
TORCH_MODULE(DeepseekV41Attention);
}  // namespace xllm::layer
