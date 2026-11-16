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

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm::layer {

// Replicated ViT and spatial aligner. Outputs only non-pad image-block rows;
// compressor-alignment pads retain their language-model token embeddings.
class DeepseekV41VisionImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV41VisionImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state);
  void verify_loaded_weights() const;
  std::vector<torch::Tensor> encode(const torch::Tensor& pixels,
                                    const torch::Tensor& grid) const;
  torch::Tensor forward(const torch::Tensor& patches,
                        int64_t n_h,
                        int64_t n_w) const;

 private:
  void add_weight(const std::string& name,
                  std::vector<int64_t> shape,
                  bool use_fp32 = false);
  torch::Tensor rms_norm(const torch::Tensor& x,
                         const torch::Tensor& weight) const;
  torch::Tensor vision_forward(const torch::Tensor& patches,
                               int64_t n_h,
                               int64_t n_w) const;
  torch::Tensor align(const torch::Tensor& features,
                      int64_t n_h,
                      int64_t n_w) const;

  ModelArgs args_;
  torch::TensorOptions options_;
  std::unordered_map<std::string, torch::Tensor> weights_;
  std::unordered_set<std::string> loaded_;
};
TORCH_MODULE(DeepseekV41Vision);

}  // namespace xllm::layer
