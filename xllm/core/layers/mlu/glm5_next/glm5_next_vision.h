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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"

namespace xllm::layer {

class Glm5NextVisionNormImpl;
class Glm5NextVisionMergerImpl;

// Prepared outside device execution; reusable for the same ordered media grids.
struct Glm5NextVisionMetadata {
  int64_t num_patches = 0;
  int64_t max_seq_len = 0;
  std::vector<int64_t> output_sizes;
  torch::Tensor cu_seq_lens;
  torch::Tensor cos;
  torch::Tensor sin;
};

// GlmOcr vision tower. Vision weights are replicated across TP/CP ranks;
// quantization and language-model head sharding do not apply to this tower.
class Glm5NextVisionImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextVisionImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& pixels,
                        const torch::Tensor& grid_thw);
  Glm5NextVisionMetadata prepare_metadata(const torch::Tensor& grid_thw) const;
  torch::Tensor forward_prepared(const torch::Tensor& pixels,
                                 const Glm5NextVisionMetadata& metadata);
  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

 private:
  ModelArgs args_;
  torch::TensorOptions options_;
  torch::nn::Conv3d patch_projection_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  torch::nn::Conv2d downsample_{nullptr};
  std::shared_ptr<Glm5NextVisionNormImpl> post_layernorm_;
  std::shared_ptr<Glm5NextVisionMergerImpl> merger_;
  torch::Tensor inv_freq_;
  std::unordered_set<std::string> loaded_weights_;
};
TORCH_MODULE(Glm5NextVision);

}  // namespace xllm::layer
