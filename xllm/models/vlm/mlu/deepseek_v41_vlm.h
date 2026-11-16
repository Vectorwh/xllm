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

#include <memory>
#include <utility>

#include "core/layers/mlu/deepseek_v41/deepseek_v41_vision.h"
#include "models/llm/mlu/deepseek_v41.h"
#include "models/model_registry.h"
#include "models/vlm/deepseek_v41_vlm.h"

namespace xllm::mlu::model {

class DeepseekV41ForConditionalGenerationImpl final : public torch::nn::Module {
 public:
  explicit DeepseekV41ForConditionalGenerationImpl(
      const ModelContext& context) {
    CHECK_EQ(context.get_model_args().mm_projection_dim(),
             context.get_model_args().hidden_size());
    language_model_ =
        register_module("language_model", DeepseekV41ForCausalLM(context));
    vision_ = register_module("vision", layer::DeepseekV41Vision(context));
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input) {
    const auto& data = input.multimodal.mm_data;
    const auto pixels = data.get<torch::Tensor>("pixel_values");
    const auto grid = data.get<torch::Tensor>("image_grid_thw");
    CHECK_EQ(pixels.has_value(), grid.has_value())
        << "Image pixels and grid must be supplied together";
    if (!pixels.has_value()) {
      return {};
    }
    return {{"image|embedding", vision_->encode(*pixels, *grid)}};
  }

  torch::Tensor get_input_embeddings(const torch::Tensor& tokens,
                                     const ModelInputParams& input) {
    torch::Tensor embeddings = language_model_->get_input_embeddings(tokens);
    const auto& data = input.multimodal.mm_data;
    const auto images = data.get<torch::Tensor>("image|embedding");
    const auto mask = data.get<torch::Tensor>("image|mask");
    CHECK_EQ(images.has_value(), mask.has_value())
        << "Image embeddings and mask must be supplied together";
    if (!images.has_value()) {
      return embeddings;
    }
    CHECK_EQ(mask->dim(), 1);
    CHECK_EQ(mask->numel(), embeddings.size(0));
    CHECK_EQ(images->dim(), 2);
    CHECK_EQ(images->size(1), embeddings.size(1));
    embeddings.index_put_({mask->to(embeddings.device(), torch::kBool)},
                          images->to(embeddings.options()));
    return embeddings;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& input) {
    return language_model_->forward(tokens, positions, caches, input);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) {
    return language_model_->logits(hidden, selected);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected,
                       torch::Tensor& output_hidden) {
    return language_model_->logits(hidden, selected, output_hidden);
  }

  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return language_model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& input) {
    language_model_->prepare_graph_forward_metadata(state, positions, input);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state : loader->get_state_dicts()) {
      vision_->load_state_dict(*state);
    }
    vision_->verify_loaded_weights();
    language_model_->load_model(std::move(loader));
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) { language_model_->set_lm_head(head); }
  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }
  void set_word_embedding(layer::WordEmbedding& embedding) {
    language_model_->set_word_embedding(embedding);
  }

 private:
  DeepseekV41ForCausalLM language_model_{nullptr};
  layer::DeepseekV41Vision vision_{nullptr};
};
TORCH_MODULE(DeepseekV41ForConditionalGeneration);

REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(deepseek_v41_vlm,
                                       deepseek_v41,
                                       DeepseekV41ForConditionalGeneration);

}  // namespace xllm::mlu::model
