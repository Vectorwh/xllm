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
#include <string>
#include <utility>
#include <vector>

#include "core/layers/mlu/glm5_next/glm5_next_vision.h"
#include "models/llm/mlu/glm5_next.h"
#include "models/model_registry.h"

namespace xllm::mlu::model {

class Glm5NextForConditionalGenerationImpl final : public torch::nn::Module {
 public:
  explicit Glm5NextForConditionalGenerationImpl(const ModelContext& context)
      : args_(context.get_model_args()) {
    CHECK_EQ(args_.mm_projection_dim(), args_.hidden_size());
    CHECK_EQ(args_.mm_image_merge_size(), args_.mm_spatial_merge_size());
    visual_ = register_module("visual", layer::Glm5NextVision(context));
    if (!args_.encoder_embedding_mode()) {
      language_model_ =
          register_module("language_model", Glm5NextForCausalLM(context));
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& params) {
    const auto& data = params.multimodal.mm_data;
    MMDict outputs;
    auto encode_modality = [&](const std::string& pixels_key,
                               const std::string& grid_key,
                               const std::string& embedding_key) {
      const auto pixels = data.get<torch::Tensor>(pixels_key);
      const auto grid = data.get<torch::Tensor>(grid_key);
      CHECK_EQ(pixels.has_value(), grid.has_value());
      if (!pixels.has_value()) {
        return;
      }
      const auto metadata = visual_->prepare_metadata(grid.value());
      const torch::Tensor features =
          visual_->forward_prepared(pixels.value(), metadata);
      outputs[embedding_key] = features.split(metadata.output_sizes, /*dim=*/0);
    };
    encode_modality("pixel_values", "image_grid_thw", "image|embedding");
    encode_modality("pixel_values_videos", "video_grid_thw", "video|embedding");
    return outputs;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor& tokens,
                                     const ModelInputParams& params) {
    torch::Tensor embeddings = language_model_->get_input_embeddings(tokens);
    const auto& data = params.multimodal.mm_data;
    auto merge = [&](const std::string& embedding_key,
                     const std::string& mask_key,
                     MMType type) {
      const auto features = data.get<torch::Tensor>(embedding_key);
      CHECK(!data.has(type) || features.has_value())
          << "Missing prepared multimodal embeddings: " << embedding_key;
      if (!features.has_value()) {
        return;
      }
      const auto mask = data.get<torch::Tensor>(mask_key);
      CHECK(mask.has_value()) << "Missing multimodal mask: " << mask_key;
      CHECK_EQ(features->dim(), 2);
      CHECK_EQ(features->size(1), embeddings.size(1));
      CHECK_EQ(mask->dim(), 1);
      CHECK_EQ(mask->size(0), tokens.numel());
      CHECK_EQ(mask->scalar_type(), torch::kBool);
      const torch::Tensor rows = mask->nonzero().flatten();
      embeddings.index_copy_(/*dim=*/0,
                             rows.to(embeddings.device()),
                             features->to(embeddings.options()));
    };
    merge("image|embedding", "image|mask", MMType::IMAGE);
    merge("video|embedding", "video|mask", MMType::VIDEO);
    return embeddings;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& caches,
                      const ModelInputParams& params) {
    return language_model_(tokens, positions, caches, params);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected) {
    return language_model_->logits(hidden, selected);
  }

  torch::Tensor logits(const torch::Tensor& hidden,
                       const torch::Tensor& selected,
                       torch::Tensor& out_hidden) {
    return language_model_->logits(hidden, selected, out_hidden);
  }

  bool is_hybrid_linear_attention() { return true; }
  bool requires_graph_forward_metadata() { return true; }

  std::unique_ptr<ModelGraphMetadataState>
  create_graph_forward_metadata_state() {
    return language_model_->create_graph_forward_metadata_state();
  }

  void prepare_graph_forward_metadata(ModelGraphMetadataState* state,
                                      const torch::Tensor& positions,
                                      ModelInputParams& params) {
    language_model_->prepare_graph_forward_metadata(state, positions, params);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }
    visual_->verify_loaded_weights();
    if (!args_.encoder_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
      CHECK(language_model_->has_loaded_vocab_weights())
          << "Missing GLM5-Next vocabulary weights";
    }
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
  ModelArgs args_;
  layer::Glm5NextVision visual_{nullptr};
  Glm5NextForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Glm5NextForConditionalGeneration);

REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(glm5_next_vlm,
                                       glm5_next,
                                       Glm5NextForConditionalGeneration);

}  // namespace xllm::mlu::model
