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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "models/vlm/mposition/mposition.h"
#include "processors/multimodal_processor.h"

namespace xllm {

// C++-owned text encoder handed to the python DiT pipeline as a proxy object
// (the "form B" encoder). The VLM (e.g. Qwen3-VL) keeps its C++ KV cache,
// TP group, tokenizer and multimodal processor; the proxy exposes ONE python
// entry point — encode_prompts(prompts, images) -> (embeds, masks) — so a
// request crosses the language boundary once for text encoding instead of
// once per tensor. The prompt-embedding contract matches the in-process C++
// pipeline: pre-norm hidden states from kPromptEmbeddingStartIndex onward.
class VlmTextEncoderProxy {
 public:
  static constexpr int64_t kPromptEmbeddingStartIndex = 34;

  VlmTextEncoderProxy(std::unique_ptr<CausalVLM> encoder,
                      ModelArgs text_encoder_model_args,
                      torch::TensorOptions options)
      : encoder_(std::move(encoder)),
        text_encoder_model_args_(std::move(text_encoder_model_args)),
        options_(std::move(options)),
        device_(options_.device()) {
    CHECK(encoder_ != nullptr) << "VlmTextEncoderProxy requires an encoder";
    text_encoder_empty_kv_caches_.resize(
        static_cast<size_t>(text_encoder_model_args_.n_layers()));
  }

  CausalVLM* encoder() { return encoder_.get(); }

  void set_tokenizer(std::shared_ptr<Tokenizer> tokenizer) {
    tokenizer_ = std::move(tokenizer);
  }

  void set_multimodal_processor(
      std::unique_ptr<MultimodalProcessorBase> processor) {
    multimodal_processor_ = std::move(processor);
  }

  void set_mposition_generator(std::unique_ptr<MPositionGenerator> generator) {
    mposition_generator_ = std::move(generator);
  }

  // Encode one prompt per batch entry against the reference images, and pad
  // to a common sequence length. Mirrors the C++ pipeline's encode_prompts.
  std::pair<torch::Tensor, torch::Tensor> encode_prompts(
      const std::vector<std::string>& prompts,
      const std::vector<torch::Tensor>& raw_images) {
    const int64_t batch_size = static_cast<int64_t>(prompts.size());
    CHECK(!raw_images.empty())
        << "VlmTextEncoderProxy requires reference images";
    CHECK(tokenizer_ != nullptr) << "Qwen3-VL tokenizer is not loaded";
    CHECK(multimodal_processor_ != nullptr)
        << "Qwen3-VL multimodal processor is not loaded";
    CHECK(mposition_generator_ != nullptr)
        << "Qwen3-VL position generator is not loaded";

    std::vector<torch::Tensor> embeddings;
    std::vector<torch::Tensor> masks;
    embeddings.reserve(static_cast<size_t>(batch_size));
    masks.reserve(static_cast<size_t>(batch_size));
    int64_t max_sequence_length = 0;
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      auto [embedding, mask] =
          encode_single_prompt(prompts[static_cast<size_t>(batch_index)],
                               raw_images,
                               batch_index,
                               batch_size);
      max_sequence_length = std::max(max_sequence_length, embedding.size(1));
      embeddings.emplace_back(std::move(embedding));
      masks.emplace_back(std::move(mask));
    }
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      embeddings[static_cast<size_t>(batch_index)] = pad_seq(
          embeddings[static_cast<size_t>(batch_index)], max_sequence_length);
      masks[static_cast<size_t>(batch_index)] =
          pad_seq(masks[static_cast<size_t>(batch_index)], max_sequence_length);
    }
    return {torch::cat(embeddings, /*dim=*/0), torch::cat(masks, /*dim=*/0)};
  }

  // JoyOV2 T2VA text path: plain chat wrap, no images, no multimodal
  // processor — mirrors JoyOV2Pipeline::encode_one_prompt. Returns ONE
  // unpadded [T, D] pre-norm residual per prompt (cast to the pipeline
  // options); the caller derives joyo_text_seqlens from the per-prompt
  // lengths and applies normalize_text_emb.
  std::vector<torch::Tensor> encode_joyo_prompts(
      const std::vector<std::string>& prompts) {
    CHECK(tokenizer_ != nullptr) << "Qwen3-VL tokenizer is not loaded";
    std::vector<torch::Tensor> embeds;
    embeds.reserve(prompts.size());
    for (const std::string& prompt : prompts) {
      const std::string wrapped =
          std::string(kJoyoChatPrefix) + prompt + kJoyoChatSuffix;
      std::vector<int32_t> token_ids;
      CHECK(tokenizer_->encode(wrapped,
                               &token_ids,
                               /*add_special_tokens=*/false))
          << "JoyOV2 Qwen3-VL tokenizer encode failed";
      CHECK(!token_ids.empty()) << "JoyOV2 Qwen3-VL encoded empty prompt";

      torch::Tensor tokens =
          torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kInt32))
              .to(device_);
      torch::Tensor positions = torch::arange(
          tokens.numel(),
          torch::TensorOptions().dtype(torch::kInt).device(device_));
      ModelInputParams input_params = build_plain_text_input(tokens);
      input_params.embedding.input_embedding =
          encoder_->get_input_embeddings(tokens, input_params);
      ModelOutput model_output = encoder_->forward(
          tokens, positions, text_encoder_empty_kv_caches_, input_params);
      CHECK(model_output.residual.defined())
          << "JoyOV2 requires Qwen3-VL pre-norm residual (TORCH backend)";
      torch::Tensor residual = model_output.residual;
      if (residual.dim() == 3 && residual.size(0) == 1) {
        residual = residual.squeeze(0);
      }
      CHECK_EQ(residual.dim(), 2)
          << "JoyOV2 encoder residual must be [T,D], got " << residual.sizes();
      embeds.emplace_back(residual.to(options_));
    }
    return embeds;
  }

 private:
  std::pair<torch::Tensor, torch::Tensor> encode_single_prompt(
      const std::string& prompt,
      const std::vector<torch::Tensor>& raw_images,
      int64_t batch_index,
      int64_t batch_size) {
    std::vector<torch::Tensor> images =
        prepare_vl_images(raw_images, batch_index, batch_size);
    std::vector<MMInputItem> input_items;
    input_items.reserve(images.size());
    for (torch::Tensor& image : images) {
      MMInputItem item;
      item.type = MMType::IMAGE;
      item.decode_image = std::move(image);
      input_items.emplace_back(std::move(item));
    }
    MMInput multimodal_input;
    multimodal_input.insert(input_items);
    MMData mm_data;
    CHECK(multimodal_processor_->process_multimodal(multimodal_input, mm_data))
        << "JoyImageEditPlus Qwen3-VL image preprocessing failed";

    std::string chat_prompt = build_qwen3_vl_prompt(prompt, images.size());
    std::vector<int32_t> token_ids;
    CHECK(
        multimodal_processor_->process_prompt(chat_prompt, mm_data, token_ids))
        << "JoyImageEditPlus Qwen3-VL prompt processing failed";
    UpdateMMItemScheduleStateVisitor schedule_visitor(
        /*computed_token_num=*/0,
        static_cast<int32_t>(token_ids.size()),
        /*seq_idx=*/0);
    CHECK(mm_data.foreach (schedule_visitor))
        << "JoyImageEditPlus failed to schedule Qwen3-VL multimodal inputs";
    auto [positions, mrope_position_delta] = mposition_generator_->generate(
        token_ids, mm_data, text_encoder_model_args_);
    static_cast<void>(mrope_position_delta);

    torch::Tensor tokens =
        torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kInt32))
            .to(device_);
    positions = positions.to(device_);
    ModelInputParams input_params = build_text_encoder_input(tokens, mm_data);
    ModelOutput model_output = encoder_->forward(
        tokens, positions, text_encoder_empty_kv_caches_, input_params);
    CHECK(model_output.residual.defined())
        << "JoyImageEditPlus requires Qwen3-VL pre-norm hidden states from "
           "the TORCH backend.";
    CHECK_GT(model_output.residual.size(0), kPromptEmbeddingStartIndex)
        << "JoyImageEditPlus Qwen3-VL prompt is shorter than the embedding "
           "prefix";

    torch::Tensor prompt_embeddings =
        model_output.residual.slice(/*dim=*/0, kPromptEmbeddingStartIndex)
            .unsqueeze(0)
            .to(options_);
    torch::Tensor prompt_mask =
        torch::ones({1, prompt_embeddings.size(1)},
                    torch::TensorOptions().device(device_).dtype(torch::kLong));
    return {prompt_embeddings, prompt_mask};
  }

  // Plain-text prefill params, no multimodal inputs — the input layout
  // JoyOV2Pipeline::build_text_encoder_input uses for its T2VA text path.
  ModelInputParams build_plain_text_input(const torch::Tensor& tokens) const {
    CHECK_LE(tokens.numel(), std::numeric_limits<int32_t>::max())
        << "JoyOV2 Qwen3-VL prompt is too long";
    const int32_t sequence_length = static_cast<int32_t>(tokens.numel());
    CHECK_GT(sequence_length, 0) << "JoyOV2 Qwen3-VL prompt must not be empty";

    ModelInputParams params;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.q_max_seq_len = sequence_length;
    params.meta.kv_max_seq_len = sequence_length;
    params.meta.batch_forward_type = BatchForwardType::PREFILL;
    params.prefill_without_cache = true;
    params.attention.host.q_seq_lens = {sequence_length};
    params.attention.host.kv_seq_lens = {sequence_length};
#if defined(USE_NPU)
    params.attention.host.q_cu_seq_lens = {sequence_length};
#else
    params.attention.host.q_cu_seq_lens = {0, sequence_length};
#endif
    params.attention.device.q_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
    params.attention.device.kv_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#if defined(USE_NPU)
    params.attention.device.q_cu_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#else
    params.attention.device.q_cu_seq_lens =
        torch::tensor({0, sequence_length}, torch::kInt).to(tokens.device());
#endif
    return params;
  }

  ModelInputParams build_text_encoder_input(const torch::Tensor& tokens,
                                            const MMData& mm_data) {
    CHECK_LE(tokens.numel(), std::numeric_limits<int32_t>::max())
        << "JoyImageEditPlus Qwen3-VL prompt is too long";
    const int32_t sequence_length = static_cast<int32_t>(tokens.numel());
    CHECK_GT(sequence_length, 0)
        << "JoyImageEditPlus Qwen3-VL prompt must not be empty";

    ModelInputParams params;
    params.meta.num_sequences = 1;
    params.meta.actual_num_sequences = 1;
    params.meta.q_max_seq_len = sequence_length;
    params.meta.kv_max_seq_len = sequence_length;
    params.meta.batch_forward_type = BatchForwardType::PREFILL;
    params.prefill_without_cache = true;
    params.attention.host.q_seq_lens = {sequence_length};
    params.attention.host.kv_seq_lens = {sequence_length};
#if defined(USE_NPU)
    params.attention.host.q_cu_seq_lens = {sequence_length};
#else
    params.attention.host.q_cu_seq_lens = {0, sequence_length};
#endif
    params.attention.device.q_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
    params.attention.device.kv_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#if defined(USE_NPU)
    params.attention.device.q_cu_seq_lens =
        torch::tensor({sequence_length}, torch::kInt).to(tokens.device());
#else
    params.attention.device.q_cu_seq_lens =
        torch::tensor({0, sequence_length}, torch::kInt).to(tokens.device());
#endif

    MMBatchData mm_batch(std::vector<MMData>{mm_data});
    EncoderInputGatherVisitor input_gather;
    mm_batch.foreach (input_gather);
    CHECK(input_gather.finish(mm_batch))
        << "JoyImageEditPlus failed to gather Qwen3-VL encoder inputs";
    mm_batch.to(tokens.device());

    ModelInputParams multimodal_params;
    multimodal_params.multimodal.mm_data = mm_batch;
    MMDict multimodal_embeddings = encoder_->encode(multimodal_params);
    EncoderOutputScatterVisitor output_scatter(multimodal_embeddings);
    CHECK(mm_batch.foreach (output_scatter));
    CHECK(output_scatter.finish())
        << "JoyImageEditPlus failed to scatter Qwen3-VL embeddings";

    EncoderEmbeddingGatherVisitor embedding_gather(
        tokens.device(),
        mm_batch.type(),
        params.attention.host.kv_seq_lens,
        params.attention.host.q_seq_lens);
    CHECK(mm_batch.foreach (embedding_gather));
    CHECK(embedding_gather.finish(mm_batch))
        << "JoyImageEditPlus failed to gather Qwen3-VL embeddings";
    params.multimodal.mm_data = std::move(mm_batch);
    params.embedding.input_embedding =
        encoder_->get_input_embeddings(tokens, params);
    return params;
  }

  std::vector<torch::Tensor> prepare_vl_images(
      const std::vector<torch::Tensor>& raw_images,
      int64_t batch_index,
      int64_t batch_size) const {
    std::vector<torch::Tensor> images;
    images.reserve(raw_images.size());
    for (const torch::Tensor& raw_image : raw_images) {
      CHECK(raw_image.dim() == 3 || raw_image.dim() == 4)
          << "JoyImageEditPlus image input must be [C,H,W] or [B,C,H,W]";
      torch::Tensor image;
      if (raw_image.dim() == 4) {
        CHECK_EQ(raw_image.size(0), batch_size)
            << "JoyImageEditPlus image batch must match prompt batch";
        image = raw_image[batch_index];
      } else {
        CHECK_EQ(batch_size, 1)
            << "Unbatched JoyImageEditPlus images require batch_size=1";
        image = raw_image;
      }
      image = image.to(torch::kCPU).to(torch::kFloat32);
      const float max_value = image.max().item<float>();
      image = max_value <= 1.1f ? image.clamp(0.0f, 1.0f) * 255.0f
                                : image.clamp(0.0f, 255.0f);
      images.emplace_back(std::move(image));
    }
    return images;
  }

  std::string build_qwen3_vl_prompt(const std::string& prompt,
                                    size_t image_count) const {
    std::string chat_prompt =
        "<|im_start|>system\n \\nDescribe the image by detailing the "
        "color, shape, size, texture, quantity, text, spatial relationships "
        "of the objects and background:<|im_end|>\n<|im_start|>user\n";
    for (size_t image_index = 0; image_index < image_count; ++image_index) {
      chat_prompt += "<|vision_start|><|image_pad|><|vision_end|>";
    }
    chat_prompt += prompt;
    chat_prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return chat_prompt;
  }

  static torch::Tensor pad_seq(const torch::Tensor& x, int64_t target_len) {
    const int64_t cur = x.size(1);
    if (cur >= target_len) {
      return x.slice(1, cur - target_len, cur);
    }
    const int64_t pad = target_len - cur;
    std::vector<int64_t> shape = x.sizes().vec();
    shape[1] = pad;
    auto zeros = torch::zeros(shape, x.options());
    return torch::cat({x, zeros}, 1);
  }

  // JoyOV2 Qwen3-VL chat wrap (JoyOV2Pipeline kJoyoChatPrefix/Suffix): the
  // T2VA caption path uses a bare user turn — no system prompt, no vision
  // tokens — and consumes the FULL pre-norm residual (no prefix slice).
  static constexpr const char* kJoyoChatPrefix = "<|im_start|>user\n";
  static constexpr const char* kJoyoChatSuffix =
      "<|im_end|>\n<|im_start|>assistant\n";

  std::unique_ptr<CausalVLM> encoder_;
  ModelArgs text_encoder_model_args_;
  torch::TensorOptions options_;
  torch::Device device_ = torch::kCPU;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<MultimodalProcessorBase> multimodal_processor_;
  std::unique_ptr<MPositionGenerator> mposition_generator_;
  std::vector<KVCache> text_encoder_empty_kv_caches_;
};

}  // namespace xllm
