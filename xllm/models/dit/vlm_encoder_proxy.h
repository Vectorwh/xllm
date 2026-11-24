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
#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {

// C++-owned text encoder proxy for the JoyOV2 caption pipeline.
class VlmTextEncoderProxy {
 public:
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

  // JoyOV2 streams: mirror JoyOV2Pipeline::set_text_encoder_on_device. The
  // text encoder is only needed inside the encode window; the python
  // pipeline offloads it right after (and flushes the NPU allocator) so the
  // fp32 XVAE decode fits — the full-depth model sits within ~0.1 GiB of
  // the device budget otherwise. NOTE: the weights live in CausalVLMImpl's
  // UNREGISTERED model_ member, so nn::Module::to() on the wrapper would be
  // a no-op — move_model_to is the hook that actually relocates them.
  void set_on_device(bool on_device) {
    if (on_device == encoder_on_device_) {
      return;
    }
    encoder_->move_model_to(on_device ? device_ : torch::Device(torch::kCPU));
    encoder_on_device_ = on_device;
  }

 private:
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

  // Whether encoder_ currently lives on device_ (it loads on-device and is
  // toggled around the encode window; see set_on_device).
  bool encoder_on_device_ = true;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::vector<KVCache> text_encoder_empty_kv_caches_;
};

}  // namespace xllm
