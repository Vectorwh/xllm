/* Copyright 2025-2026 The xLLM Authors.

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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/dit_cache/dit_cache.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/runtime/dit_forward_params.h"
#include "models/dit/autoencoders/autoencoder_kl_wan.h"
#include "models/dit/encoders/joyimage_qwen3_vl.h"
#include "models/dit/processors/joyimage_qwen3_image_processor.h"
#include "models/dit/processors/vae_image_processor.h"
#include "models/dit/schedulers/flowmatch_euler_discrete_scheduler.h"
#include "models/dit/transformers/transformer_joyimage_edit_plus.h"
#include "models/dit/utils/dit_parallel_mixin.h"
#include "models/dit/utils/util.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "processors/multimodal_processor.h"

namespace xllm {

using JoyImageEditTextEncoder = JoyImageQwen3VLForConditionalGeneration;

class JoyImageEditPlusPipelineImpl : public torch::nn::Module,
                                     public dit::CFGParallelMixin {
 public:
  using JoyImageShapeList = std::vector<std::vector<std::array<int64_t, 3>>>;

  struct TransformerForwardContext {
    torch::Tensor rope_cos;
    torch::Tensor rope_sin;
    torch::Tensor attention_mask;
  };

  JoyImageEditPlusPipelineImpl(const DiTModelContext& context)
      : dit::CFGParallelMixin(context),
        parallel_args_(context.get_parallel_args()),
        vae_model_args_(context.get_model_args("vae")) {
    options_ = context.get_tensor_options();
    dtype_ = options_.dtype().toScalarType();
    device_ = options_.device();

    const ModelArgs& transformer_model_args =
        context.get_model_args("transformer");
    in_channels_ = transformer_model_args.in_channels();
    num_layers_ = transformer_model_args.num_layers();
    const int64_t hidden_size = transformer_model_args.hidden_size();
    const int64_t num_heads = transformer_model_args.num_attention_heads();
    CHECK_EQ(hidden_size % num_heads, 0)
        << "hidden_size must be divisible by num_attention_heads";
    head_dim_ = hidden_size / num_heads;
    rope_dim_list_ = transformer_model_args.rope_dim_list();
    theta_ = transformer_model_args.rope_theta_dit();
    auto patch = transformer_model_args.wan_patch_size();
    patch_t_ = patch[0];
    patch_h_ = patch[1];
    patch_w_ = patch[2];

    latent_channels_ = vae_model_args_.z_dim();
    vae_scale_factor_spatial_ = vae_model_args_.vae_scale_factor_spatial();
    if (vae_scale_factor_spatial_ <= 0) {
      vae_scale_factor_spatial_ = 8;
    }

    vae_ = AutoencoderKLWan(context.get_model_context("vae"));
    transformer_ = joyimage::JoyImageEditPlusTransformer3DModel(
        context.get_model_context("transformer"), parallel_args_);
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));

    if (context.has_component("text_encoder")) {
      text_encoder_model_args_ = context.get_model_args("text_encoder");
#if defined(USE_NPU)
      CHECK_EQ(KernelConfig::get_instance().npu_kernel_backend(), "TORCH")
          << "JoyImageEditPlus in-process Qwen3-VL text encoding requires "
             "--npu_kernel_backend=TORCH.";
#endif
      ProcessGroup* tp_group = parallel_args_.dit_text_encoder_tp_group_;
      CHECK(tp_group != nullptr)
          << "JoyImageEditPlus requires an injected text encoder TP group.";
      ParallelArgs vlm_parallel_args(
          tp_group->rank(), tp_group->world_size(), tp_group);
      vlm_parallel_args.tp_size(tp_group->world_size());
      vlm_parallel_args.tp_group_ = tp_group;
      text_encoder_empty_kv_caches_.resize(
          static_cast<size_t>(text_encoder_model_args_.n_layers()));
      const ModelContext& source_vlm_context =
          context.get_model_context("text_encoder");
      ModelContext vlm_context(vlm_parallel_args,
                               source_vlm_context.get_model_args(),
                               source_vlm_context.get_quant_args(),
                               source_vlm_context.get_tensor_options());
      text_encoder_ = JoyImageEditTextEncoder(vlm_context);
      CHECK(!text_encoder_.is_empty())
          << "Failed to create JoyImageEditPlus Qwen3-VL text encoder";
      mposition_generator_ =
          MPositionGeneratorFactory::get_instance().create_mposition_generator(
              "qwen3_vl");
    }

    vae_image_processor_ =
        xllm::VAEImageProcessor(context.get_model_context("vae"),
                                /*do_resize=*/true,
                                /*do_normalize=*/true,
                                /*do_binarize=*/false,
                                /*do_convert_rgb=*/false,
                                /*do_convert_grayscale=*/false,
                                latent_channels_,
                                /*scale_factor=*/vae_scale_factor_spatial_);

    register_module("vae", vae_);
    register_module("scheduler", scheduler_);
    register_module("transformer", transformer_);
    if (!text_encoder_.is_empty()) {
      register_module("text_encoder", text_encoder_);
    }
    register_module("vae_image_processor", vae_image_processor_);

    latents_mean_ = vae_model_args_.latents_mean();
    latents_std_ = vae_model_args_.latents_std();
  }

  torch::Tensor latents_mean_tensor(const torch::Tensor& ref) const {
    return torch::tensor(latents_mean_, torch::kFloat32)
        .view({1, latent_channels_, 1, 1, 1})
        .to(ref.device(), ref.dtype());
  }
  torch::Tensor latents_std_tensor(const torch::Tensor& ref) const {
    return torch::tensor(latents_std_, torch::kFloat32)
        .view({1, latent_channels_, 1, 1, 1})
        .to(ref.device(), ref.dtype());
  }

  // Patchify a [C, T, H, W] latent into [num_patches, C, pt, ph, pw] and
  // return {patches, (lt, lh, lw)}.
  std::pair<torch::Tensor, std::array<int64_t, 3>> patchify(
      const torch::Tensor& item) {
    int64_t c = item.size(0), t = item.size(1), h = item.size(2),
            w = item.size(3);
    int64_t lt = t / patch_t_, lh = h / patch_h_, lw = w / patch_w_;
    auto p = item.reshape({c, lt, patch_t_, lh, patch_h_, lw, patch_w_});
    p = p.permute({1, 3, 5, 0, 2, 4, 6})
            .reshape({-1, c, patch_t_, patch_h_, patch_w_});
    return {p, {lt, lh, lw}};
  }

  // Build the 6D padded latent tensor: target noise + reference latents.
  // Returns {padded_latents[B,N,C,pt,ph,pw], target_mask[B,N],
  //          shape_list (per-sample list of (t,h,w))}.
  std::tuple<torch::Tensor, torch::Tensor, JoyImageShapeList> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      const std::vector<std::vector<torch::Tensor>>& reference_images,
      const torch::Tensor& provided_latents) {
    std::vector<torch::Tensor> all_patches;
    std::vector<torch::Tensor> all_target_masks;
    std::vector<std::vector<std::array<int64_t, 3>>> all_shapes;
    int64_t max_patches = 0;
    int64_t h_target = height / vae_scale_factor_spatial_;
    int64_t w_target = width / vae_scale_factor_spatial_;

    for (int64_t b = 0; b < batch_size; ++b) {
      std::vector<torch::Tensor> items;
      // Target noise: [C, 1, h', w'].
      torch::Tensor noise;
      if (provided_latents.defined()) {
        noise = provided_latents[b].to(device_, dtype_);
      } else {
        noise = xllm::dit::randn_tensor(
            {num_channels_latents, 1, h_target, w_target}, seed + b, options_);
      }
      items.push_back(noise);

      // References: VAE-encode each, normalize, squeeze to [C, 1, h', w'].
      if (b < static_cast<int64_t>(reference_images.size())) {
        for (const auto& ref_img : reference_images[b]) {
          auto ref = ref_img.to(device_, dtype_);
          if (ref.dim() == 4) ref = ref.unsqueeze(2);  // [B,C,H,W]->[B,C,1,H,W]
          auto lat = vae_->encode(ref.to(dtype_)).latent_dist.mode();
          lat = lat.to(dtype_);
          lat = (lat - latents_mean_tensor(lat)) / latents_std_tensor(lat);
          items.push_back(lat.squeeze(0));  // [C, 1, h', w']
        }
      }

      std::vector<torch::Tensor> sample_patches;
      std::vector<torch::Tensor> sample_masks;
      std::vector<std::array<int64_t, 3>> sample_shapes;
      for (size_t j = 0; j < items.size(); ++j) {
        auto pr = patchify(items[j]);
        sample_shapes.push_back(pr.second);
        sample_patches.push_back(pr.first);
        auto n = pr.first.size(0);
        sample_masks.push_back(torch::full(
            {n},
            /*value=*/(j == 0),
            torch::TensorOptions().device(device_).dtype(torch::kBool)));
      }
      auto combined = torch::cat(sample_patches, 0);
      auto combined_mask = torch::cat(sample_masks, 0);
      all_patches.push_back(combined);
      all_target_masks.push_back(combined_mask);
      all_shapes.push_back(sample_shapes);
      max_patches = std::max(max_patches, combined.size(0));
    }

    auto padded = torch::zeros({batch_size,
                                max_patches,
                                num_channels_latents,
                                patch_t_,
                                patch_h_,
                                patch_w_},
                               options_);
    auto target_mask = torch::zeros(
        {batch_size, max_patches},
        torch::TensorOptions().device(device_).dtype(torch::kBool));
    for (int64_t b = 0; b < batch_size; ++b) {
      int64_t n = all_patches[b].size(0);
      padded.index_put_({b, torch::indexing::Slice(0, n)}, all_patches[b]);
      target_mask.index_put_({b, torch::indexing::Slice(0, n)},
                             all_target_masks[b]);
    }
    return std::make_tuple(padded, target_mask, all_shapes);
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    const auto& gp = input.generation_params;
    int64_t num_inference_steps = gp.num_inference_steps;
    double guidance_scale = gp.guidance_scale;
    if (gp.true_cfg_scale > 1.0) {
      guidance_scale = gp.true_cfg_scale;
    }
    int64_t seed = gp.seed >= 0 ? gp.seed : 42;

    // Collect reference images (one sample per batch entry).
    std::vector<torch::Tensor> raw_images = input.media_sources.get();
    if (raw_images.empty()) {
      LOG(FATAL) << "JoyImageEditPlus requires reference images";
    }

    int64_t batch_size = input.batch_size;
    if (batch_size <= 0) {
      if (input.tensor_sources.contains("prompt_embed")) {
        batch_size = input.tensor_sources.get("prompt_embed")
                         .value_or(torch::Tensor())
                         .size(0);
      } else if (!input.prompts.empty()) {
        batch_size = static_cast<int64_t>(input.prompts.size());
      } else {
        batch_size = raw_images[0].dim() == 4 ? raw_images[0].size(0) : 1;
      }
    }

    // Determine output resolution from the last reference image if unset.
    int64_t height = gp.height;
    int64_t width = gp.width;
    if (height <= 0 || width <= 0) {
      const auto& last = raw_images.back();
      int64_t ih = last.size(last.dim() - 2);
      int64_t iw = last.size(last.dim() - 1);
      auto hw = JoyImageQwen3ImageProcessor::select_bucket(ih, iw);
      height = hw.first;
      width = hw.second;
    }
    height = (height / vae_scale_factor_spatial_) * vae_scale_factor_spatial_;
    width = (width / vae_scale_factor_spatial_) * vae_scale_factor_spatial_;

    const std::vector<std::vector<torch::Tensor>> processed_images =
        preprocess_images(raw_images, batch_size);

    std::vector<std::vector<torch::Tensor>> vae_refs(batch_size);
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      vae_refs[batch_index].reserve(processed_images[batch_index].size());
      for (const torch::Tensor& image : processed_images[batch_index]) {
        vae_refs[batch_index].emplace_back(normalize_reference_image(image));
      }
    }
    const bool do_true_cfg = guidance_scale > 1.0;
    const bool reuse_image_embeddings =
        do_true_cfg && !input.tensor_sources.contains("prompt_embed") &&
        !input.tensor_sources.contains("negative_prompt_embed");
    std::vector<MMDict> image_embeddings_cache(
        reuse_image_embeddings ? static_cast<size_t>(batch_size) : 0);
    auto* image_embeddings_cache_ptr =
        reuse_image_embeddings ? &image_embeddings_cache : nullptr;
    torch::Tensor prompt_embeds;
    torch::Tensor prompt_embeds_mask;
    const bool encode_text = !input.tensor_sources.contains("prompt_embed") ||
                             (do_true_cfg && !input.tensor_sources.contains(
                                                 "negative_prompt_embed"));
    if (encode_text) {
      CHECK(!text_encoder_.is_empty()) << "Qwen3-VL text encoder is not loaded";
    }
    if (input.tensor_sources.contains("prompt_embed")) {
      prompt_embeds = input.tensor_sources.get("prompt_embed")
                          .value_or(torch::Tensor())
                          .to(options_.device(), dtype_);
      prompt_embeds_mask = torch::ones(
          {prompt_embeds.size(0), prompt_embeds.size(1)},
          torch::TensorOptions().device(device_).dtype(torch::kLong));
    } else {
      CHECK(!input.prompts.empty())
          << "JoyImageEditPlus requires `prompts` or `prompt_embeds`";
      std::tie(prompt_embeds, prompt_embeds_mask) =
          encode_prompts(input.prompts,
                         processed_images,
                         batch_size,
                         gp.max_sequence_length,
                         image_embeddings_cache_ptr);
    }

    torch::Tensor neg_embeds, neg_embeds_mask;
    if (do_true_cfg) {
      if (input.tensor_sources.contains("negative_prompt_embed")) {
        neg_embeds = input.tensor_sources.get("negative_prompt_embed")
                         .value_or(torch::Tensor())
                         .to(options_.device(), dtype_);
        neg_embeds_mask = torch::ones(
            {neg_embeds.size(0), neg_embeds.size(1)},
            torch::TensorOptions().device(device_).dtype(torch::kLong));
      } else {
        std::vector<std::string> negative_prompts = input.negative_prompts;
        if (negative_prompts.empty()) {
          negative_prompts.assign(static_cast<size_t>(batch_size), " ");
        } else {
          std::replace(negative_prompts.begin(),
                       negative_prompts.end(),
                       std::string(),
                       std::string(" "));
        }
        std::tie(neg_embeds, neg_embeds_mask) =
            encode_prompts(negative_prompts,
                           processed_images,
                           batch_size,
                           gp.max_sequence_length,
                           image_embeddings_cache_ptr);
      }
      // Pad/concat [negative, positive] to equal sequence length.
      int64_t max_l = std::max(prompt_embeds.size(1), neg_embeds.size(1));
      prompt_embeds = pad_seq(prompt_embeds, max_l);
      neg_embeds = pad_seq(neg_embeds, max_l);
      prompt_embeds_mask = pad_seq(prompt_embeds_mask, max_l);
      neg_embeds_mask = pad_seq(neg_embeds_mask, max_l);
    }
    image_embeddings_cache.clear();
    // Latents.
    int64_t num_channels_latents = in_channels_;
    auto lp = prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        seed,
        vae_refs,
        input.tensor_sources.get("latent").value_or(torch::Tensor()));
    auto latents = std::get<0>(lp);
    auto target_mask = std::get<1>(lp);
    auto shape_list = std::get<2>(lp);

    auto clean_backup = latents.clone();
    const std::array<int64_t, 3>& target_shape = shape_list.front().front();
    const int64_t target_patch_count =
        target_shape[0] * target_shape[1] * target_shape[2];

    // Timesteps (static shift; no dynamic shifting for Joy).
    scheduler_->set_timesteps(num_inference_steps, device_);
    scheduler_->set_begin_index(0);
    auto timesteps = scheduler_->timesteps();
    DiTCache::get_instance().set_context({/*infer_steps=*/num_inference_steps,
                                          /*num_blocks=*/num_layers_});

    TransformerForwardContext prompt_transformer_context =
        prepare_transformer_context(batch_size,
                                    latents.size(1),
                                    latents.device(),
                                    prompt_embeds_mask,
                                    shape_list);
    TransformerForwardContext negative_transformer_context;
    if (do_true_cfg) {
      negative_transformer_context =
          prepare_transformer_context(batch_size,
                                      latents.size(1),
                                      latents.device(),
                                      neg_embeds_mask,
                                      shape_list);
    }

    for (int64_t i = 0; i < timesteps.size(0); ++i) {
      auto t = timesteps[i];

      torch::Tensor noise_pred;
      if (do_true_cfg) {
        torch::Tensor t_expand = t.repeat({batch_size});
        auto [cond, uncond] = exec_with_cfg([&](bool is_positive) {
          const torch::Tensor& encoder_hidden_states =
              is_positive ? prompt_embeds : neg_embeds;
          const TransformerForwardContext& transformer_context =
              is_positive ? prompt_transformer_context
                          : negative_transformer_context;
          return transformer_forward(latents,
                                     t_expand,
                                     encoder_hidden_states,
                                     transformer_context,
                                     /*use_cfg=*/!is_positive,
                                     /*step_index=*/i + 1,
                                     do_true_cfg);
        });
        auto comb = uncond + guidance_scale * (cond - uncond);
        // Norm-rescale (diffusers): comb * (||cond|| / ||comb||) over channel
        // dim (2) of the 6D [B, N, C, pt, ph, pw] prediction.
        auto cond_norm =
            torch::norm(cond, 2, std::vector<int64_t>{2}, /*keepdim=*/true);
        auto noise_norm =
            torch::norm(comb, 2, std::vector<int64_t>{2}, /*keepdim=*/true);
        noise_pred = comb * (cond_norm / noise_norm.clamp_min(1e-6));
      } else {
        torch::Tensor t_expand = t.repeat({batch_size});
        noise_pred = transformer_forward(latents,
                                         t_expand,
                                         prompt_embeds,
                                         prompt_transformer_context,
                                         /*use_cfg=*/false,
                                         /*step_index=*/i + 1,
                                         do_true_cfg);
      }

      latents = scheduler_->step(noise_pred, t, latents).to(latents.dtype());
      latents.slice(/*dim=*/1, target_patch_count, latents.size(1))
          .copy_(clean_backup.slice(
              /*dim=*/1, target_patch_count, clean_backup.size(1)));
    }

    // Decode target patches per sample.
    std::vector<torch::Tensor> images;
    for (int64_t b = 0; b < batch_size; ++b) {
      auto thw = shape_list[b][0];
      int64_t lt = thw[0], lh = thw[1], lw = thw[2];
      int64_t target_len = lt * lh * lw;
      auto patches = latents[b].slice(0, 0, target_len);  // [len, C, pt,ph,pw]
      int64_t c = patches.size(1);
      auto vid = patches.reshape({lt, lh, lw, c, patch_t_, patch_h_, patch_w_});
      vid = vid.permute({3, 0, 4, 1, 5, 2, 6})
                .reshape({1, c, lt * patch_t_, lh * patch_h_, lw * patch_w_});
      vid = vid * latents_std_tensor(vid) + latents_mean_tensor(vid);
      auto img = vae_->decode(vid.to(dtype_)).sample;       // [1,C,1,H,W]
      img = img.to(torch::kFloat32).squeeze(0).squeeze(1);  // [C,H,W]
      images.push_back(img.unsqueeze(0));
    }
    auto image = torch::cat(images, 0);
    image = vae_image_processor_->postprocess(image);

    DiTForwardOutput out;
    out.tensors = torch::chunk(image, batch_size, 0);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "JoyImageEditPlusPipeline loading from "
              << loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    std::unique_ptr<DiTFolderLoader> text_encoder_loader;
    if (!text_encoder_.is_empty()) {
      text_encoder_loader = loader->take_component_loader("text_encoder");
      std::unique_ptr<Tokenizer> tokenizer = text_encoder_loader->tokenizer();
      CHECK(tokenizer != nullptr)
          << "Failed to load JoyImageEditPlus Qwen3-VL tokenizer";
      tokenizer_ = std::shared_ptr<Tokenizer>(std::move(tokenizer));
      multimodal_processor_ =
          create_multimodal_processor(text_encoder_model_args_,
                                      tokenizer_,
                                      /*max_cache_items=*/0,
                                      text_encoder_loader->tokenizer_args());
    }

    vae_->load_model(std::move(vae_loader));
    vae_->to(options_.device(), dtype_);

    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(options_.device(), dtype_);
    transformer_->keep_fp32_modules();

    if (text_encoder_loader != nullptr) {
      text_encoder_->load_model(std::move(text_encoder_loader));
    }

    LOG(INFO) << "JoyImageEditPlusPipeline loaded.";
  }

 private:
  static constexpr int64_t kPromptEmbeddingStartIndex = 34;

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

  std::vector<std::vector<torch::Tensor>> preprocess_images(
      const std::vector<torch::Tensor>& raw_images,
      int64_t batch_size) const {
    std::vector<std::vector<torch::Tensor>> processed_images(batch_size);
    for (std::vector<torch::Tensor>& images : processed_images) {
      images.reserve(raw_images.size());
    }
    for (const torch::Tensor& images : raw_images) {
      CHECK(images.dim() == 3 || images.dim() == 4)
          << "JoyImageEditPlus image input must be [C,H,W] or [B,C,H,W]";
      if (images.dim() == 4) {
        CHECK_EQ(images.size(0), batch_size)
            << "JoyImageEditPlus image batch must match prompt batch";
      } else {
        CHECK_EQ(batch_size, 1)
            << "Unbatched JoyImageEditPlus images require batch_size=1";
      }
      for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        const torch::Tensor image =
            images.dim() == 4 ? images[batch_index] : images;
        processed_images[batch_index].emplace_back(
            JoyImageQwen3ImageProcessor::preprocess(image));
      }
    }
    return processed_images;
  }

  ModelInputParams build_text_encoder_input(const torch::Tensor& tokens,
                                            const MMData& mm_data,
                                            MMDict* image_embeddings_cache) {
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
    MMDict uncached_embeddings;
    MMDict& multimodal_embeddings = image_embeddings_cache != nullptr
                                        ? *image_embeddings_cache
                                        : uncached_embeddings;
    if (multimodal_embeddings.empty()) {
      multimodal_embeddings =
          text_encoder_->get_multimodal_embeddings(multimodal_params);
    }
    EncoderOutputScatterVisitor output_scatter(multimodal_embeddings);
    CHECK(mm_batch.foreach (output_scatter));
    CHECK(output_scatter.finish())
        << "JoyImageEditPlus failed to scatter Qwen3-VL embeddings";

    EncoderEmbeddingGatherVisitor embedding_gather(
        /*num_computed_tokens=*/{0},
        /*num_scheduled_tokens=*/{sequence_length});
    CHECK(mm_batch.foreach (embedding_gather));
    CHECK(embedding_gather.finish(mm_batch))
        << "JoyImageEditPlus failed to gather Qwen3-VL embeddings";
    params.multimodal.mm_data = std::move(mm_batch);
    params.embedding.input_embedding =
        text_encoder_->get_input_embeddings(tokens, params);
    return params;
  }

  std::pair<torch::Tensor, torch::Tensor> encode_single_prompt(
      const std::string& prompt,
      const std::vector<torch::Tensor>& images,
      MMDict* image_embeddings_cache) {
    CHECK(!text_encoder_.is_empty()) << "Qwen3-VL text encoder is not loaded";
    CHECK(tokenizer_ != nullptr) << "Qwen3-VL tokenizer is not loaded";
    CHECK(multimodal_processor_ != nullptr)
        << "Qwen3-VL multimodal processor is not loaded";
    CHECK(mposition_generator_ != nullptr)
        << "Qwen3-VL position generator is not loaded";

    std::vector<MMInputItem> input_items;
    input_items.reserve(images.size());
    for (const torch::Tensor& image : images) {
      MMInputItem item;
      item.type = MMType::IMAGE;
      item.decode_image = image;
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
    ModelInputParams input_params =
        build_text_encoder_input(tokens, mm_data, image_embeddings_cache);
    ModelOutput model_output = text_encoder_->forward(
        tokens, positions, text_encoder_empty_kv_caches_, input_params);
    CHECK(model_output.residual.defined())
        << "JoyImageEditPlus requires Qwen3-VL pre-norm hidden states from "
           "the TORCH backend.";
    CHECK_GT(model_output.residual.size(0), kPromptEmbeddingStartIndex)
        << "JoyImageEditPlus Qwen3-VL prompt is shorter than the embedding "
           "prefix";

    torch::Tensor prompt_embeddings =
        model_output.residual
            .slice(
                /*dim=*/0, kPromptEmbeddingStartIndex)
            .unsqueeze(0)
            .to(options_);
    torch::Tensor prompt_mask =
        torch::ones({1, prompt_embeddings.size(1)},
                    torch::TensorOptions().device(device_).dtype(torch::kLong));
    return {prompt_embeddings, prompt_mask};
  }

  std::pair<torch::Tensor, torch::Tensor> encode_prompts(
      const std::vector<std::string>& prompts,
      const std::vector<std::vector<torch::Tensor>>& processed_images,
      int64_t batch_size,
      int64_t max_sequence_length,
      std::vector<MMDict>* image_embeddings_cache) {
    CHECK_EQ(static_cast<int64_t>(prompts.size()), batch_size)
        << "JoyImageEditPlus prompt batch size mismatch";
    CHECK_EQ(static_cast<int64_t>(processed_images.size()), batch_size)
        << "JoyImageEditPlus image batch size mismatch";
    if (image_embeddings_cache != nullptr) {
      CHECK_EQ(static_cast<int64_t>(image_embeddings_cache->size()), batch_size)
          << "JoyImageEditPlus image embedding cache batch size mismatch";
    }
    std::vector<torch::Tensor> embeddings;
    std::vector<torch::Tensor> masks;
    embeddings.reserve(static_cast<size_t>(batch_size));
    masks.reserve(static_cast<size_t>(batch_size));
    int64_t batch_sequence_length = 0;
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      auto [embedding, mask] =
          encode_single_prompt(prompts[static_cast<size_t>(batch_index)],
                               processed_images[batch_index],
                               image_embeddings_cache != nullptr
                                   ? &(*image_embeddings_cache)[batch_index]
                                   : nullptr);
      const int64_t sequence_length = embedding.size(1);
      if (max_sequence_length > 0 && sequence_length > max_sequence_length) {
        const int64_t start = sequence_length - max_sequence_length;
        embedding = embedding.slice(1, start, sequence_length);
        mask = mask.slice(1, start, sequence_length);
      }
      batch_sequence_length =
          std::max(batch_sequence_length, embedding.size(1));
      embeddings.emplace_back(std::move(embedding));
      masks.emplace_back(std::move(mask));
    }
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      embeddings[static_cast<size_t>(batch_index)] = pad_seq(
          embeddings[static_cast<size_t>(batch_index)], batch_sequence_length);
      masks[static_cast<size_t>(batch_index)] = pad_seq(
          masks[static_cast<size_t>(batch_index)], batch_sequence_length);
    }
    return {torch::cat(embeddings, /*dim=*/0), torch::cat(masks, /*dim=*/0)};
  }

  torch::Tensor transformer_forward(
      const torch::Tensor& hidden_states,
      const torch::Tensor& timestep,
      const torch::Tensor& encoder_hidden_states,
      const TransformerForwardContext& forward_context,
      bool use_cfg,
      int64_t step_index,
      bool do_true_cfg) {
    return transformer_->forward(hidden_states,
                                 timestep,
                                 encoder_hidden_states,
                                 forward_context.rope_cos,
                                 forward_context.rope_sin,
                                 forward_context.attention_mask,
                                 use_cfg,
                                 step_index,
                                 do_true_cfg);
  }

  TransformerForwardContext prepare_transformer_context(
      int64_t batch_size,
      int64_t image_sequence_length,
      const torch::Device& device,
      const torch::Tensor& encoder_hidden_states_mask,
      const JoyImageShapeList& shape_list) {
    CHECK_EQ(static_cast<int64_t>(shape_list.size()), batch_size)
        << "shape_list batch size must match transformer batch size";

    std::vector<torch::Tensor> cos_list;
    std::vector<torch::Tensor> sin_list;
    cos_list.reserve(batch_size);
    sin_list.reserve(batch_size);
    for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      std::vector<torch::Tensor> cos_parts;
      std::vector<torch::Tensor> sin_parts;
      int64_t temporal_offset = 0;
      for (const auto& shape : shape_list[batch_index]) {
        std::array<int64_t, 3> start = {temporal_offset, 0, 0};
        std::array<int64_t, 3> stop = {
            temporal_offset + shape[0], shape[1], shape[2]};
        auto rope = rope_for_range(start, stop, device);
        cos_parts.emplace_back(rope.first);
        sin_parts.emplace_back(rope.second);
        temporal_offset += shape[0];
      }

      torch::Tensor cos = torch::cat(cos_parts, /*dim=*/0);
      torch::Tensor sin = torch::cat(sin_parts, /*dim=*/0);
      const int64_t actual_sequence_length = cos.size(0);
      if (actual_sequence_length < image_sequence_length) {
        cos = torch::constant_pad_nd(
            cos,
            {0, 0, 0, image_sequence_length - actual_sequence_length},
            /*value=*/1.0);
        sin = torch::constant_pad_nd(
            sin,
            {0, 0, 0, image_sequence_length - actual_sequence_length},
            /*value=*/0.0);
      }
      cos_list.emplace_back(cos);
      sin_list.emplace_back(sin);
    }

    torch::Tensor rope_cos = torch::stack(cos_list, /*dim=*/0);
    torch::Tensor rope_sin = torch::stack(sin_list, /*dim=*/0);
    torch::Tensor attention_mask;

    if (encoder_hidden_states_mask.defined()) {
      torch::Tensor image_mask = torch::zeros(
          {batch_size, image_sequence_length},
          torch::TensorOptions().device(device).dtype(torch::kBool));
      for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        int64_t actual_sequence_length = 0;
        for (const auto& shape : shape_list[batch_index]) {
          actual_sequence_length += shape[0] * shape[1] * shape[2];
        }
        image_mask.index_put_(
            {batch_index, torch::indexing::Slice(0, actual_sequence_length)},
            true);
      }
      torch::Tensor full_mask =
          torch::cat({image_mask, encoder_hidden_states_mask.to(torch::kBool)},
                     /*dim=*/1);
#if defined(USE_NPU)
      const int64_t sequence_length = full_mask.size(1);
      attention_mask = full_mask.logical_not()
                           .unsqueeze(1)
                           .unsqueeze(1)
                           .expand({batch_size,
                                    /*num_heads=*/1,
                                    sequence_length,
                                    sequence_length})
                           .contiguous();
#else
      attention_mask = full_mask.unsqueeze(1).unsqueeze(1);
#endif
    }

    return {rope_cos, rope_sin, attention_mask};
  }

  std::pair<torch::Tensor, torch::Tensor> rope_for_range(
      const std::array<int64_t, 3>& start,
      const std::array<int64_t, 3>& stop,
      const torch::Device& device) {
    std::vector<int64_t> dimensions = rope_dim_list_;
    if (dimensions.empty()) {
      const int64_t dimension = head_dim_ / 3;
      dimensions = {dimension, dimension, dimension};
    }

    torch::TensorOptions float_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    std::vector<torch::Tensor> grids;
    grids.reserve(3);
    for (int64_t dimension_index = 0; dimension_index < 3; ++dimension_index) {
      grids.emplace_back(torch::arange(
          start[dimension_index], stop[dimension_index], float_options));
    }
    auto mesh = torch::meshgrid({grids[0], grids[1], grids[2]}, "ij");

    std::vector<torch::Tensor> cos_parts;
    std::vector<torch::Tensor> sin_parts;
    cos_parts.reserve(3);
    sin_parts.reserve(3);
    for (int64_t dimension_index = 0; dimension_index < 3; ++dimension_index) {
      torch::Tensor position = mesh[dimension_index].reshape({-1});
      const int64_t dimension = dimensions[dimension_index];
      torch::Tensor index =
          torch::arange(0, dimension, 2, float_options)
              .slice(/*dim=*/0, /*start=*/0, /*end=*/dimension / 2);
      torch::Tensor frequencies =
          1.0 / torch::pow(static_cast<double>(theta_), index / dimension);
      torch::Tensor angles = torch::outer(position, frequencies);
      cos_parts.emplace_back(angles.cos().repeat_interleave(2, /*dim=*/1));
      sin_parts.emplace_back(angles.sin().repeat_interleave(2, /*dim=*/1));
    }
    return {torch::cat(cos_parts, /*dim=*/1).to(device),
            torch::cat(sin_parts, /*dim=*/1).to(device)};
  }

  torch::Tensor normalize_reference_image(const torch::Tensor& image) const {
    torch::Tensor processed = image.to(torch::kFloat32) / 255.0f;
    if (processed.dim() == 3) {
      processed = processed.unsqueeze(0);
    }
    processed = 2.0f * processed - 1.0f;
    return processed.to(device_, dtype_);
  }

  torch::Tensor pad_seq(const torch::Tensor& x, int64_t target_len) {
    int64_t cur = x.size(1);
    if (cur >= target_len) return x.slice(1, cur - target_len, cur);
    int64_t pad = target_len - cur;
    std::vector<int64_t> shape = x.sizes().vec();
    shape[1] = pad;
    auto zeros = torch::zeros(shape, x.options());
    return torch::cat({x, zeros}, 1);
  }

  const ParallelArgs parallel_args_;
  const ModelArgs& vae_model_args_;
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;

  AutoencoderKLWan vae_{nullptr};
  joyimage::JoyImageEditPlusTransformer3DModel transformer_{nullptr};
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  xllm::VAEImageProcessor vae_image_processor_{nullptr};
  JoyImageEditTextEncoder text_encoder_{nullptr};
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<MultimodalProcessorBase> multimodal_processor_;
  std::unique_ptr<MPositionGenerator> mposition_generator_;
  ModelArgs text_encoder_model_args_;
  std::vector<KVCache> text_encoder_empty_kv_caches_;

  int64_t in_channels_;
  int64_t num_layers_;
  int64_t head_dim_;
  int64_t theta_;
  int64_t patch_t_, patch_h_, patch_w_;
  int64_t latent_channels_;
  int64_t vae_scale_factor_spatial_;
  std::vector<int64_t> rope_dim_list_;
  std::vector<double> latents_mean_;
  std::vector<double> latents_std_;
};

TORCH_MODULE(JoyImageEditPlusPipeline);

REGISTER_DIT_MODEL(JoyImageEditPlusPipeline, JoyImageEditPlusPipeline);
}  // namespace xllm
