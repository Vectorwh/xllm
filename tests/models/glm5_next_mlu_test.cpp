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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model/causal_vlm.h"
#include "core/platform/device.h"
#include "core/platform/layer_synchronizer.h"
#include "core/platform/platform.h"
#include "models/llm/mlu/glm5_next.h"
#include "models/vlm/mlu/glm5_next.h"

namespace xllm::mlu::model {
namespace {

TEST(Glm5NextMluDeathTest, RejectsNonSpanVerificationBeforeUsingInputs) {
  const torch::Device device(torch::kCPU);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &group);
  parallel_args.tp_group_ = &group;
  ModelArgs args;
  args.model_type("glm5_next");
  args.n_layers(0);
  args.hidden_size(4);
  args.vocab_size(8);
  args.hc_mult(1);
  args.linear_num_key_heads(1);
  const ModelContext context(
      parallel_args,
      args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  Glm5NextModelImpl model(context);
  std::vector<KVCache> caches;

  // Undefined inputs must never reach embedding, metadata building, or layers.
  const torch::Tensor tokens;
  const torch::Tensor positions;
  ModelInputParams params;
  params.is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");

  params.is_spec_verify = false;
  params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
  params.attn_metadata->is_spec_verify = true;
  EXPECT_DEATH(model.forward(tokens, positions, caches, params),
               "requires chunked-prefill");
}

TEST(Glm5NextMluMetadataTest,
     CancelledForwardPreservesCallerAcrossPrefixReuse) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const torch::Device device(torch::kCPU);
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
  parallel.tp_group_ = &group;
  ModelArgs args;
  args.model_type("glm5_next")
      .n_layers(1)
      .layer_types({"linear_attention"})
      .mlp_layer_types({"dense"})
      .hidden_size(128)
      .intermediate_size(256)
      .hidden_act("silu")
      .vocab_size(8)
      .hc_mult(4)
      .hc_sinkhorn_iters(4)
      .hc_eps(1e-6)
      .rms_norm_eps(1e-6)
      .linear_num_key_heads(1)
      .linear_num_value_heads(1)
      .linear_key_head_dim(128)
      .linear_value_head_dim(128)
      .linear_conv_kernel_dim(4)
      .linear_lower_bound(-5.0f)
      .num_speculative_tokens(1);
  Glm5NextModelImpl model(ModelContext(parallel, args, QuantArgs(), options));
  std::vector<KVCache> caches;
  caches.emplace_back(
      LinearAttentionKVCacheTensors{torch::zeros({4, 4, 384}, options),
                                    torch::zeros({8, 1, 128, 128}, options)});
  // This is the real failed-copy lifecycle: metadata is prepared before the
  // layer wait aborts, so no decoder weights or device kernels are consumed.
  auto synchronizer = create_layer_synchronizer(/*num_layers=*/1);
  ASSERT_NE(synchronizer, nullptr);
  synchronizer->abort();
  ASSERT_FALSE(synchronizer->synchronize_layer(/*layer_index=*/0));
  const auto make_input = [&](BatchForwardType type) {
    ModelInputParams params;
    params.is_spec_verify = type.is_chunked_prefill();
    const int32_t width = params.is_spec_verify ? 2 : 1;
    params.meta.batch_forward_type = type;
    params.meta.num_sequences = 2;
    params.meta.actual_num_sequences = 2;
    params.meta.q_max_seq_len = width;
    params.meta.kv_max_seq_len = 4 + width;
    params.attention.host.q_seq_lens = {0, width, 2 * width};
    params.attention.host.q_cu_seq_lens = params.attention.host.q_seq_lens;
    params.attention.host.kv_seq_lens = {0, 4 + width, 2 * (4 + width)};
    params.attention.host.kv_cache_tokens_nums = {4, 4};
    params.attention.device.q_seq_lens =
        torch::tensor({0, width, 2 * width}, torch::kInt32);
    params.attention.device.q_cu_seq_lens = params.attention.device.q_seq_lens;
    params.attention.device.block_tables =
        torch::tensor({{1}, {2}}, torch::kInt32);
    params.attention.device.kv_seq_lens =
        torch::tensor({0, 4 + width, 2 * (4 + width)}, torch::kInt32);
    params.attention.device.new_cache_slots =
        torch::arange(2 * width, torch::kInt32);
    params.embedding.input_embedding = torch::ones({2 * width, 128}, options);
    params.embedding.linear_state_ids = {2, 3};
    params.embedding.linear_state_indices =
        torch::tensor({2, 3}, torch::kInt32);
    params.embedding.linear_state_read_ids = {1, 1};
    params.linear_state_validity_mask = {1, 1};
    params.parallel.dp_global_token_nums = {2 * width};
    params.parallel.dp_is_decode = {type.is_decode() ? 1 : 0};
    params.parallel.layers_per_event = 1;
    params.parallel.layer_wise_load_synchronizer = synchronizer;
    if (params.is_spec_verify) {
      params.num_accepted_tokens = torch::ones({2}, torch::kInt32);
    }
    return params;
  };
  const auto run_forward = [&](const ModelInputParams& params) {
    const int32_t width = params.meta.q_max_seq_len;
    const torch::Tensor tokens = torch::ones({2 * width}, torch::kInt32);
    const torch::Tensor positions =
        torch::arange(4, 4 + width, torch::kInt32).repeat({2});
    return model.forward(tokens, positions, caches, params);
  };
  auto cold = make_input(BatchForwardType::PREFILL);
  EXPECT_FALSE(run_forward(cold).hidden_states.defined());
  EXPECT_EQ(cold.attn_metadata, nullptr);
  cold.embedding.linear_state_read_indices =
      torch::tensor({1, 1}, torch::kInt32);
  EXPECT_FALSE(run_forward(cold).hidden_states.defined());
  EXPECT_EQ(cold.attn_metadata, nullptr);
  cold.embedding.linear_state_read_indices = torch::tensor({1}, torch::kInt32);
  EXPECT_DEATH(run_forward(cold), "source index count");

  const auto attach_cached_metadata = [](ModelInputParams& params) {
    params.attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(params, /*enable_mla=*/false));
    // A caller may reuse graph metadata that still owns a previous prefix's
    // read slots. All slots here are in the allocated recurrent pool.
    params.attn_metadata->linear_state_read_indices =
        torch::tensor({3, 3}, torch::kInt32);
  };
  const auto expect_caller_preserved = [&](const ModelInputParams& params) {
    ASSERT_NE(params.attn_metadata, nullptr);
    const auto caller = params.attn_metadata;
    const torch::Tensor sources = caller->linear_state_read_indices;
    const torch::Tensor source_values = sources.clone();
    const torch::Tensor destination = caller->linear_state_indices;
    const torch::Tensor destination_values = destination.clone();
    const torch::Tensor query_lengths = caller->q_cu_seq_lens;
    const torch::Tensor query_values = query_lengths.clone();
    const torch::Tensor embeddings = params.embedding.input_embedding.clone();
    EXPECT_FALSE(run_forward(params).hidden_states.defined());
    EXPECT_EQ(params.attn_metadata, caller);
    ASSERT_TRUE(caller->linear_state_read_indices.defined());
    EXPECT_EQ(caller->linear_state_read_indices.data_ptr(), sources.data_ptr());
    EXPECT_TRUE(torch::equal(caller->linear_state_read_indices, source_values));
    EXPECT_EQ(caller->linear_state_indices.data_ptr(), destination.data_ptr());
    EXPECT_TRUE(torch::equal(caller->linear_state_indices, destination_values));
    EXPECT_EQ(caller->q_cu_seq_lens.data_ptr(), query_lengths.data_ptr());
    EXPECT_TRUE(torch::equal(caller->q_cu_seq_lens, query_values));
    EXPECT_TRUE(torch::equal(params.embedding.input_embedding, embeddings));
  };
  auto cached = make_input(BatchForwardType::PREFILL);
  attach_cached_metadata(cached);
  expect_caller_preserved(cached);
  cached.embedding.linear_state_read_indices =
      torch::tensor({1, 1}, torch::kInt32);
  expect_caller_preserved(cached);
  cached.embedding.linear_state_read_ids = {2, 2};
  cached.embedding.linear_state_read_indices =
      torch::tensor({2, 2}, torch::kInt32);
  expect_caller_preserved(cached);
  // Validation must read this call's source buffer, not silently reuse the
  // valid cached buffer from the previous prefix.
  cached.embedding.linear_state_read_indices =
      torch::tensor({2}, torch::kInt32);
  EXPECT_DEATH(run_forward(cached), "source index count");
  cached.embedding.linear_state_read_indices =
      torch::tensor({2, 2}, torch::kInt32).view({1, 2});
  EXPECT_DEATH(run_forward(cached), "source indices must be 1D");
  cached.embedding.linear_state_read_indices =
      torch::tensor({2, 2}, torch::kFloat32);
  EXPECT_DEATH(run_forward(cached), "source indices must be integers");

  auto decode = make_input(BatchForwardType::DECODE);
  attach_cached_metadata(decode);
  expect_caller_preserved(decode);
  auto mixed = make_input(BatchForwardType::MIXED);
  attach_cached_metadata(mixed);
  expect_caller_preserved(mixed);
  auto verify = make_input(BatchForwardType::CHUNKED_PREFILL);
  attach_cached_metadata(verify);
  expect_caller_preserved(verify);
}

ModelArgs small_vlm_args() {
  ModelArgs args;
  args.model_type("glm5_next")
      .n_layers(0)
      .hidden_size(4)
      .vocab_size(8)
      .hc_mult(1)
      .linear_num_key_heads(1)
      .mm_num_hidden_layers(1)
      .mm_hidden_size(8)
      .mm_num_attention_heads(2)
      .mm_num_channels(1)
      .mm_patch_size(2)
      .mm_temporal_patch_size(2)
      .mm_spatial_merge_size(2)
      .mm_image_merge_size(2)
      .mm_projection_dim(4)
      .mm_projector_hidden_size(8)
      .mm_intermediate_size(12)
      .mm_hidden_act("silu")
      .mm_layer_norm_eps(1e-5);
  return args;
}

torch::Tensor vision_test_weight(const std::string& name,
                                 const torch::Tensor& parameter,
                                 int64_t index) {
  torch::Tensor weight =
      (torch::sin(torch::arange(parameter.numel(), torch::kFloat32) * 0.37 +
                  static_cast<double>(index)) *
       0.02)
          .reshape(parameter.sizes());
  if (name.find("norm") != std::string::npos && name.ends_with("weight")) {
    weight = torch::ones_like(weight) + weight;
  }
  return weight;
}

void expect_vision_reference(const torch::Tensor& output) {
  // CPU BF16 reference with per-frame FP32 attention, fused residual RMSNorm,
  // and SiLU rounded to BF16 before the gated multiply.
  const torch::Tensor expected = torch::tensor(
      {{3.892928e-07f, -2.179295e-07f, 4.237518e-08f, 1.341105e-07f},
       {6.817281e-07f, -3.706664e-07f, 5.331822e-08f, 2.663583e-07f},
       {1.296401e-06f, -7.376075e-07f, 1.639128e-07f, 4.153699e-07f},
       {2.922025e-08f, -1.29221e-08f, -3.536115e-09f, 1.97906e-08f},
       {9.834766e-07f, -5.289912e-07f, 6.286427e-08f, 4.04194e-07f},
       {6.109476e-07f, -3.46452e-07f, 7.683411e-08f, 1.946464e-07f},
       {4.190952e-07f, -2.346933e-07f, 4.400499e-08f, 1.47149e-07f},
       {1.296401e-06f, -7.227063e-07f, 1.369044e-07f, 4.544854e-07f},
       {4.121102e-08f, -1.967419e-08f, -2.153683e-09f, 2.386514e-08f},
       {9.909272e-07f, -5.327165e-07f, 6.239861e-08f, 4.079193e-07f},
       {8.940697e-07f, -5.140901e-07f, 1.257285e-07f, 2.682209e-07f},
       {2.868474e-07f, -1.620501e-07f, 3.352761e-08f, 9.546056e-08f}});
  const torch::Tensor actual = output.to(torch::kCPU, torch::kFloat32);
  ASSERT_EQ(actual.sizes(), expected.sizes());
  EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
  const float relative_error =
      ((actual - expected).norm() / expected.norm()).item<float>();
  const float cosine =
      torch::cosine_similarity(actual.flatten(), expected.flatten(), 0)
          .item<float>();
  LOG(INFO) << "Vision BF16 relative L2 error=" << relative_error
            << ", cosine similarity=" << cosine;
  EXPECT_LT(relative_error, 0.03) << actual;
  EXPECT_GT(cosine, 0.995);
}

TEST(Glm5NextVisionTest, VisionPreservesImageAndTemporalBoundaries) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "Requires an MLU device.";
  }
  Device device(torch::Device("mlu:0"));
  device.set_device();
  ModelArgs args = small_vlm_args();
  args.mm_hidden_size(128).mm_num_attention_heads(2).mm_intermediate_size(192);
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  const ParallelArgs parallel(/*rank=*/0,
                              /*world_size=*/1,
                              /*process_group=*/nullptr);
  layer::Glm5NextVision vision(
      ModelContext(parallel, args, QuantArgs(), options));
  torch::NoGradGuard no_grad;
  std::unordered_map<std::string, torch::Tensor> weights;
  int64_t index = 0;
  for (const auto& parameter : vision->named_parameters()) {
    weights.emplace(
        parameter.key(),
        vision_test_weight(parameter.key(), parameter.value(), index++));
  }
  vision->load_state_dict(StateDict(std::move(weights)));
  vision->verify_loaded_weights();
  const torch::Tensor pixels =
      torch::sin(torch::arange(48 * 8, torch::kFloat32) / 13)
          .reshape({48, 8})
          .to(options);
  const torch::Tensor grid = torch::tensor({{1, 4, 4}, {2, 4, 4}});
  const torch::Tensor together = vision(pixels, grid);
  ASSERT_EQ(together.size(0), 12);
  ASSERT_EQ(together.size(1), 4);
  expect_vision_reference(together);
  const auto metadata = vision->prepare_metadata(grid);
  EXPECT_TRUE(
      torch::equal(vision->forward_prepared(pixels, metadata), together));
  EXPECT_TRUE(torch::equal(vision->forward_prepared(pixels * 2, metadata),
                           vision(pixels * 2, grid)));

  const torch::Tensor first =
      vision(pixels.narrow(0, 0, 16), torch::tensor({{1, 4, 4}}));
  const torch::Tensor video =
      vision(pixels.narrow(0, 16, 32), torch::tensor({{2, 4, 4}}));
  const torch::Tensor last =
      vision(pixels.narrow(0, 32, 16), torch::tensor({{1, 4, 4}}));
  EXPECT_TRUE(torch::allclose(together.narrow(0, 0, 4), first));
  EXPECT_TRUE(torch::allclose(together.narrow(0, 4, 8), video));
  EXPECT_TRUE(torch::allclose(video.narrow(0, 4, 4), last));

  ModelArgs encoder_args = args;
  encoder_args.encoder_embedding_mode(true);
  Glm5NextForConditionalGeneration encoder(
      ModelContext(parallel, encoder_args, QuantArgs(), options));
  index = 0;
  for (const auto& parameter : encoder->named_parameters()) {
    parameter.value().copy_(
        vision_test_weight(parameter.key(), parameter.value(), index++));
  }
  ModelInputParams params;
  params.multimodal.mm_data =
      MMBatchData(MMType::VIDEO,
                  {{"pixel_values_videos", pixels.narrow(0, 16, 32)},
                   {"video_grid_thw", torch::tensor({{2, 4, 4}})}});
  const MMDict encoded = encoder->get_multimodal_embeddings(params);
  const auto& clips =
      std::get<std::vector<torch::Tensor>>(encoded.at("video|embedding"));
  ASSERT_EQ(clips.size(), 1);
  EXPECT_TRUE(torch::allclose(clips[0], video));
}

TEST(Glm5NextVisionTest, MluPackedWeightsAndPreparedMetadataMatchReference) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "Requires an MLU device.";
  }
  Device device(torch::Device("mlu:0"));
  device.set_device();
  ModelArgs args = small_vlm_args();
  args.mm_hidden_size(128).mm_num_attention_heads(2).mm_intermediate_size(192);
  const ParallelArgs parallel(/*rank=*/0,
                              /*world_size=*/1,
                              /*process_group=*/nullptr);
  const auto options = torch::TensorOptions().dtype(torch::kBFloat16);
  layer::Glm5NextVision model(ModelContext(
      parallel, args, QuantArgs(), options.device(device.unwrap())));
  torch::NoGradGuard no_grad;
  int64_t index = 0;
  for (const auto& parameter : model->named_parameters()) {
    const torch::Tensor weight =
        vision_test_weight(parameter.key(), parameter.value(), index++);
    // Gate and up weights may arrive in different checkpoint shards.
    model->load_state_dict(StateDict({{parameter.key(), weight}}));
  }
  model->verify_loaded_weights();
  const torch::Tensor grid = torch::tensor({{1, 4, 4}, {2, 4, 4}});
  const auto metadata = model->prepare_metadata(grid.to(device.unwrap()));
  const torch::Tensor pixels =
      torch::sin(torch::arange(48 * 8, torch::kFloat32) / 13)
          .reshape({48, 8})
          .to(options.device(device.unwrap()));
  expect_vision_reference(model->forward_prepared(pixels, metadata));
}

class Glm5NextVlmTest : public ::testing::Test {
 protected:
  void SetUp() override {
    group_ = std::make_unique<ProcessGroup>(
        /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, group_.get());
    parallel.tp_group_ = group_.get();
    model_ = Glm5NextForConditionalGeneration(
        ModelContext(parallel,
                     small_vlm_args(),
                     QuantArgs(),
                     torch::TensorOptions().dtype(torch::kFloat32)));
    model_->get_word_embedding()->load_state_dict(StateDict(
        {{"weight", torch::arange(32, torch::kFloat32).reshape({8, 4})}}));
  }
  std::unique_ptr<ProcessGroup> group_;
  Glm5NextForConditionalGeneration model_{nullptr};
};

TEST_F(Glm5NextVlmTest, MergesCachedImageAndVideoFeaturesAtTheirMasks) {
  const torch::Tensor tokens = torch::tensor({1, 2, 3, 4}, torch::kInt32);
  ModelInputParams params;
  const torch::Tensor text =
      model_->get_input_embeddings(tokens, params).clone();
  const torch::Tensor image = torch::full({1, 4}, 2.0);
  const torch::Tensor video = torch::full({1, 4}, 3.0);
  params.multimodal.mm_data =
      MMBatchData(MMType::IMAGE | MMType::VIDEO,
                  {{"image|embedding", image},
                   {"image|mask", torch::tensor({false, true, false, false})},
                   {"video|embedding", video},
                   {"video|mask", torch::tensor({false, false, false, true})}});
  const torch::Tensor merged = model_->get_input_embeddings(tokens, params);
  EXPECT_TRUE(torch::equal(merged[0], text[0]));
  EXPECT_TRUE(torch::equal(merged[2], text[2]));
  EXPECT_TRUE(torch::equal(merged[1], image[0]));
  EXPECT_TRUE(torch::equal(merged[3], video[0]));
}

TEST_F(Glm5NextVlmTest, ExposesHybridGraphMetadataThroughCausalVlm) {
  CausalVLMImpl<Glm5NextForConditionalGeneration> vlm(
      model_, torch::TensorOptions().dtype(torch::kFloat32));
  EXPECT_TRUE(vlm.is_hybrid_linear_attention());
  EXPECT_TRUE(vlm.requires_graph_forward_metadata());
  const auto state = vlm.create_graph_forward_metadata_state();
  ASSERT_NE(state, nullptr);
  EXPECT_NE(dynamic_cast<Glm5NextGraphMetadataState*>(state.get()), nullptr);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = 1;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 1;
  // Exercise metadata forwarding before device graph capture on CPU.
  params.attention.device.block_tables = torch::tensor({{0, 1}}, torch::kInt32);
  params.embedding.linear_state_ids = {0};
  params.embedding.linear_state_indices = torch::tensor({0}, torch::kInt32);
  params.attention.device.q_seq_lens = torch::tensor({0, 1}, torch::kInt32);
  params.attention.device.kv_seq_lens = torch::tensor({0, 1}, torch::kInt32);
  params.attention.host.q_seq_lens = {0, 1};
  params.attention.host.kv_seq_lens = {0, 1};
  const torch::Tensor positions = torch::tensor({0}, torch::kInt32);
  vlm.prepare_graph_forward_metadata(state.get(), positions, params);
  ASSERT_NE(params.attn_metadata, nullptr);
  EXPECT_TRUE(params.attn_metadata->kpool_batch_metadata != nullptr);
  const auto metadata = params.attn_metadata;
  params.attention.device.kv_seq_lens = torch::tensor({0, 2}, torch::kInt32);
  vlm.prepare_graph_forward_metadata(state.get(), positions, params);
  EXPECT_EQ(params.attn_metadata, metadata);
  EXPECT_EQ(params.attn_metadata->kv_seq_lens[0].item<int32_t>(), 2);
}

}  // namespace
}  // namespace xllm::mlu::model
