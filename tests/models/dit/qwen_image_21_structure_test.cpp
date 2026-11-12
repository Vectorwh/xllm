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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "models/dit/autoencoders/autoencoder_kl_qwenimage21.h"
#include "models/dit/transformers/transformer_qwen_image_21.h"

namespace xllm {
namespace {

QwenImage21TransformerConfig small_transformer_config() {
  QwenImage21TransformerConfig config;
  config.in_channels = 4;
  config.out_channels = 4;
  config.context_in_dim = 8;
  config.num_heads = 2;
  config.head_dim = 8;
  config.num_layers = 2;
  config.axes_dims_rope = {2, 2, 4};
  return config;
}

void initialize_weights(torch::nn::Module& module) {
  torch::NoGradGuard guard;
  torch::manual_seed(21);
  for (auto& tensor : module.parameters()) {
    tensor.normal_(0.0, 0.02);
  }
}

// Expose views under the original checkpoint keys for manifest/shard checks.
// Runtime parameters remain fused and own the only copy of each weight.
torch::OrderedDict<std::string, torch::Tensor> checkpoint_parameters(
    const torch::nn::Module& module) {
  torch::OrderedDict<std::string, torch::Tensor> parameters;
  for (const auto& parameter : module.named_parameters()) {
    const auto add_parts = [&](const std::string& suffix,
                               const std::vector<std::string>& names) {
      const std::string& key = parameter.key();
      if (key.size() < suffix.size() ||
          key.compare(key.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
      }
      const std::string prefix = key.substr(0, key.size() - suffix.size());
      auto parts = parameter.value().chunk(static_cast<int64_t>(names.size()),
                                           /*dim=*/0);
      for (size_t i = 0; i < names.size(); ++i) {
        parameters.insert(prefix + names[i], parts[i]);
      }
      return true;
    };
    if (add_parts(
            "attn.to_qkv.weight",
            {"attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"}) ||
        add_parts("img_mlp.gate_up.weight",
                  {"img_mlp.gate_layer.weight", "img_mlp.proj.weight"})) {
      continue;
    }
    parameters.insert(parameter.key(), parameter.value());
  }
  return parameters;
}

TEST(QwenImage21Structure, CheckpointParameterCountsAndShapes) {
  const auto options = torch::TensorOptions().device(torch::kMeta);
  ProcessGroup process_group(/*rank=*/0, /*world_size=*/1, options.device());
  QwenImage21Transformer2DModel transformer(QwenImage21TransformerConfig{},
                                            &process_group,
                                            options.dtype(torch::kBFloat16));
  AutoencoderKLQwenImage21 vae(
      ModelContext{}, QwenImage21VaeConfig{}, options.dtype(torch::kFloat32));
  EXPECT_EQ(transformer->named_parameters().size(), 201);
  EXPECT_EQ(checkpoint_parameters(*transformer).size(), 297);
  EXPECT_EQ(vae->named_parameters().size(), 238);
  EXPECT_EQ(
      transformer->named_parameters()["transformer_blocks.0.attn.to_qkv.weight"]
          .sizes(),
      (torch::IntArrayRef{12288, 4096}));
  EXPECT_EQ(
      transformer
          ->named_parameters()["transformer_blocks.0.img_mlp.gate_up.weight"]
          .sizes(),
      (torch::IntArrayRef{24576, 4096}));
  EXPECT_EQ(transformer->named_parameters()["modulation.1.weight"].sizes(),
            (torch::IntArrayRef{16384, 4096}));
  EXPECT_EQ(vae->named_parameters()["decoder.conv_in.weight"].sizes(),
            (torch::IntArrayRef{1152, 64, 3, 3}));
  const char* manifest = std::getenv("XLLM_QWEN_IMAGE21_WEIGHT_MANIFEST");
  if (manifest == nullptr) {
    return;
  }
  std::ifstream stream(manifest);
  ASSERT_TRUE(stream.good());
  nlohmann::json expected;
  stream >> expected;
  for (const auto& [component, module] :
       std::vector<std::pair<std::string, torch::nn::Module*>>{
           {"transformer", transformer.get()}, {"vae", vae.get()}}) {
    auto parameters = checkpoint_parameters(*module);
    ASSERT_EQ(parameters.size(), expected.at(component).size());
    for (const auto& [key, value] : expected.at(component).items()) {
      ASSERT_TRUE(parameters.contains(key)) << component << "." << key;
      EXPECT_EQ(parameters[key].sizes().vec(),
                value.at("shape").get<std::vector<int64_t>>())
          << key;
      EXPECT_EQ(
          parameters[key].scalar_type(),
          component == "transformer" ? torch::kBFloat16 : torch::kFloat32);
    }
  }
}

TEST(QwenImage21Structure, AdjacentImagesStaySeparateCausalBlocks) {
  auto mask = torch::tensor({{false, true, true, false}}, torch::kBool);
  auto layout = qwen_image_21::prepare_sequence_layout(
      mask, {}, {{1, 2, 2}, {1, 2, 2}, {1, 2, 2}}, {2, 2, 4}, torch::kCPU);
  EXPECT_EQ(layout.target_tokens, 4);
  EXPECT_EQ(layout.prefix_tokens, 10);
  auto allowed = layout.allowed_attention.squeeze();
  EXPECT_TRUE(allowed[1][4].item<bool>());
  EXPECT_FALSE(allowed[1][5].item<bool>());
  EXPECT_TRUE(allowed[5][1].item<bool>());
  EXPECT_FALSE(allowed[9][10].item<bool>());
  EXPECT_TRUE(allowed[10][13].item<bool>());
  EXPECT_TRUE(torch::equal(
      layout.image_ids,
      torch::tensor({-1, 0, 0, 0, 0, 1, 1, 1, 1, -1, 2, 2, 2, 2})));
}

TEST(QwenImage21Structure, PaddingKeysAreMaskedIncludingTargetQueries) {
  auto image_mask = torch::zeros({1, 3}, torch::kBool);
  auto valid = torch::tensor({{1, 1, 0}}, torch::kLong);
  auto layout = qwen_image_21::prepare_sequence_layout(
      image_mask, valid, {{1, 2, 2}}, {2, 2, 4}, torch::kCPU);
  EXPECT_FALSE(layout.allowed_attention.select(-1, 2).any().item<bool>());
  EXPECT_TRUE(layout.allowed_attention[0][0][3][6].item<bool>());
  EXPECT_FALSE(layout.allowed_attention[0][0][0][3].item<bool>());
}

TEST(QwenImage21Structure, ExportedSingleImageLayout) {
  // Layout of edit_20261003_203530_606670; no inference or fixture payload is
  // needed.
  auto mask = torch::zeros({1, 1464}, torch::kBool);
  mask.slice(1, 8, 1032).fill_(true);
  auto layout = qwen_image_21::prepare_sequence_layout(
      mask, {}, {{1, 64, 64}, {1, 64, 64}}, {16, 56, 56}, torch::kCPU);
  EXPECT_EQ(layout.prefix_tokens, 4536);
  EXPECT_EQ(layout.target_tokens, 4096);
  EXPECT_EQ(layout.source_indices.numel(), 8632);
  EXPECT_EQ(layout.image_positions.numel(), 8192);
  EXPECT_FALSE(layout.allowed_attention[0][0][8][4536].item<bool>());
  EXPECT_TRUE(layout.allowed_attention[0][0][4536][8631].item<bool>());
}

TEST(QwenImage21Structure, TransformerRejectsMissingOrShardedTpGroup) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  auto config = small_transformer_config();
  EXPECT_DEATH((QwenImage21Transformer2DModel(config, nullptr)),
               "requires a TP process group");
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/2, torch::Device(torch::kCPU));
  EXPECT_DEATH((QwenImage21Transformer2DModel(config, &process_group)),
               "requires TP=1");
}

TEST(QwenImage21Structure, ModulationKeepsRawScalesAndTemporalSelection) {
  torch::NoGradGuard guard;
  qwen_image_21::SequenceLayout layout;
  layout.target_mask = torch::tensor({false, true, false}, torch::kBool);
  auto parameters =
      torch::tensor({{0.25f, 0.5f, 0.0f, 0.0f, -0.5f, -0.25f, 0.0f, 0.0f},
                     {-0.25f, -0.5f, 1.0f, -1.0f, 0.5f, 0.25f, -1.0f, 1.0f}});
  auto prefill = qwen_image_21::prepare_block_modulation(
      parameters, layout, /*causal_condition=*/true, /*decode=*/false);
  auto expected_scale1 =
      torch::tensor({{{-0.25f, -0.5f}, {0.25f, 0.5f}, {-0.25f, -0.5f}}});
  auto expected_scale2 =
      torch::tensor({{{0.5f, 0.25f}, {-0.5f, -0.25f}, {0.5f, 0.25f}}});
  EXPECT_TRUE(torch::equal(prefill.scale1, expected_scale1));
  EXPECT_TRUE(torch::equal(prefill.scale2, expected_scale2));
  const float gate = std::tanh(1.0f);
  EXPECT_TRUE(torch::allclose(
      prefill.gate1_tanh,
      torch::tensor({{{gate, -gate}, {0.0f, 0.0f}, {gate, -gate}}})));
  // CPU tanh implementations may round positive and negative inputs
  // differently.
  EXPECT_TRUE(torch::allclose(prefill.gate2_tanh,
                              -prefill.gate1_tanh,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));

  auto decode = qwen_image_21::prepare_block_modulation(
      parameters, layout, /*causal_condition=*/true, /*decode=*/true);
  EXPECT_EQ(decode.scale1.sizes(), (torch::IntArrayRef{1, 1, 2}));
  EXPECT_TRUE(torch::equal(decode.scale1, expected_scale1.slice(1, 1, 2)));
  EXPECT_TRUE(torch::equal(decode.scale2, expected_scale2.slice(1, 1, 2)));
  EXPECT_TRUE(torch::equal(decode.gate1_tanh, torch::zeros({1, 1, 2})));
  auto unconditioned =
      qwen_image_21::prepare_block_modulation(parameters.slice(0, 0, 1),
                                              layout,
                                              /*causal_condition=*/false,
                                              /*decode=*/false);
  EXPECT_TRUE(torch::equal(unconditioned.scale1, decode.scale1));
  EXPECT_TRUE(torch::equal(unconditioned.scale2, decode.scale2));
}

TEST(QwenImage21Structure, StaticTemporalShortcutSemantics) {
  auto input = torch::ones({1, 2, 1, 2, 2});
  auto down = qwen_image_21::average_downsample(input, 2, 2, 2);
  EXPECT_EQ(down.sizes(), (torch::IntArrayRef{1, 2, 1, 1, 1}));
  EXPECT_TRUE(torch::equal(down, torch::full_like(down, 0.5)));
  auto up = qwen_image_21::duplicate_upsample(down, 2, 2, 2);
  EXPECT_EQ(up.sizes(), input.sizes());
  EXPECT_TRUE(torch::equal(up, torch::full_like(input, 0.5)));
}

TEST(QwenImage21Structure, VaeSyntheticEncodeDecodeRgba) {
  torch::NoGradGuard guard;
  QwenImage21VaeConfig config;
  config.base_dim = 2;
  config.decoder_base_dim = 4;
  config.z_dim = 2;
  config.num_res_blocks = 1;
  config.dim_mult = {1, 2, 2};
  config.temperal_downsample = {false, true};
  config.latents_mean = {0.5, -0.5};
  config.latents_std = {2.0, 3.0};
  AutoencoderKLQwenImage21 model(ModelContext{}, config);
  initialize_weights(*model);
  auto encoded = model->encode_mode(torch::randn({1, 4, 1, 8, 12}));
  EXPECT_EQ(encoded.sizes(), (torch::IntArrayRef{1, 2, 1, 2, 3}));
  auto decoded = model->decode(encoded);
  EXPECT_EQ(decoded.sizes(), (torch::IntArrayRef{1, 4, 1, 8, 12}));
  EXPECT_TRUE(torch::isfinite(decoded).all().item<bool>());
  EXPECT_LE(decoded.abs().max().item<float>(), 1.0);
  EXPECT_TRUE(torch::allclose(
      encoded,
      model->denormalize_latents(model->normalize_latents(encoded)),
      1e-5,
      1e-6));
}

TEST(QwenImage21Structure, LayerWeightLoadingChecksMissingDuplicateAndShape) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  qwen_image_21::RmsNorm norm(3,
                              1e-6,
                              /*zero_centered=*/false,
                              torch::TensorOptions().dtype(torch::kFloat32));
  // An unrelated shard must not mark the layer loaded.
  norm->load_state_dict(StateDict({}));
  EXPECT_DEATH(norm->verify_loaded_weights("norm."), "norm.weight");
  EXPECT_DEATH(
      norm->load_state_dict(StateDict({{"weight", torch::zeros({2})}})),
      "weight size mismatch");
  norm->load_state_dict(StateDict({{"weight", torch::ones({3})}}));
  norm->verify_loaded_weights("norm.");
  EXPECT_DEATH(
      norm->load_state_dict(StateDict({{"weight", torch::zeros({3})}})),
      "weight already loaded");
}

template <typename Model>
void check_sharded_layer_loading(Model& model) {
  auto parameters = checkpoint_parameters(*model);
  std::unordered_map<std::string, torch::Tensor> first, second;
  first.reserve(parameters.size());
  second.reserve(parameters.size());
  int64_t index = 0;
  for (const auto& parameter : parameters) {
    auto& shard = index++ % 2 == 0 ? first : second;
    shard.emplace("component." + parameter.key(),
                  torch::randn(parameter.value().sizes(), torch::kFloat64));
  }
  // Match EditPlus: unused checkpoint keys are ignored by layer loaders.
  first.emplace("component.unused", torch::zeros({1}));
  StateDict first_shard(std::move(first));
  StateDict second_shard(std::move(second));
  model->load_state_dict(first_shard.get_dict_with_prefix("component."));
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(model->verify_loaded_weights("component."), "is not loaded for");
  model->load_state_dict(second_shard.get_dict_with_prefix("component."));
  model->verify_loaded_weights("component.");
  EXPECT_DEATH(
      model->load_state_dict(first_shard.get_dict_with_prefix("component.")),
      "weight already loaded");
  auto loaded = checkpoint_parameters(*model);
  for (const auto& parameter : loaded) {
    auto value = first_shard.get_tensor("component." + parameter.key());
    if (!value.defined()) {
      value = second_shard.get_tensor("component." + parameter.key());
    }
    ASSERT_TRUE(value.defined()) << parameter.key();
    EXPECT_EQ(parameter.value().data_ptr(),
              parameters[parameter.key()].data_ptr());
    EXPECT_TRUE(
        torch::equal(parameter.value(), value.to(parameter.value().options())))
        << parameter.key();
  }
}

TEST(QwenImage21Structure, TransformerLoadsWeightsWithNestedPrefixes) {
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  QwenImage21Transformer2DModel model(small_transformer_config(),
                                      &process_group);
  check_sharded_layer_loading(model);
}

TEST(QwenImage21Structure, VaeLoadsWeightsWithNestedPrefixes) {
  QwenImage21VaeConfig config;
  config.base_dim = 4;
  config.decoder_base_dim = 4;
  config.z_dim = 2;
  config.num_res_blocks = 1;
  config.dim_mult = {1, 2, 2};
  config.temperal_downsample = {false, true};
  AutoencoderKLQwenImage21 model(ModelContext{}, config);
  check_sharded_layer_loading(model);
}

TEST(QwenImage21Structure, SharedLinearMatchesBiasFreeFunctionalLinear) {
  for (const auto dtype : {torch::kFloat32, torch::kBFloat16}) {
    layer::AddMatmul linear(
        4, 3, /*with_bias=*/false, torch::TensorOptions().dtype(dtype));
    auto weight = torch::randn({3, 4}).to(dtype);
    linear->load_state_dict(StateDict({{"weight", weight}}));
    linear->verify_loaded_weights("linear.");
    EXPECT_EQ(linear->named_parameters().size(), 1);
    EXPECT_FALSE(linear->named_parameters().contains("bias"));
    auto input = torch::randn({2, 5, 4}).to(dtype);
    for (const auto& value : {input, input.transpose(0, 1), input[0]}) {
      EXPECT_TRUE(torch::equal(linear->forward(value),
                               torch::nn::functional::linear(value, weight)));
    }
  }
}

TEST(QwenImage21Structure, FusedQkvLoadsOutOfOrderAndChecksCompleteness) {
  torch::NoGradGuard guard;
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ProcessGroup process_group(
      /*rank=*/0, /*world_size=*/1, torch::Device(torch::kCPU));
  layer::ColumnParallelLinear linear(
      /*in_features=*/4,
      /*out_features=*/9,
      /*bias=*/false,
      /*gather_output=*/false,
      QuantArgs{},
      &process_group,
      torch::TensorOptions().dtype(torch::kFloat32));
  const std::vector<std::string> prefixes{"to_q.", "to_k.", "to_v."};
  auto query = torch::arange(12, torch::kFloat64).reshape({3, 4}) / 8.0;
  auto key = query + 1.0;
  auto value = query - 2.0;
  linear->load_state_dict(StateDict({}), prefixes);
  EXPECT_FALSE(linear->is_weight_loaded());
  EXPECT_DEATH(
      linear->load_state_dict(StateDict({{"to_q.weight", torch::zeros({4, 4})},
                                         {"to_k.weight", key},
                                         {"to_v.weight", value}}),
                              prefixes),
      "weight size mismatch");
  // A later projection can arrive first without shifting its destination.
  linear->load_state_dict(StateDict({{"to_k.weight", key}}), prefixes);
  EXPECT_FALSE(linear->is_weight_loaded());
  linear->load_state_dict(StateDict({{"to_q.weight", query}}), prefixes);
  EXPECT_FALSE(linear->is_weight_loaded());
  linear->load_state_dict(StateDict({{"to_v.weight", value}}), prefixes);
  ASSERT_TRUE(linear->is_weight_loaded());
  EXPECT_EQ(linear->named_parameters().size(), 1);
  auto input = torch::arange(40, torch::kFloat32).reshape({2, 5, 4}) / 16.0;
  for (const auto& x : {input, input.transpose(0, 1), input[0]}) {
    auto actual = linear->forward(x).chunk(/*chunks=*/3, /*dim=*/-1);
    EXPECT_TRUE(torch::allclose(
        actual[0],
        torch::nn::functional::linear(x, query.to(torch::kFloat32))));
    EXPECT_TRUE(torch::allclose(
        actual[1], torch::nn::functional::linear(x, key.to(torch::kFloat32))));
    EXPECT_TRUE(torch::allclose(
        actual[2],
        torch::nn::functional::linear(x, value.to(torch::kFloat32))));
  }
}

TEST(QwenImage21Structure, RejectInconsistentImageSlotCount) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      qwen_image_21::prepare_sequence_layout(torch::zeros({1, 2}, torch::kBool),
                                             {},
                                             {{1, 2, 2}, {1, 2, 2}},
                                             {2, 2, 4},
                                             torch::kCPU),
      "VLM image slots");
}

}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TEST(QwenImage21Sp, TargetShardsReassembleAndPrefixIsReplicated) {
  auto layout = qwen_image_21::prepare_sequence_layout(
      torch::tensor({{false, true, false}}, torch::kBool),
      {},
      {{1, 2, 2}, {1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  auto joint = torch::arange((layout.prefix_tokens + 12) * 8, torch::kFloat32)
                   .reshape({1, -1, 8});
  std::vector<torch::Tensor> targets;
  for (int32_t rank = 0; rank < 2; ++rank) {
    qwen_image_21::SpSequenceLayout sp(layout, rank, 2);
    auto local = sp.split_joint(joint);
    EXPECT_EQ(local.size(1), layout.prefix_tokens + 6);
    EXPECT_TRUE(torch::equal(local.slice(1, 0, layout.prefix_tokens),
                             joint.slice(1, 0, layout.prefix_tokens)));
    targets.emplace_back(local.slice(1, layout.prefix_tokens));
    EXPECT_EQ(sp.local(false).target_mask.sum().item<int64_t>(), 6);
    EXPECT_TRUE(torch::equal(
        sp.local(true).rope_cos,
        layout.rope_cos.slice(0,
                              layout.prefix_tokens + rank * 6,
                              layout.prefix_tokens + (rank + 1) * 6)));
    EXPECT_FALSE(sp.attention_mask(true).defined());
  }
  EXPECT_TRUE(torch::equal(torch::cat(targets, 1),
                           joint.slice(1, layout.prefix_tokens)));
}

TEST(QwenImage21Sp, PaddingKeysMaskedAndPaddedQueryRowsValid) {
  auto full = qwen_image_21::prepare_sequence_layout(
      torch::zeros({1, 3}, torch::kBool),
      torch::tensor({{1, 1, 0}}, torch::kLong),
      {{1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  std::vector<torch::Tensor> outputs;
  auto target = torch::arange(12, torch::kFloat32).reshape({1, 12, 1});
  for (int32_t rank = 0; rank < 5; ++rank) {
    qwen_image_21::SpSequenceLayout sp(full, rank, 5);
    EXPECT_EQ(sp.padding(), 3);
    EXPECT_EQ(sp.shard_tokens(), 3);
    EXPECT_EQ(sp.attention_mask(false).sizes(),
              (torch::IntArrayRef{1, 1, 18, 18}));
    EXPECT_FALSE(sp.attention_mask(true).select(3, 2).any().item<bool>());
    EXPECT_FALSE(sp.attention_mask(true).slice(3, 15).any().item<bool>());
    EXPECT_TRUE(sp.attention_mask(true)
                    .slice(2, 12)
                    .slice(3, 3, 15)
                    .all()
                    .item<bool>());
    EXPECT_EQ(sp.local(true).attention_query_offset, 3);
    outputs.emplace_back(sp.shard(target, 1));
    if (rank == 4) {
      EXPECT_TRUE(sp.local(true).rope_cos.eq(1).all().item<bool>());
      EXPECT_TRUE(sp.local(true).rope_sin.eq(0).all().item<bool>());
    }
  }
  EXPECT_TRUE(torch::equal(torch::cat(outputs, 1).slice(1, 0, 12), target));
}

}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TEST(QwenImage21SpDegree, AcceptsPositiveDivisorsWithoutAWhitelist) {
  for (int32_t degree : {1, 2, 4, 8, 16, 32}) {
    qwen_image_21::validate_ulysses_head_partition(32, degree);
  }
  // General rule, not a hard-coded power-of-two rule or a 32-head check.
  for (int32_t degree : {1, 2, 3, 4, 6, 8, 12, 24}) {
    qwen_image_21::validate_ulysses_head_partition(24, degree);
  }
}

TEST(QwenImage21SpDegree, RejectsInvalidDegreesBeforeModuloOrCommunication) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(32, 0),
               "sp_size must be positive");
  EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(32, -2),
               "sp_size must be positive");
  EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(0, 2),
               "num_heads must be positive");
  for (int32_t degree : {3, 5, 6, 7, 31, 33, 64}) {
    EXPECT_DEATH(qwen_image_21::validate_ulysses_head_partition(32, degree),
                 "num_heads divisible by sp_size");
  }
}

TEST(QwenImage21SpDegree, HigherDegreesPreservePrefixRopeAndPadding) {
  // Twelve target tokens forces padding for SP8/16/32, including ranks whose
  // shards consist entirely of padding. No rank can expose dummy keys.
  auto full = qwen_image_21::prepare_sequence_layout(
      torch::tensor({{false, true, false, false}}, torch::kBool),
      torch::tensor({{1, 1, 1, 0}}, torch::kLong),
      {{1, 2, 2}, {1, 2, 6}},
      {2, 2, 4},
      torch::kCPU);
  auto joint = torch::arange((full.prefix_tokens + 12) * 8, torch::kFloat32)
                   .reshape({1, -1, 8});
  for (int32_t degree : {2, 4, 8, 16, 32}) {
    qwen_image_21::validate_ulysses_head_partition(32, degree);
    std::vector<torch::Tensor> targets, cosines, sines;
    targets.reserve(degree);
    cosines.reserve(degree);
    sines.reserve(degree);
    for (int32_t rank = 0; rank < degree; ++rank) {
      qwen_image_21::SpSequenceLayout sp(full, rank, degree);
      auto local = sp.split_joint(joint);
      EXPECT_EQ(local.size(1), full.prefix_tokens + sp.shard_tokens());
      EXPECT_TRUE(torch::equal(local.slice(1, 0, full.prefix_tokens),
                               joint.slice(1, 0, full.prefix_tokens)));
      targets.emplace_back(local.slice(1, full.prefix_tokens));
      cosines.emplace_back(sp.local(true).rope_cos);
      sines.emplace_back(sp.local(true).rope_sin);
      EXPECT_EQ(sp.local(true).attention_query_offset, full.prefix_tokens);
      EXPECT_FALSE(sp.attention_mask(true)
                       .select(3, full.prefix_tokens - 1)
                       .any()
                       .item<bool>());
      EXPECT_TRUE(sp.attention_mask(true).any(-1).all().item<bool>());
      if (sp.padding() > 0) {
        EXPECT_FALSE(sp.attention_mask(true)
                         .slice(3, full.prefix_tokens + 12)
                         .any()
                         .item<bool>());
      }
    }
    auto target = torch::cat(targets, 1);
    EXPECT_TRUE(torch::equal(target.slice(1, 0, 12),
                             joint.slice(1, full.prefix_tokens)));
    EXPECT_TRUE(target.slice(1, 12).eq(0).all().item<bool>());
    auto cosine = torch::cat(cosines, 0);
    auto sine = torch::cat(sines, 0);
    EXPECT_TRUE(torch::equal(cosine.slice(0, 0, 12),
                             full.rope_cos.slice(0, full.prefix_tokens)));
    EXPECT_TRUE(torch::equal(sine.slice(0, 0, 12),
                             full.rope_sin.slice(0, full.prefix_tokens)));
    EXPECT_TRUE(cosine.slice(0, 12).eq(1).all().item<bool>());
    EXPECT_TRUE(sine.slice(0, 12).eq(0).all().item<bool>());
  }
}

}  // namespace
}  // namespace xllm

namespace xllm {
namespace {
TaylorSeerOptions taylor_options(int32_t warmup,
                                 int32_t interval,
                                 int32_t order) {
  TaylorSeerOptions result;
  result.warmup_steps = warmup;
  result.skip_interval_steps = interval;
  result.n_derivatives = order;
  return result;
}

TEST(QwenImage21TaylorSeer, PrefixPrefillIsExcludedAndFirstDecodeForcedFull) {
  qwen_image_21::RequestTaylorSeer state(taylor_options(0, 2, 1), 6);
  EXPECT_FALSE(state.begin_step(1, true));
  state.record(torch::ones({1, 12, 8}));
  EXPECT_EQ(state.bytes(), 0);
  EXPECT_FALSE(state.begin_step(2, false));
  state.record(torch::full({1, 4, 8}, 2.0));
  EXPECT_TRUE(state.begin_step(3, false));
  EXPECT_TRUE(torch::equal(state.predict(torch::zeros({1, 4, 8})),
                           torch::full({1, 4, 8}, 2.0)));
  EXPECT_FALSE(state.begin_step(4, false));
  state.record(torch::full({1, 4, 8}, 4.0));
  EXPECT_TRUE(state.begin_step(5, false));
  EXPECT_TRUE(torch::equal(state.predict(torch::zeros({1, 4, 8})),
                           torch::full({1, 4, 8}, 5.0)));
  EXPECT_FALSE(state.begin_step(6, false));
  state.record(torch::full({1, 4, 8}, 6.0));
  EXPECT_EQ(state.full_steps(), 4);
  EXPECT_EQ(state.hit_steps(), 2);
}

TEST(QwenImage21TaylorSeer, FiniteDifferencesAndFactorialScaling) {
  qwen_image_21::RequestTaylorSeer state(taylor_options(3, 2, 2), 5);
  for (int64_t step = 1; step <= 3; ++step) {
    EXPECT_FALSE(state.begin_step(step, false));
    state.record(torch::full({1}, static_cast<float>(step * step)));
  }
  EXPECT_EQ(state.available_order(), 2);
  EXPECT_TRUE(state.begin_step(4, false));
  // Backward finite differences: Y=9, D1=5, D2=2 -> 9+5+2/2=15.
  EXPECT_FLOAT_EQ(state.predict(torch::zeros({1})).item<float>(), 15);
  EXPECT_FALSE(state.begin_step(5, false));
  state.record(torch::full({1}, 25.0f));
}

TEST(QwenImage21TaylorSeer, ScheduleUsesActualStepsAndIsRankIndependent) {
  for (bool prefix : {false, true}) {
    qwen_image_21::RequestTaylorSeer a(taylor_options(8, 2, 1), 50);
    qwen_image_21::RequestTaylorSeer b(taylor_options(8, 2, 1), 50);
    for (int64_t step = 1; step <= 50; ++step) {
      const bool hit = a.begin_step(step, prefix && step == 1);
      EXPECT_EQ(hit, b.begin_step(step, prefix && step == 1));
      if (hit) {
        a.predict(torch::zeros({1, 5, 8}));
        b.predict(torch::ones({1, 7, 8}));
      } else {
        a.record(torch::full({1, 5, 8}, static_cast<float>(step)));
        b.record(torch::full({1, 7, 8}, static_cast<float>(step * 2)));
      }
    }
    EXPECT_EQ(a.full_steps(), 29);
    EXPECT_EQ(a.hit_steps(), 21);
  }
}

TEST(QwenImage21TaylorSeer, ShortRequestsAndDisabledSkippingAlwaysCompute) {
  for (int64_t total : {2, 3, 8}) {
    qwen_image_21::RequestTaylorSeer state(taylor_options(8, 2, 1), total);
    for (int64_t step = 1; step <= total; ++step) {
      EXPECT_FALSE(state.begin_step(step, step == 1));
      state.record(torch::zeros({1, 4, 8}));
    }
    EXPECT_EQ(state.hit_steps(), 0);
  }
  qwen_image_21::RequestTaylorSeer state(taylor_options(0, 1, 4), 10);
  for (int64_t step = 1; step <= 10; ++step) {
    EXPECT_FALSE(state.begin_step(step, false));
    state.record(torch::ones({1}));
  }
  EXPECT_EQ(state.hit_steps(), 0);
}

TEST(QwenImage21TaylorSeer, OwnsHistoryAndBranchesDoNotLeak) {
  qwen_image_21::RequestTaylorSeer positive(taylor_options(0, 3, 0), 4);
  qwen_image_21::RequestTaylorSeer negative(taylor_options(0, 3, 0), 4);
  auto output = torch::ones({1, 4, 8}, torch::kBFloat16);
  EXPECT_FALSE(positive.begin_step(1, false));
  positive.record(output);
  output.fill_(7);
  EXPECT_FALSE(negative.begin_step(1, false));
  negative.record(output);
  EXPECT_TRUE(positive.begin_step(2, false));
  EXPECT_TRUE(negative.begin_step(2, false));
  EXPECT_TRUE(positive.predict(output).eq(1).all().item<bool>());
  EXPECT_TRUE(negative.predict(output).eq(7).all().item<bool>());
  EXPECT_EQ(positive.bytes(), 1 * 4 * 8 * sizeof(float));
  qwen_image_21::RequestTaylorSeer fresh(taylor_options(0, 3, 0), 4);
  EXPECT_EQ(fresh.bytes(), 0);
  EXPECT_FALSE(fresh.begin_step(1, false));
}

TEST(QwenImage21TaylorSeer, InvalidOptionsAndSequencingAreRejected) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(0, 0, 1), 4),
               "interval must be positive");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(-1, 2, 1), 4),
               "warmup_steps");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(0, 2, -1), 4),
               "n_derivatives");
  EXPECT_DEATH(qwen_image_21::RequestTaylorSeer(taylor_options(0, 2, 5), 4),
               "order 0..4");
  qwen_image_21::RequestTaylorSeer state(taylor_options(0, 2, 1), 5);
  EXPECT_DEATH(state.begin_step(2, false), "consecutive");
  state.begin_step(1, false);
  EXPECT_DEATH(state.begin_step(2, false), "not recorded");
  state.record(torch::zeros({1, 4, 8}));
  state.begin_step(2, false);
  EXPECT_DEATH(state.predict(torch::zeros({1, 8, 8})), "shape changed");
}

}  // namespace
}  // namespace xllm
