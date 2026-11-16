/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/batch.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/forward_shared_memory_manager.h"
#include "core/runtime/params_utils.h"

namespace xllm {

#if defined(USE_NPU)
TEST(BatchPackedInputTest,
     DeferredSharedMemoryMaterializationReusesPinnedPayload) {
  ASSERT_TRUE(ExecutionConfig::get_instance().use_contiguous_input_buffer());
  const torch::Device device(torch::kPrivateUse1, 0);
  ForwardInput input;
  input.token_ids = torch::tensor({1, 2, 3}, torch::kInt32);
  input.positions = torch::tensor({0, 1, 2}, torch::kInt32);

  const std::string name = ForwardSharedMemoryManager::create_unique_name(
      "batch_test_deferred_pinned_payload",
      /*dp_group=*/0,
      ForwardType::RAW_INPUT,
      /*rank=*/0);
  bool is_creator = false;
  ForwardSharedMemoryManager writer(
      name, /*size=*/1 << 20, is_creator, ForwardType::RAW_INPUT);
  bool is_reader_creator = false;
  ForwardSharedMemoryManager reader(
      name, /*size=*/1 << 20, is_reader_creator, ForwardType::RAW_INPUT);
  ASSERT_TRUE(writer.input_write(input));

  ForwardInput deferred_input;
  reader.input_read(deferred_input,
                    device,
                    InputDeviceMaterializationPolicy::DEFER_TO_WORKER_PREPARE);
  ASSERT_TRUE(deferred_input.input_host_buffer.is_pinned());
  ASSERT_FALSE(deferred_input.device_tensors_ready);
  const void* payload = deferred_input.input_host_buffer.data_ptr();
  input.token_ids.fill_(9);
  ASSERT_TRUE(writer.input_write(input));

  const ForwardInput materialized_input =
      deferred_input.to(device, torch::kFloat32);
  EXPECT_EQ(materialized_input.input_host_buffer.data_ptr(), payload);
  EXPECT_TRUE(materialized_input.input_host_buffer.is_pinned());
  EXPECT_TRUE(materialized_input.device_tensors_ready);
  EXPECT_EQ(materialized_input.device_input_buffer.device(), device);
  deferred_input = ForwardInput();
  EXPECT_TRUE(torch::equal(materialized_input.token_ids.cpu(),
                           torch::tensor({1, 2, 3}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(materialized_input.token_ids_host,
                           torch::tensor({1, 2, 3}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(materialized_input.positions.cpu(),
                           torch::tensor({0, 1, 2}, torch::kInt32)));
}

TEST(BatchPackedInputTest, MaterializationPinsPageablePackedPayload) {
  ASSERT_TRUE(ExecutionConfig::get_instance().use_contiguous_input_buffer());
  const torch::Device device(torch::kPrivateUse1, 0);
  ForwardInput input;
  input.token_ids = torch::tensor({4, 5}, torch::kInt32);
  input.positions = torch::tensor({0, 1}, torch::kInt32);
  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(packed_input, lazy_input, device, nullptr);
  lazy_input.input_host_buffer = lazy_input.input_host_buffer.clone();
  ASSERT_FALSE(lazy_input.input_host_buffer.is_pinned());

  const ForwardInput materialized_input =
      lazy_input.to(device, torch::kFloat32);
  EXPECT_NE(materialized_input.input_host_buffer.data_ptr(),
            lazy_input.input_host_buffer.data_ptr());
  EXPECT_TRUE(materialized_input.input_host_buffer.is_pinned());
  EXPECT_TRUE(torch::equal(materialized_input.input_host_buffer,
                           lazy_input.input_host_buffer));
  lazy_input = ForwardInput();
  EXPECT_TRUE(
      torch::equal(materialized_input.token_ids.cpu(), input.token_ids));
  EXPECT_TRUE(
      torch::equal(materialized_input.positions.cpu(), input.positions));
}
#endif

namespace {

class ScopedDeathTestConfiguration final {
 public:
  ScopedDeathTestConfiguration()
      : death_test_style_(::testing::FLAGS_gtest_death_test_style),
        log_to_stderr_(FLAGS_logtostderr) {}

  ~ScopedDeathTestConfiguration() {
    ::testing::FLAGS_gtest_death_test_style = std::move(death_test_style_);
    FLAGS_logtostderr = log_to_stderr_;
  }

 private:
  std::string death_test_style_;
  bool log_to_stderr_;
};

class PackedDecodeInput final {
 public:
  PackedDecodeInput() : manager_(make_options()) {
    sampling_param_.logprobs = true;
    stopping_checker_.set_max_generated_tokens(4);
  }

  ForwardInput make_decode_input(int32_t num_sequences = 1) {
    SequenceParams params;
    params.seq_capacity = 32;
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;
    params.enable_schedule_overlap = true;
    Batch batch;
    for (int32_t row = 0; row < num_sequences; ++row) {
      const int32_t first_token = row * 4 + 1;
      sequences_.emplace_back(std::make_unique<Sequence>(
          row,
          std::vector<int32_t>{first_token, first_token + 1, first_token + 2},
          torch::Tensor(),
          MMData(),
          IncrementalDecoder("", 3, false, false),
          params));
      Sequence* sequence = sequences_.back().get();
      sequence->kv_state().add_blocks(BlockType::KV, manager_.allocate(1));
      sequence->kv_state().set_kv_cache_tokens_num(3);
      sequence->append_token(first_token + 3);
      batch.add(sequence, 1);
    }
    return batch.prepare_forward_input(1, 0, ModelArgs());
  }

 private:
  static BlockManager::Options make_options() {
    BlockManager::Options options;
    options.num_blocks(8).block_size(4);
    return options;
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  BlockManagerImpl manager_;
  std::vector<std::unique_ptr<Sequence>> sequences_;
};

template <typename T>
bool tensor_equals_vector(const torch::Tensor& tensor,
                          const std::vector<T>& values) {
  auto flat = tensor.flatten();
  if (flat.size(0) != values.size()) {
    return false;
  }
  for (int64_t i = 0; i < flat.size(0); ++i) {
    if (flat[i].item<T>() != values[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(BatchPackedInputTest, ModelInputConversionExpandsMixedAdapterIds) {
  ModelInputParams input_params;
  input_params.adapter_ids = {7, 0, 11};
#if defined(USE_NPU)
  input_params.attention.host.q_seq_lens = {2, 1, 3};
#else
  input_params.attention.host.q_seq_lens = {0, 2, 3, 6};
#endif
  const ModelInputParams converted =
      input_params.to(torch::Device(torch::kCPU));
  ASSERT_TRUE(converted.adapter_ids_per_token.defined());
  EXPECT_TRUE(tensor_equals_vector<int64_t>(converted.adapter_ids_per_token,
                                            {7, 7, 0, 11, 11, 11}));
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackPreservesLinearReadWriteIds) {
  PackedDecodeInput fixture;
  ForwardInput input = fixture.make_decode_input(2);
  ASSERT_EQ(input.input_params.meta.num_sequences, 2);
  input.input_params.embedding.engram_lookback_token_ids = {24, 23, 34, 33};
  input.input_params.embedding.linear_state_ids = {7, 8};
  input.input_params.embedding.linear_state_indices =
      torch::tensor({7, 8}, torch::kInt32);
  input.input_params.embedding.linear_state_read_ids = {3, 4};
  input.input_params.embedding.linear_state_read_indices =
      torch::tensor({3, 4}, torch::kInt32);

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_TRUE(lazy_input.input_params.embedding.linear_state_read_ids.empty());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  unpacked_input.input_params.dit_forward_input.emplace().batch_size = 1;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  EXPECT_EQ(unpacked_input.input_params.embedding.engram_lookback_token_ids,
            (std::vector<int32_t>{24, 23, 34, 33}));
  EXPECT_FALSE(unpacked_input.input_params.dit_forward_input.has_value());
  EXPECT_EQ(unpacked_input.input_params.embedding.linear_state_ids,
            std::vector<int32_t>({7, 8}));
  EXPECT_EQ(unpacked_input.input_params.embedding.linear_state_read_ids,
            std::vector<int32_t>({3, 4}));
  EXPECT_FALSE(unpacked_input.input_params.embedding.linear_state_read_indices
                   .defined());
  ModelInputParams device_params =
      unpacked_input.input_params.to(torch::Device(torch::kCPU));
  EXPECT_EQ(device_params.embedding.engram_lookback_token_ids,
            (std::vector<int32_t>{24, 23, 34, 33}));
  EXPECT_TRUE(torch::equal(device_params.embedding.linear_state_read_indices,
                           torch::tensor({3, 4}, torch::kInt32)));
}

TEST(BatchPackedInputTest,
     LinearReadIndicesPreserveHostIdsAndClearWithTargetState) {
  ModelInputParams params;
  params.meta.num_sequences = 2;
  params.linear_state_validity_mask = {0, 1};
  params.linear_state_validity_mask_tensor =
      torch::tensor({0, 1}, torch::kInt32);
  params.embedding.linear_state_ids = {7, 8};
  params.embedding.linear_state_read_ids = {3, 4};
  ModelInputParams device_params = params.to(torch::Device(torch::kCPU));
  EXPECT_EQ(device_params.embedding.linear_state_read_ids,
            std::vector<int32_t>({3, 4}));
  EXPECT_TRUE(torch::equal(device_params.embedding.linear_state_read_indices,
                           torch::tensor({3, 4}, torch::kInt32)));
  EXPECT_EQ(device_params.linear_state_validity_mask,
            (std::vector<int64_t>{0, 1}));
  EXPECT_TRUE(torch::equal(device_params.linear_state_validity_mask_tensor,
                           torch::tensor({0, 1}, torch::kInt32)));
  device_params.clear_linear_attention_state();
  EXPECT_TRUE(device_params.linear_state_validity_mask.empty());
  EXPECT_FALSE(device_params.linear_state_validity_mask_tensor.defined());
  EXPECT_TRUE(device_params.embedding.linear_state_ids.empty());
  EXPECT_TRUE(device_params.embedding.linear_state_read_ids.empty());
  EXPECT_FALSE(device_params.embedding.linear_state_indices.defined());
  EXPECT_FALSE(device_params.embedding.linear_state_read_indices.defined());
}

TEST(BatchPackedInputTest, PackedProtoLazyUnpackRestoresSampleIdxes) {
  PackedDecodeInput fixture;
  ForwardInput input = fixture.make_decode_input(1);
  ASSERT_TRUE(input.sampling_params.sample_idxes.defined());
  input.sampling_params.filter_bitmask =
      torch::tensor({{static_cast<int32_t>(0x5)}},
                    torch::TensorOptions().dtype(torch::kInt32));

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_FALSE(lazy_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);

  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));
  ASSERT_TRUE(unpacked_input.sampling_params.sample_idxes.defined());
  EXPECT_TRUE(tensor_equals_vector<int32_t>(
      unpacked_input.sampling_params.sample_idxes, {0}));
  ASSERT_TRUE(unpacked_input.sampling_params.filter_bitmask.defined());
  EXPECT_TRUE(torch::equal(unpacked_input.sampling_params.filter_bitmask,
                           input.sampling_params.filter_bitmask));
}

TEST(BatchPackedInputTest, PackedProtoPreservesDiTGenerationParams) {
  ForwardInput input;
  DiTForwardInput& dit_input = input.input_params.dit_forward_input.emplace();
  dit_input.batch_size = 1;
  dit_input.media_sources.add(
      "prompt_video",
      "video",
      torch::tensor({1, 2}, torch::dtype(torch::kUInt8)),
      {{std::string("prompt_video_fps"), 24.0}});
  dit_input.media_sources.add(
      "image", "image", torch::tensor({3, 4}, torch::dtype(torch::kUInt8)));
  dit_input.media_sources.add(
      "mask_image",
      "image",
      torch::tensor({5, 6}, torch::dtype(torch::kUInt8)));
  dit_input.tensor_sources.add(
      "prompt_embed",
      torch::tensor({1.5f, 2.5f}),
      {{std::string("prompt_token_tags"), std::vector<int64_t>{0, 1}}});
  dit_input.tensor_sources.add("latent", torch::tensor({3.5f, 4.5f}));
  DiTGenerationParams& params = dit_input.generation_params;
  params.width = 640;
  params.height = 480;
  params.num_inference_steps = 31;
  params.true_cfg_scale = 2.5f;
  params.guidance_scale = 6.5f;
  params.num_images_per_prompt = 2;
  params.num_videos_per_prompt = 3;
  params.seed = 1234567;
  params.seed_is_set = true;
  params.max_sequence_length = 768;
  params.strength = 0.75f;
  params.enable_cfg_renorm = false;
  params.cfg_renorm_min = 0.25f;
  params.audio_duration_frames = 321;
  params.audio_steps = 24;
  params.audio_guidance_method = "apg";
  params.audio_sampling_rate = 48000;
  params.num_frames = 49;
  params.video_fps = 12.5;
  params.guidance_scale_2 = 4.25f;
  params.seconds = 7;
  params.boundary_ratio = 0.8f;
  params.flow_shift = 1.25f;
  params.max_new_tokens = 96;
  params.diffusion_steps = 12;
  params.temperature = 0.6f;
  params.top_k = 32;
  params.top_p = 0.85f;
  params.repetition_penalty = 1.2f;

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  ForwardInput unpacked_input;
  ASSERT_TRUE(detail::unpack_from_input_host_buffer(lazy_input,
                                                    torch::Device(torch::kCPU),
                                                    torch::kFloat32,
                                                    unpacked_input,
                                                    false));

  ASSERT_TRUE(unpacked_input.input_params.dit_forward_input.has_value());
  const DiTForwardInput& unpacked_dit_input =
      *unpacked_input.input_params.dit_forward_input;
  // Direct fields are the wire oracle: using only
  // DiTGenerationParams::operator== could conceal a transport omission if that
  // operator also missed the field.
  const auto expect_text_params = [](const DiTGenerationParams& restored) {
    EXPECT_EQ(restored.max_new_tokens, 96);
    EXPECT_EQ(restored.diffusion_steps, 12);
    EXPECT_FLOAT_EQ(restored.temperature, 0.6F);
    EXPECT_EQ(restored.top_k, 32);
    EXPECT_FLOAT_EQ(restored.top_p, 0.85F);
    EXPECT_FLOAT_EQ(restored.repetition_penalty, 1.2F);
    EXPECT_TRUE(restored.seed_is_set);
    EXPECT_EQ(restored.seed, 1234567);
  };
  expect_text_params(unpacked_dit_input.generation_params);
  EXPECT_EQ(unpacked_dit_input.generation_params, params);
  ASSERT_EQ(unpacked_dit_input.media_sources.size(), 3u);
  EXPECT_EQ(unpacked_dit_input.media_sources.at(0).name, "prompt_video");
  EXPECT_EQ(unpacked_dit_input.media_sources.at(0).modality, "video");
  EXPECT_EQ(unpacked_dit_input.media_sources.at(1).name, "image");
  EXPECT_EQ(unpacked_dit_input.media_sources.at(2).name, "mask_image");
  EXPECT_TRUE(torch::equal(unpacked_dit_input.media_sources.at(0).tensor,
                           dit_input.media_sources.at(0).tensor));
  EXPECT_EQ(unpacked_dit_input.media_sources.at(0).parameters,
            dit_input.media_sources.at(0).parameters);
  ASSERT_EQ(unpacked_dit_input.tensor_sources.size(), 2u);
  EXPECT_TRUE(
      torch::equal(*unpacked_dit_input.tensor_sources.get("prompt_embed"),
                   *dit_input.tensor_sources.get("prompt_embed")));
  EXPECT_EQ(unpacked_dit_input.tensor_sources.get_named_tensor("prompt_embed")
                ->get()
                .parameters,
            dit_input.tensor_sources.get_named_tensor("prompt_embed")
                ->get()
                .parameters);
  EXPECT_TRUE(torch::equal(*unpacked_dit_input.tensor_sources.get("latent"),
                           *dit_input.tensor_sources.get("latent")));

  proto::DiTForwardInput proto_input;
  ASSERT_TRUE(dit_forward_input_to_proto(dit_input, &proto_input));
  DiTForwardInput proto_input_round_trip;
  ASSERT_TRUE(proto_to_dit_forward_input(proto_input, proto_input_round_trip));
  ASSERT_EQ(proto_input_round_trip.media_sources.size(), 3u);
  EXPECT_EQ(proto_input_round_trip.media_sources.at(2).name, "mask_image");
  EXPECT_EQ(proto_input_round_trip.media_sources.at(2).modality, "image");
  EXPECT_TRUE(torch::equal(proto_input_round_trip.media_sources.at(1).tensor,
                           dit_input.media_sources.at(1).tensor));
  EXPECT_EQ(proto_input_round_trip.media_sources.at(0).parameters,
            dit_input.media_sources.at(0).parameters);
  EXPECT_TRUE(
      torch::equal(*proto_input_round_trip.tensor_sources.get("prompt_embed"),
                   *dit_input.tensor_sources.get("prompt_embed")));
  EXPECT_EQ(
      proto_input_round_trip.tensor_sources.get_named_tensor("prompt_embed")
          ->get()
          .parameters,
      dit_input.tensor_sources.get_named_tensor("prompt_embed")
          ->get()
          .parameters);
  EXPECT_TRUE(torch::equal(*proto_input_round_trip.tensor_sources.get("latent"),
                           *dit_input.tensor_sources.get("latent")));

  expect_text_params(proto_input_round_trip.generation_params);
}

TEST(BatchPackedInputTest, PackedProtoLazyToPreservesJsonMetadata) {
  PackedDecodeInput fixture;
  ForwardInput input = fixture.make_decode_input(1);
  JsonObjectGrammar grammar({"{", "}", "stop"}, /*stop_token_ids=*/{2});
  JsonObjectGrammarState state = grammar.initial_state();
  ASSERT_TRUE(state.accept_token(/*open_object=*/0));
  input.json_object_state_snapshots = {state.snapshot()};
  input.sample_sequence_ids = {"req-json#0"};
  input.sample_prior_output_rows = {-1};

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));

  ForwardInput lazy_input;
  packed_proto_to_forward_input(
      packed_input, lazy_input, torch::Device(torch::kCPU), nullptr);
  EXPECT_TRUE(lazy_input.input_host_buffer_has_layout);
  EXPECT_TRUE(lazy_input.json_object_state_snapshots.empty());

  const ForwardInput materialized_input =
      lazy_input.to(torch::Device(torch::kCPU), torch::kFloat32);
  ASSERT_EQ(materialized_input.json_object_state_snapshots.size(), 1u);
  EXPECT_EQ(materialized_input.json_object_state_snapshots[0].token_ids,
            std::vector<int32_t>({0}));
  EXPECT_FALSE(
      materialized_input.json_object_state_snapshots[0].reasoning_enabled);
  EXPECT_EQ(materialized_input.sample_sequence_ids,
            std::vector<std::string>({"req-json#0"}));
  EXPECT_EQ(materialized_input.sample_prior_output_rows,
            std::vector<int32_t>({-1}));
}

TEST(BatchPackedInputTest, PackedProtoRejectsIncompatibleLayoutHeader) {
  ScopedDeathTestConfiguration death_test_configuration;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  FLAGS_logtostderr = true;
  constexpr size_t kRawLayoutHeaderBytes = 4 * sizeof(uint64_t);
  PackedDecodeInput fixture;
  ForwardInput input = fixture.make_decode_input();

  proto::PackedForwardInput packed_input;
  ASSERT_TRUE(forward_input_to_packed_proto(input, &packed_input));
  const std::string payload = packed_input.payload();
  ASSERT_GE(payload.size(), kRawLayoutHeaderBytes);

  // v5 has no occurrence identity; reject it before parsing the descriptor.
  proto::PackedForwardInput v5_input = packed_input;
  const uint64_t v5_tag = 0x584C4C4D00000005ULL;
  std::memcpy(v5_input.mutable_payload()->data(), &v5_tag, sizeof(v5_tag));
  ForwardInput v5_lazy;
  packed_proto_to_forward_input(
      v5_input, v5_lazy, torch::Device(torch::kCPU), nullptr);
  ForwardInput v5_unpacked;
  EXPECT_DEATH(detail::unpack_from_input_host_buffer(
                   v5_lazy,
                   torch::Device(torch::kCPU),
                   torch::kFloat32,
                   v5_unpacked,
                   /*materialize_device_buffer=*/false),
               "incompatible raw input format");

  // An engine build without the format tag leads with the old descriptor
  // length instead, which must not be parsed as a matching tag.
  std::string legacy_payload = payload;
  std::memcpy(legacy_payload.data(),
              payload.data() + sizeof(uint64_t),
              sizeof(uint64_t));
  proto::PackedForwardInput legacy_input;
  legacy_input.mutable_payload()->swap(legacy_payload);
  ForwardInput legacy_lazy;
  packed_proto_to_forward_input(
      legacy_input, legacy_lazy, torch::Device(torch::kCPU), nullptr);
  ForwardInput legacy_unpacked;
  EXPECT_DEATH(detail::unpack_from_input_host_buffer(
                   legacy_lazy,
                   torch::Device(torch::kCPU),
                   torch::kFloat32,
                   legacy_unpacked,
                   /*materialize_device_buffer=*/false),
               "incompatible raw input format");

  // An untrusted arena length must not wrap the layout bounds check.
  std::string corrupt_payload = payload;
  const uint64_t huge_arena_bytes = std::numeric_limits<uint64_t>::max();
  std::memcpy(corrupt_payload.data() + 3 * sizeof(uint64_t),
              &huge_arena_bytes,
              sizeof(huge_arena_bytes));
  proto::PackedForwardInput corrupt_input;
  corrupt_input.mutable_payload()->swap(corrupt_payload);
  ForwardInput corrupt_lazy;
  packed_proto_to_forward_input(
      corrupt_input, corrupt_lazy, torch::Device(torch::kCPU), nullptr);
  ForwardInput corrupt_unpacked;
  EXPECT_DEATH(detail::unpack_from_input_host_buffer(
                   corrupt_lazy,
                   torch::Device(torch::kCPU),
                   torch::kFloat32,
                   corrupt_unpacked,
                   /*materialize_device_buffer=*/false),
               "raw input layout overflow");
}

}  // namespace xllm
