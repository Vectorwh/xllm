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

#include "core/framework/multimodal/mm_data.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_visitor.h"

namespace xllm {
namespace {

struct TypeMaskCase {
  uint32_t stored_type;
  uint32_t query_type;
  bool expected;
};

class MMDataTypeMaskTest : public ::testing::TestWithParam<TypeMaskCase> {};

TEST_P(MMDataTypeMaskTest, ChecksDataTypes) {
  const TypeMaskCase& test_case = GetParam();
  MMData data;
  data.set(test_case.stored_type, MMDict{});

  EXPECT_EQ(data.has(test_case.query_type), test_case.expected);
  EXPECT_EQ(data.has(MMType{static_cast<MMType::Value>(test_case.query_type)}),
            test_case.expected);
}

TEST_P(MMDataTypeMaskTest, ChecksBatchTypes) {
  const TypeMaskCase& test_case = GetParam();
  const MMBatchData data(test_case.stored_type, MMDict{});

  EXPECT_EQ(data.has(test_case.query_type), test_case.expected);
}

TEST(MMDataEncoderHandoffTest, RestoresIndependentModalityMasksInChunk) {
  MMDataItem video(MMType::VIDEO);
  auto& video_state = video.mutable_state();
  video_state.mutable_token_pos() = {0, 4};
  video_state.mutable_mm_token_num() = 2;
  video_state.mutable_mm_token_mask() =
      torch::tensor({false, true, false, true}, torch::kBool);
  video_state.mutable_seq_index() = 0;
  video_state.mutable_schedule_data().start_pos = 2;
  video_state.mutable_schedule_data().end_pos = 4;

  MMDataItem image(MMType::IMAGE);
  auto& image_state = image.mutable_state();
  image_state.mutable_token_pos() = {4, 1};
  image_state.mutable_mm_token_num() = 1;
  image_state.mutable_mm_token_mask() = torch::ones({1}, torch::kBool);
  image_state.mutable_seq_index() = 0;
  image_state.mutable_schedule_data().end_pos = 1;
  // An already-embedded item must bypass encoder output scattering.
  image.add("image|embedding", torch::full({1, 2}, 14.0));

  MMData data(MMType::VIDEO | MMType::IMAGE, MMItemVec{video, image});
  MMBatchData batch(std::vector<MMData>{data});
  MMTokenNumVisitor video_counts(MMType::VIDEO);
  MMTokenNumVisitor image_counts(MMType::IMAGE);
  ASSERT_TRUE(batch.foreach (video_counts));
  ASSERT_TRUE(batch.foreach (image_counts));
  EXPECT_EQ(video_counts.token_nums(), (std::vector<int32_t>{2}));
  EXPECT_TRUE(image_counts.token_nums().empty());

  MMDict encoded;
  encoded["video|embedding"] =
      std::vector<torch::Tensor>{torch::tensor({{20.0, 20.0}, {27.0, 27.0}})};
  EncoderOutputScatterVisitor scatter(encoded);
  ASSERT_TRUE(batch.foreach (scatter));
  ASSERT_TRUE(scatter.finish());
  EncoderEmbeddingGatherVisitor gather(
      /*num_computed_tokens=*/{2}, /*num_scheduled_tokens=*/{3});
  ASSERT_TRUE(batch.foreach (gather));
  ASSERT_TRUE(gather.finish(batch));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("video|embedding").value(),
                           torch::full({1, 2}, 27.0)));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("image|embedding").value(),
                           torch::full({1, 2}, 14.0)));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("video|mask").value(),
                           torch::tensor({false, true, false}, torch::kBool)));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("image|mask").value(),
                           torch::tensor({false, false, true}, torch::kBool)));
}

TEST(MMDataEncoderHandoffTest, RejectsScheduledMaskBeyondEmbeddingRows) {
  MMDataItem image(MMType::IMAGE);
  auto& state = image.mutable_state();
  state.mutable_token_pos() = {0, 2};
  state.mutable_mm_token_num() = 1;
  state.mutable_mm_token_mask() = torch::tensor({true, true});
  state.mutable_seq_index() = 0;
  state.mutable_schedule_data().end_pos = 2;
  ASSERT_TRUE(image.add("image|embedding", torch::ones({1, 2})));

  EncoderEmbeddingGatherVisitor gather(
      /*num_computed_tokens=*/{0}, /*num_scheduled_tokens=*/{2});
  EXPECT_FALSE(gather.visit(image));
}

TEST(MMDataEncoderHandoffTest, GathersImagesAcrossSequenceWindows) {
  MMDataItem first(MMType::IMAGE);
  auto& first_state = first.mutable_state();
  first_state.mutable_token_pos() = {2, 2};
  first_state.mutable_mm_token_num() = 1;
  first_state.mutable_mm_token_mask() =
      torch::tensor({true, false}, torch::kBool);
  first_state.mutable_seq_index() = 0;
  first_state.mutable_schedule_data().end_pos = 2;
  ASSERT_TRUE(first.add("image|embedding", torch::tensor({{10.0, 11.0}})));

  MMDataItem second = first;
  auto& second_state = second.mutable_state();
  second_state.mutable_token_pos() = {4, 2};
  second_state.mutable_mm_token_mask() =
      torch::tensor({false, true}, torch::kBool);
  second_state.mutable_seq_index() = 1;
  second.mutable_data()["image|embedding"] = torch::tensor({{20.0, 21.0}});

  MMBatchData batch(
      std::vector<MMData>{MMData(MMType::IMAGE, MMItemVec{first}),
                          MMData(MMType::IMAGE, MMItemVec{second})});
  EncoderEmbeddingGatherVisitor gather(
      /*num_computed_tokens=*/{2, 3}, /*num_scheduled_tokens=*/{3, 4});
  ASSERT_TRUE(batch.foreach (gather));
  ASSERT_TRUE(gather.finish(batch));
  EXPECT_TRUE(torch::equal(batch.get<torch::Tensor>("image|embedding").value(),
                           torch::tensor({{10.0, 11.0}, {20.0, 21.0}})));
  EXPECT_TRUE(torch::equal(
      batch.get<torch::Tensor>("image|mask").value(),
      torch::tensor({true, false, false, false, false, true, false},
                    torch::kBool)));
}

MMData make_codec_media() {
  MMDataItem item(MMType::IMAGE);
  item.mutable_data()["pixel_values"] = torch::arange(6).reshape({2, 3});
  item.mutable_data()["grids"] = std::vector<torch::Tensor>{
      torch::tensor({1, 2}, torch::kInt64), torch::tensor({3}, torch::kInt32)};
  auto& state = item.mutable_state();
  state.mutable_token_pos() = {2, 5};
  state.mutable_mm_token_num() = 3;
  state.mutable_mm_token_mask() =
      torch::tensor({true, false, true, false, true}, torch::kBool);
  state.mutable_schedule_data().end_pos = 5;
  std::memset(
      state.mutable_schedule_data().key.data, 'a', XXH3_128BITS_HASH_VALUE_LEN);
  state.mutable_seq_index() = 19;
  return MMData(MMType::IMAGE, MMItemVec{item});
}

TEST(MMCodecTest, TensorsAndMetadata) {
  const MMData original = make_codec_media();
  proto::MMDataEntry encoded;
  ASSERT_TRUE(mmdata_entry_to_proto(original, &encoded));
  auto decoded = proto_to_mmdata_entry(encoded);
  ASSERT_TRUE(decoded.has_value());
  const auto& item = decoded->items<MMItemVec>().front();
  EXPECT_EQ(item.state().seq_index(), 19);
  EXPECT_EQ(item.state().mm_token_num(), 3);
  EXPECT_TRUE(torch::equal(*item.get<torch::Tensor>("pixel_values"),
                           torch::arange(6).reshape({2, 3})));
  const auto grids = item.get<std::vector<torch::Tensor>>("grids");
  ASSERT_TRUE(grids.has_value());
  ASSERT_EQ(grids->size(), 2u);
  EXPECT_EQ(grids->at(0).scalar_type(), torch::kInt64);
  EXPECT_EQ(grids->at(1).scalar_type(), torch::kInt32);
  EXPECT_TRUE(torch::equal(grids->at(1), torch::tensor({3}, torch::kInt32)));
  EXPECT_TRUE(torch::equal(
      item.state().mm_token_mask(),
      original.items<MMItemVec>().front().state().mm_token_mask()));
  ASSERT_TRUE(mmdata_entry_to_proto(original, &encoded, false));
  EXPECT_TRUE(encoded.items(0).dict().empty());
  EXPECT_FALSE(encoded.items(0).state().has_mm_token_mask());
  decoded = proto_to_mmdata_entry(encoded);
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->items<MMItemVec>().front().state().mm_token_num(), 3);
  MMDataItem absent(MMType::IMAGE);
  proto::MMDataItem reused = encoded.items(0);
  ASSERT_TRUE(mmdata_item_to_proto(absent, &reused, false));
  EXPECT_TRUE(reused.dict().empty());
  EXPECT_EQ(reused.state().mm_token_num(), 0);
}

TEST(MMCodecTest, DictAndItemOrder) {
  MMData first = make_codec_media();
  MMDataItem second = first.items<MMItemVec>().front();
  second.mutable_state().mutable_token_pos() = {7, 5};
  first.items<MMItemVec>().emplace_back(second);
  const MMData dict(MMType::AUDIO, MMDict{{"audio", torch::tensor({4, 5})}});
  MMBatchData batch(std::vector<MMData>{first, dict});
  proto::MMData wire;
  ASSERT_TRUE(mmdata_to_proto(batch, &wire));
  MMBatchData restored;
  ASSERT_TRUE(proto_to_mmdata(wire, &restored));
  ASSERT_EQ(restored.mm_data_vec().size(), 2u);
  const auto& items = restored.mm_data_vec().front().items<MMItemVec>();
  ASSERT_EQ(items.size(), 2u);
  EXPECT_TRUE(restored.mm_data_vec()[1].hold<MMDict>());
  EXPECT_TRUE(
      torch::equal(*restored.mm_data_vec()[1].get<torch::Tensor>("audio"),
                   torch::tensor({4, 5})));
}

TEST(MMCodecTest, SourceWireGolden) {
  // Frozen source mm_data_entry_test.cpp:PreservesExistingWireFields.
  const std::string wire = std::string(
                               "\x08\x01\x10\x01\x1a\x20\x08\x01\x1a\x1a"
                               "\x08\x02\x10\x05\x22\x10") +
                           std::string(16, 'a') + "\x30\x05\x38\x05\x20\x03";
  proto::MMDataEntry entry;
  ASSERT_TRUE(entry.ParseFromString(wire));
  auto decoded = proto_to_mmdata_entry(entry);
  ASSERT_TRUE(decoded.has_value());
  const auto& state = decoded->items<MMItemVec>().front().state();
  EXPECT_EQ(state.token_pos().offset, 2);
  EXPECT_EQ(state.token_pos().length, 5);
  EXPECT_EQ(state.mm_token_num(), 5);
  EXPECT_EQ(state.seq_index(), 3);
  proto::MMDataEntry encoded;
  ASSERT_TRUE(mmdata_entry_to_proto(*decoded, &encoded));
  EXPECT_EQ(encoded.SerializeAsString(), wire);
}

INSTANTIATE_TEST_SUITE_P(
    TypeMasks,
    MMDataTypeMaskTest,
    ::testing::Values(
        TypeMaskCase{MMType::NONE, MMType::NONE, false},
        TypeMaskCase{MMType::NONE, MMType::IMAGE, false},
        TypeMaskCase{MMType::NONE, MMType::VIDEO, false},
        TypeMaskCase{MMType::NONE, MMType::AUDIO, false},
        TypeMaskCase{MMType::NONE, MMType::EMBEDDING, false},
        TypeMaskCase{MMType::IMAGE, MMType::NONE, false},
        TypeMaskCase{MMType::IMAGE, MMType::IMAGE, true},
        TypeMaskCase{MMType::IMAGE, MMType::VIDEO, false},
        TypeMaskCase{MMType::VIDEO, MMType::VIDEO, true},
        TypeMaskCase{MMType::VIDEO, MMType::IMAGE, false},
        TypeMaskCase{MMType::AUDIO, MMType::AUDIO, true},
        TypeMaskCase{MMType::AUDIO, MMType::IMAGE, false},
        TypeMaskCase{MMType::EMBEDDING, MMType::EMBEDDING, true},
        TypeMaskCase{MMType::EMBEDDING, MMType::IMAGE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::NONE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::VIDEO, true},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING,
                     MMType::EMBEDDING,
                     true},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::IMAGE, false},
        TypeMaskCase{MMType::VIDEO | MMType::EMBEDDING, MMType::AUDIO, false},
        TypeMaskCase{MMType::VIDEO, MMType::VIDEO | MMType::AUDIO, true},
        TypeMaskCase{MMType::VIDEO, MMType::IMAGE | MMType::AUDIO, false},
        TypeMaskCase{MMType::IMAGE | MMType::AUDIO,
                     MMType::VIDEO | MMType::AUDIO,
                     true}));

}  // namespace
}  // namespace xllm
