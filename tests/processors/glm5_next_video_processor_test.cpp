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

#include <string>
#include <vector>

#include "processors/glm4v_prompt_processor.h"
#include "processors/glm4v_video_processor.h"

namespace xllm {
namespace {

ModelArgs video_args(const std::string& model_type = "glm5_next") {
  ModelArgs args;
  args.model_type(model_type)
      .mm_video_patch_size(14)
      .mm_video_temporal_patch_size(2)
      .mm_video_merge_size(2)
      .mm_image_merge_size(2)
      .mm_video_normalize_mean({0.5, 0.5, 0.5})
      .mm_video_normalize_std({0.5, 0.5, 0.5})
      .mm_video_min_tokens(1)
      .mm_video_max_tokens(240000);
  return args;
}

class Glm5NextVideoAliasTest : public ::testing::TestWithParam<std::string> {};

TEST_P(Glm5NextVideoAliasTest, SamplesVideoAndExpandsWholeSecondTimestamps) {
  const ModelArgs args = video_args(GetParam());
  const Glm4VVideoProcessor processor(
      args, Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 3.0;
  metadata.total_num_frames = 36;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({36, 3, 28, 28}, torch::kUInt8), metadata, item));
  const auto sampled = item.get_metadata<VideoMetadata>();
  ASSERT_TRUE(sampled.has_value());
  EXPECT_TRUE(
      torch::equal(sampled->frame_indices,
                   torch::tensor({0, 7, 14, 21, 28, 35}, torch::kLong)));
  EXPECT_TRUE(torch::equal(*item.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{3, 2, 2}}, torch::kLong)));
  EXPECT_EQ(item.get<torch::Tensor>("pixel_values_videos")->size(0), 12);
  EXPECT_EQ(item.get<torch::Tensor>("pixel_values_videos")->size(1), 1176);

  MMData data;
  data.add(MMType::VIDEO, item);
  GLM4VPromptProcessor prompt_processor(args);
  std::string prompt = "<|begin_of_video|><|video|><|end_of_video|>";
  ASSERT_TRUE(prompt_processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "<|begin_of_video|>"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|end_of_video|>");
}

INSTANTIATE_TEST_SUITE_P(ModelTypes,
                         Glm5NextVideoAliasTest,
                         ::testing::Values("glm5_next", "glm5_3_flash"));

TEST(Glm5NextVideoProcessorTest, HandlesSubsecondVideoWithOneSourceFrame) {
  const Glm4VVideoProcessor processor(
      video_args(), Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 1.0 / 12.0;
  metadata.total_num_frames = 1;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({1, 3, 28, 28}, torch::kUInt8), metadata, item));
  EXPECT_TRUE(torch::equal(item.get_metadata<VideoMetadata>()->frame_indices,
                           torch::tensor({0, 0}, torch::kLong)));
  EXPECT_TRUE(torch::equal(*item.get<torch::Tensor>("video_grid_thw"),
                           torch::tensor({{1, 2, 2}}, torch::kLong)));
}

TEST(Glm5NextVideoProcessorTest, HonorsConfiguredSamplingRate) {
  ModelArgs args = video_args();
  args.mm_fps(1.0);
  const Glm4VVideoProcessor processor(
      args, Glm4VVideoProcessor::SamplingMode::FIXED_FPS);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 3.0;
  metadata.total_num_frames = 36;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({36, 3, 28, 28}, torch::kUInt8), metadata, item));
  EXPECT_TRUE(torch::equal(item.get_metadata<VideoMetadata>()->frame_indices,
                           torch::tensor({0, 17, 35, 35}, torch::kLong)));
}

TEST(Glm5NextVideoProcessorTest,
     KeepsDefaultDynamicSamplingAndFloorsTimestamps) {
  const ModelArgs args = video_args("glm4v");
  const Glm4VVideoProcessor processor(args);
  VideoMetadata metadata;
  metadata.fps = 12.0;
  metadata.duration = 3.0;
  metadata.total_num_frames = 36;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({36, 3, 28, 28}, torch::kUInt8), metadata, item));
  EXPECT_EQ(item.get_metadata<VideoMetadata>()->frame_indices.numel(), 18);
  EXPECT_EQ((*item.get<torch::Tensor>("video_grid_thw"))[0][0].item<int64_t>(),
            9);

  MMData data;
  data.add(MMType::VIDEO, item);
  GLM4VPromptProcessor prompt_processor(args);
  std::string prompt = "<|begin_of_video|><|video|><|end_of_video|>";
  ASSERT_TRUE(prompt_processor.process(prompt, data));
  EXPECT_EQ(prompt,
            "<|begin_of_video|>"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>0.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>1.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|begin_of_image|><|image|><|end_of_image|>2.0 seconds"
            "<|end_of_video|>");
}

TEST(Glm5NextVideoProcessorTest, KeepsDefaultDynamicFrameCap) {
  const Glm4VVideoProcessor processor(video_args("glm4v"));
  VideoMetadata metadata;
  metadata.fps = 1.0;
  metadata.duration = 3000.0;
  metadata.total_num_frames = 3000;
  MMDataItem item(MMType::NONE);
  ASSERT_TRUE(processor.process(
      torch::zeros({3000, 3, 28, 28}, torch::kUInt8), metadata, item));
  const auto sampled = item.get_metadata<VideoMetadata>();
  ASSERT_TRUE(sampled.has_value());
  EXPECT_EQ(sampled->frame_indices.numel(), 640);
  EXPECT_EQ(sampled->frame_indices[0].item<int64_t>(), 0);
  EXPECT_EQ(sampled->frame_indices[-1].item<int64_t>(), 2999);
}

}  // namespace
}  // namespace xllm
