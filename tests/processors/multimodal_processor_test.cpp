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

#include "processors/multimodal_processor.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "processors/glm4v_image_processor.h"
#include "processors/glm4v_video_processor.h"
#include "processors/kimi25_image_processor.h"
#include "processors/minicpmv_image_processor.h"
#include "processors/minicpmv_prompt_processor.h"
#include "processors/mistral3_image_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/qwen2_vl_video_processor.h"
#include "processors/qwen3_omni_video_processor.h"
#include "processors/qwen3_vl_video_processor.h"

namespace xllm {
namespace {

ModelArgs pixel_dtype_args() {
  ModelArgs args;
  args.dtype("float32")
      .mm_image_patch_size(14)
      .mm_image_merge_size(2)
      .mm_image_temporal_patch_size(2)
      .mm_image_min_pixels(3136)
      .mm_image_max_pixels(12544)
      .mm_image_normalize_mean({0.48145466, 0.4578275, 0.40821073})
      .mm_image_normalize_std({0.26862954, 0.26130258, 0.27577711})
      .mm_patch_size(14)
      .mm_spatial_merge_size(2)
      .mm_image_size(56)
      .vision_max_slice_nums(4)
      .mm_scale_resolution(56)
      .mm_slice_mode(true)
      .mm_image_feature_size(4)
      .mm_video_patch_size(14)
      .mm_video_merge_size(2)
      .mm_video_temporal_patch_size(2)
      .mm_video_shortest_edge(6272)
      .mm_video_longest_edge(25088)
      .mm_video_normalize_mean({0.48145466, 0.4578275, 0.40821073})
      .mm_video_normalize_std({0.26862954, 0.26130258, 0.27577711});
  return args;
}

void expect_pixel_item_matches(const MMDataItem& actual,
                               const MMDataItem& expected) {
  ASSERT_TRUE(expected.has("pixel_values") ||
              expected.has("pixel_values_videos"));
  ASSERT_EQ(actual.data().size(), expected.data().size());
  for (const auto& [key, value] : expected.data()) {
    SCOPED_TRACE(key);
    ASSERT_TRUE(actual.has(key));
    const bool is_pixel = key == "pixel_values" || key == "pixel_values_videos";
    std::vector<torch::Tensor> expected_tensors;
    std::vector<torch::Tensor> actual_tensors;
    expected.get(key, expected_tensors);
    actual.get(key, actual_tensors);
    if (is_pixel) {
      ASSERT_FALSE(expected_tensors.empty());
    }
    ASSERT_EQ(actual_tensors.size(), expected_tensors.size());
    for (size_t index = 0; index < expected_tensors.size(); ++index) {
      const torch::Tensor reference =
          is_pixel ? expected_tensors[index].to(torch::kBFloat16)
                   : expected_tensors[index];
      EXPECT_EQ(actual_tensors[index].scalar_type(), reference.scalar_type());
      EXPECT_TRUE(torch::equal(actual_tensors[index], reference));
    }
  }
}

template <typename Processor>
class ImagePixelDtypeTest : public ::testing::Test {};

using ImagePixelProcessors = ::testing::Types<Qwen2VLImageProcessor,
                                              Glm4VImageProcessor,
                                              Mistral3ImageProcessor,
                                              MiniCPMVImageProcessor,
                                              KimiK25ImageProcessor>;
TYPED_TEST_SUITE(ImagePixelDtypeTest, ImagePixelProcessors);

TYPED_TEST(ImagePixelDtypeTest, MatchesFinalFloat32CastAndPreservesMetadata) {
  ModelArgs args = pixel_dtype_args();
  const TypeParam baseline(args);
  const torch::Tensor image = torch::arange(3 * 56 * 84, torch::kInt64)
                                  .remainder(256)
                                  .to(torch::kUInt8)
                                  .reshape({3, 56, 84});
  const std::vector<torch::Tensor> images{image, image.flip({2})};
  std::vector<MMDataItem> expected_items;
  ASSERT_TRUE(baseline.process(images, expected_items));
  ASSERT_EQ(expected_items.size(), images.size());
  args.dtype("bfloat16");
  const TypeParam processor(args);
  std::vector<MMDataItem> actual_items;
  ASSERT_TRUE(processor.process(images, actual_items));
  ASSERT_EQ(actual_items.size(), expected_items.size());
  for (size_t index = 0; index < actual_items.size(); ++index) {
    expect_pixel_item_matches(actual_items[index], expected_items[index]);
  }
}

template <typename Processor>
class VideoPixelDtypeTest : public ::testing::Test {};

using VideoPixelProcessors = ::testing::Types<Qwen2VLVideoProcessor,
                                              Qwen3VLVideoProcessor,
                                              Qwen3OmniVideoProcessor,
                                              Glm4VVideoProcessor,
                                              KimiK25VideoProcessor>;
TYPED_TEST_SUITE(VideoPixelDtypeTest, VideoPixelProcessors);

TYPED_TEST(VideoPixelDtypeTest, MatchesFinalFloat32CastAndPreservesMetadata) {
  ModelArgs args = pixel_dtype_args();
  const TypeParam baseline(args);
  const torch::Tensor video = torch::arange(8 * 3 * 56 * 84, torch::kInt64)
                                  .remainder(256)
                                  .to(torch::kUInt8)
                                  .reshape({8, 3, 56, 84});
  VideoMetadata metadata;
  metadata.fps = 2.0;
  metadata.total_num_frames = 8;
  metadata.duration = 4.0;
  metadata.frame_indices = torch::arange(8, torch::kInt64);
  MMDataItem expected(MMType::VIDEO);
  ASSERT_TRUE(baseline.process(video, metadata, expected));
  args.dtype("bfloat16");
  const TypeParam processor(args);
  MMDataItem actual(MMType::VIDEO);
  ASSERT_TRUE(processor.process(video, metadata, actual));
  expect_pixel_item_matches(actual, expected);
  const auto expected_metadata =
      expected.template get_metadata<VideoMetadata>();
  const auto actual_metadata = actual.template get_metadata<VideoMetadata>();
  ASSERT_TRUE(expected_metadata.has_value());
  ASSERT_TRUE(actual_metadata.has_value());
  EXPECT_EQ(actual_metadata->timestamps, expected_metadata->timestamps);
  EXPECT_EQ(actual_metadata->sampled_fps, expected_metadata->sampled_fps);
  EXPECT_TRUE(torch::equal(actual_metadata->frame_indices,
                           expected_metadata->frame_indices));
}

TEST(MultimodalProcessorTest, LegacyProcessorDoesNotConsumeOverrides) {
  const ModelArgs args{};
  const ProcessorConfig defaults;
  const auto processor = create_processor<ImageNoneProcessor>(args, defaults);
  defaults.validate("image");
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"max_pixels", 4096}}}});
  const auto configured = create_processor<ImageNoneProcessor>(args, config);
  EXPECT_DEATH(config.validate("image"), "image.max_pixels");
}

TEST(MultimodalProcessorTest, MistralUsesItsOwnEdgeParameter) {
  ModelArgs args;
  args.mm_patch_size(14).mm_spatial_merge_size(2).mm_image_size(112);
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"longest_edge", 56}}}});
  const auto processor = create_processor<Mistral3ImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 112, 112}, torch::kUInt8)}, items));
  ASSERT_EQ(items.size(), 1);
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 4, 4}}, torch::kInt64)));
  EXPECT_EQ(args.mm_image_size(), 112);
}

TEST(MultimodalProcessorTest, KimiUsesItsOwnPatchBudgetParameter) {
  ModelArgs args;
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"in_patch_limit", 16}}}});
  const auto processor = create_processor<KimiK25ImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 112, 112}, torch::kUInt8)}, items));
  ASSERT_EQ(items.size(), 1);
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 4, 4}}, torch::kInt64)));
}

TEST(MultimodalProcessorTest, UnknownParametersAreRejectedByActualConsumer) {
  ModelArgs args;
  args.mm_patch_size(14).mm_spatial_merge_size(2).mm_image_size(112);
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"max_pixels", 4096}}}}, "mm_args");
  const auto processor = create_processor<Mistral3ImageProcessor>(args, config);
  EXPECT_DEATH(config.validate(),
               "Unsupported processor parameter.*max_pixels");
}

TEST(MultimodalProcessorTest, VideoOverridesDoNotEnableImagePixelValidation) {
  ModelArgs args;
  args.mm_image_min_tokens(4).mm_image_max_tokens(16);
  const ProcessorConfig config(
      nlohmann::json{{"video", {{"frame_stride", 8}}}});
  const auto processor = create_processor<Qwen2VLImageProcessor>(args, config);
  config.validate("image");
  EXPECT_DEATH(config.validate("video"), "video.frame_stride");
}

TEST(MultimodalProcessorTest, GlmPreservesItsTemporalAreaSemantics) {
  ModelArgs args;
  args.mm_image_patch_size(14)
      .mm_image_merge_size(2)
      .mm_image_temporal_patch_size(2)
      .mm_image_min_pixels(3136)
      .mm_image_max_pixels(12845056)
      .mm_image_normalize_mean({0.5, 0.5, 0.5})
      .mm_image_normalize_std({0.5, 0.5, 0.5});
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"max_pixels", 12544}}}});
  const auto processor = create_processor<Glm4VImageProcessor>(args, config);
  const auto qwen_processor =
      create_processor<Qwen2VLImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 112, 112}, torch::kUInt8)}, items));
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 4, 4}}, torch::kInt64)));
  std::vector<MMDataItem> qwen_items;
  ASSERT_TRUE(qwen_processor->process(
      {torch::zeros({3, 112, 112}, torch::kUInt8)}, qwen_items));
  const auto qwen_grid = qwen_items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(qwen_grid.has_value());
  EXPECT_TRUE(
      torch::equal(*qwen_grid, torch::tensor({{1, 8, 8}}, torch::kInt64)));
}

TEST(MultimodalProcessorTest, PixelBudgetsPreserveProcessorMinimumGridSizes) {
  ModelArgs args;
  args.mm_image_patch_size(14)
      .mm_image_merge_size(2)
      .mm_image_temporal_patch_size(2)
      .mm_image_normalize_mean({0.5, 0.5, 0.5})
      .mm_image_normalize_std({0.5, 0.5, 0.5});
  const ProcessorConfig config(
      nlohmann::json{{"image", {{"min_pixels", 784}, {"max_pixels", 784}}}});
  const auto processor = create_processor<Qwen2VLImageProcessor>(args, config);
  config.validate();
  std::vector<MMDataItem> items;
  ASSERT_TRUE(
      processor->process({torch::zeros({3, 28, 28}, torch::kUInt8)}, items));
  const auto grid = items[0].get<torch::Tensor>("image_grid_thw");
  ASSERT_TRUE(grid.has_value());
  EXPECT_TRUE(torch::equal(*grid, torch::tensor({{1, 2, 2}}, torch::kInt64)));
  EXPECT_DEATH(create_processor<Glm4VImageProcessor>(args, config),
               "one aligned image patch group");
}

TEST(MultimodalProcessorTest, MiniCpmSlicesAndPromptShareConfiguration) {
  ModelArgs args;
  args.vision_max_slice_nums(9)
      .mm_scale_resolution(112)
      .mm_patch_size(14)
      .mm_slice_mode(true)
      .mm_image_feature_size(4)
      .mm_image_normalize_mean({0.5, 0.5, 0.5})
      .mm_image_normalize_std({0.5, 0.5, 0.5});
  const auto image = torch::zeros({3, 112, 112}, torch::kUInt8);
  for (int32_t max_slices : {1, 4}) {
    const ProcessorConfig config(nlohmann::json{
        {"image", {{"max_slice_nums", max_slices}, {"scale_resolution", 56}}}});
    const auto processor =
        create_processor<MiniCPMVImageProcessor>(args, config);
    const auto prompt_processor =
        create_processor<npu::model::MiniCPMPromptProcessor>(args, config);
    config.validate();
    std::vector<MMDataItem> items;
    ASSERT_TRUE(processor->process({image}, items));
    ASSERT_EQ(items.size(), 1);
    const auto pixels =
        items[0].get<std::vector<torch::Tensor>>("pixel_values");
    ASSERT_TRUE(pixels.has_value());
    EXPECT_EQ(pixels->size(), max_slices == 1 ? 1 : 5);
    MMData data(MMType::IMAGE, items);
    std::string prompt = "(<image>./</image>)";
    ASSERT_TRUE(prompt_processor->process(prompt, data));
    size_t unknown_count = 0;
    size_t offset = 0;
    while ((offset = prompt.find("<unk>", offset)) != std::string::npos) {
      ++unknown_count;
      offset += 5;
    }
    EXPECT_EQ(unknown_count, pixels->size() * 4);
  }
}

}  // namespace
}  // namespace xllm
