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

#include "core/framework/multimodal/mm_codec.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {
namespace {

std::string make_mjpeg_video(const std::vector<cv::Size>& frame_sizes) {
  std::string video;
  for (const auto& size : frame_sizes) {
    const cv::Mat frame(size, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<uint8_t> encoded;
    EXPECT_TRUE(cv::imencode(".jpg", frame, encoded));
    video.append(reinterpret_cast<const char*>(encoded.data()), encoded.size());
  }
  return video;
}

TEST(FFmpegVideoDecoderTest, DecodesFramesWithMatchingShapes) {
  const std::string video = make_mjpeg_video({{16, 16}, {16, 16}});
  FFmpegVideoDecoder decoder;
  torch::Tensor frames;
  VideoMetadata metadata;

  ASSERT_TRUE(decoder.decode(video, frames, metadata));
  EXPECT_EQ(frames.sizes(), (std::vector<int64_t>{2, 3, 16, 16}));
  EXPECT_EQ(metadata.total_num_frames, 2);
}

TEST(FFmpegVideoDecoderTest, RejectsFramesWithDifferentShapes) {
  const std::string video = make_mjpeg_video({{16, 16}, {32, 16}});
  FFmpegVideoDecoder decoder;
  torch::Tensor frames;
  VideoMetadata metadata;

  EXPECT_FALSE(decoder.decode(video, frames, metadata));
  EXPECT_FALSE(frames.defined());
  EXPECT_EQ(metadata.total_num_frames, 0);
}

}  // namespace
}  // namespace xllm
