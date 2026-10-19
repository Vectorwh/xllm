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

#pragma once

#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace xllm {

class JoyImageQwen3ImageProcessor final {
 public:
  static torch::Tensor preprocess(const torch::Tensor& image) {
    CHECK(image.dim() == 3 || image.dim() == 4)
        << "JoyImage Qwen3-VL image must be CHW or BCHW";
    torch::Tensor input = to_uint8(image);
    const bool squeeze_batch = input.dim() == 3;
    if (squeeze_batch) {
      input = input.unsqueeze(0);
    }

    const auto [target_height, target_width] =
        select_bucket(input.size(2), input.size(3));
    const double scale =
        std::max(static_cast<double>(target_height) / input.size(2),
                 static_cast<double>(target_width) / input.size(3));
    const int64_t resize_height =
        static_cast<int64_t>(std::ceil(input.size(2) * scale));
    const int64_t resize_width =
        static_cast<int64_t>(std::ceil(input.size(3) * scale));
    torch::Tensor output = resize_uint8_bilinear(
        input, resize_height, resize_width, target_height, target_width);
    return squeeze_batch ? output.squeeze(0) : output;
  }

  static std::pair<int64_t, int64_t> select_bucket(int64_t height,
                                                   int64_t width) {
    static const std::vector<std::pair<int64_t, int64_t>> kBuckets = {
        {512, 1792},  {512, 1856}, {512, 1920}, {512, 1984}, {512, 2048},
        {576, 1600},  {576, 1664}, {576, 1728}, {576, 1792}, {640, 1472},
        {640, 1536},  {640, 1600}, {704, 1344}, {704, 1408}, {704, 1472},
        {768, 1216},  {768, 1280}, {768, 1344}, {832, 1152}, {832, 1216},
        {896, 1088},  {896, 1152}, {960, 1024}, {960, 1088}, {1024, 960},
        {1024, 1024}, {1088, 896}, {1088, 960}, {1152, 832}, {1152, 896},
        {1216, 768},  {1216, 832}, {1280, 768}, {1344, 704}, {1344, 768},
        {1408, 704},  {1472, 640}, {1472, 704}, {1536, 640}, {1600, 576},
        {1600, 640},  {1664, 576}, {1728, 576}, {1792, 512}, {1792, 576},
        {1856, 512},  {1920, 512}, {1984, 512}, {2048, 512}};
    const double target_ratio =
        static_cast<double>(height) / static_cast<double>(width);
    std::pair<int64_t, int64_t> selected = {1024, 1024};
    double minimum_difference = std::numeric_limits<double>::max();
    for (const auto& bucket : kBuckets) {
      const double difference = std::abs(
          static_cast<double>(bucket.first) / bucket.second - target_ratio);
      if (difference < minimum_difference) {
        minimum_difference = difference;
        selected = bucket;
      }
    }
    return selected;
  }

 private:
  using Coefficients = std::vector<int32_t>;
  using CoefficientTable = std::vector<Coefficients>;

  static torch::Tensor to_uint8(const torch::Tensor& image) {
    CHECK_GT(image.numel(), 0) << "JoyImage image must not be empty";
    torch::Tensor input = image.to(torch::kCPU);
    if (input.scalar_type() == torch::kUInt8) {
      return input.contiguous();
    }
    input = input.to(torch::kFloat32);
    const float normalized_max = image.is_floating_point() ? 1.1f : 1.0f;
    if (input.max().item<float>() <= normalized_max) {
      input = input.clamp(0.0f, 1.0f) * 255.0f;
    }
    return input.round().clamp(0.0f, 255.0f).to(torch::kUInt8).contiguous();
  }

  static void build_coefficients(int64_t input_size,
                                 int64_t output_size,
                                 int64_t output_start,
                                 int64_t output_count,
                                 CoefficientTable* coefficient_table,
                                 std::vector<int64_t>* bounds) {
    constexpr int64_t kPrecisionBits = 22;
    const double scale = static_cast<double>(input_size) / output_size;
    const double filter_scale = std::max(scale, 1.0);
    const double support = filter_scale;
    const int64_t precision_scale = 1LL << kPrecisionBits;
    coefficient_table->reserve(static_cast<size_t>(output_count));
    bounds->reserve(static_cast<size_t>(output_count));

    for (int64_t output_index = output_start;
         output_index < output_start + output_count;
         ++output_index) {
      const double center = (output_index + 0.5) * scale;
      const int64_t first =
          std::max<int64_t>(0, static_cast<int64_t>(center - support + 0.5));
      const int64_t last = std::min<int64_t>(
          input_size, static_cast<int64_t>(center + support + 0.5));
      const int64_t sample_count = last - first;

      std::vector<double> weights;
      weights.reserve(static_cast<size_t>(sample_count));
      double weight_sum = 0.0;
      for (int64_t sample_index = 0; sample_index < sample_count;
           ++sample_index) {
        const double distance =
            std::abs((sample_index + first - center + 0.5) / filter_scale);
        const double weight = distance < 1.0 ? 1.0 - distance : 0.0;
        weights.emplace_back(weight);
        weight_sum += weight;
      }

      Coefficients coefficients(static_cast<size_t>(sample_count));
      for (size_t sample_index = 0; sample_index < weights.size();
           ++sample_index) {
        coefficients[sample_index] = static_cast<int32_t>(
            0.5 + weights[sample_index] / weight_sum * precision_scale);
      }
      coefficient_table->emplace_back(std::move(coefficients));
      bounds->emplace_back(first);
    }
  }

  static torch::Tensor resize_uint8_bilinear(const torch::Tensor& image,
                                             int64_t resize_height,
                                             int64_t resize_width,
                                             int64_t height,
                                             int64_t width) {
    CHECK_EQ(image.dim(), 4);
    CHECK_EQ(image.scalar_type(), torch::kUInt8);
    CHECK_GT(height, 0);
    CHECK_GT(width, 0);
    constexpr int64_t kPrecisionBits = 22;
    const int64_t input_height = image.size(2);
    const int64_t input_width = image.size(3);
    const torch::TensorOptions cpu_options =
        image.options().device(torch::kCPU);
    CoefficientTable horizontal_coefficients;
    CoefficientTable vertical_coefficients;
    std::vector<int64_t> horizontal_bounds;
    std::vector<int64_t> vertical_bounds;
    const int64_t top = (resize_height - height) / 2;
    const int64_t left = (resize_width - width) / 2;
    build_coefficients(input_width,
                       resize_width,
                       left,
                       width,
                       &horizontal_coefficients,
                       &horizontal_bounds);
    build_coefficients(input_height,
                       resize_height,
                       top,
                       height,
                       &vertical_coefficients,
                       &vertical_bounds);
    const int64_t first_input_row = vertical_bounds.front();
    const int64_t input_row_count =
        vertical_bounds.back() +
        static_cast<int64_t>(vertical_coefficients.back().size()) -
        first_input_row;
    torch::Tensor horizontal = torch::empty(
        {image.size(0), image.size(1), input_row_count, width}, cpu_options);
    torch::Tensor output = torch::empty(
        {image.size(0), image.size(1), height, width}, cpu_options);

    const uint8_t* input_data = image.data_ptr<uint8_t>();
    uint8_t* horizontal_data = horizontal.data_ptr<uint8_t>();
    uint8_t* output_data = output.data_ptr<uint8_t>();
    const int64_t channels = image.size(1);
    const int64_t input_plane = input_height * input_width;
    const int64_t horizontal_plane = input_row_count * width;
    const int64_t output_plane = height * width;
    const int64_t rounding = 1LL << (kPrecisionBits - 1);

    const int64_t planes = image.size(0) * channels;
    torch::parallel_for(
        0, planes * input_row_count, 0, [&](int64_t begin, int64_t end) {
          for (int64_t row = begin; row < end; ++row) {
            const int64_t plane = row / input_row_count;
            const int64_t input_y = row % input_row_count;
            const uint8_t* input_plane_data = input_data + plane * input_plane;
            uint8_t* horizontal_plane_data =
                horizontal_data + plane * horizontal_plane;

            for (int64_t output_x = 0; output_x < width; ++output_x) {
              const Coefficients& coefficients =
                  horizontal_coefficients[static_cast<size_t>(output_x)];
              const int64_t first =
                  horizontal_bounds[static_cast<size_t>(output_x)];
              int64_t value = rounding;
              for (size_t index = 0; index < coefficients.size(); ++index) {
                value +=
                    static_cast<int64_t>(
                        input_plane_data[(input_y + first_input_row) *
                                             input_width +
                                         first + static_cast<int64_t>(index)]) *
                    coefficients[index];
              }
              value >>= kPrecisionBits;
              horizontal_plane_data[input_y * width + output_x] =
                  static_cast<uint8_t>(
                      std::max<int64_t>(0, std::min<int64_t>(255, value)));
            }
          }
        });

    torch::parallel_for(0, planes * height, 0, [&](int64_t begin, int64_t end) {
      for (int64_t row = begin; row < end; ++row) {
        const int64_t plane = row / height;
        const int64_t output_y = row % height;
        const uint8_t* horizontal_plane_data =
            horizontal_data + plane * horizontal_plane;
        uint8_t* output_plane_data = output_data + plane * output_plane;
        const Coefficients& coefficients =
            vertical_coefficients[static_cast<size_t>(output_y)];
        const int64_t first =
            vertical_bounds[static_cast<size_t>(output_y)] - first_input_row;
        for (int64_t output_x = 0; output_x < width; ++output_x) {
          int64_t value = rounding;
          for (size_t index = 0; index < coefficients.size(); ++index) {
            value += static_cast<int64_t>(
                         horizontal_plane_data
                             [(first + static_cast<int64_t>(index)) * width +
                              output_x]) *
                     coefficients[index];
          }
          value >>= kPrecisionBits;
          output_plane_data[output_y * width + output_x] = static_cast<uint8_t>(
              std::max<int64_t>(0, std::min<int64_t>(255, value)));
        }
      }
    });
    return output;
  }
};

}  // namespace xllm
