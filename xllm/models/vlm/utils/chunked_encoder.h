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

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace xllm {
namespace detail {

struct ViTAttentionChunk {
  std::vector<std::array<int64_t, 3>> grid_thw;
  int64_t input_begin = 0;
  int64_t input_end = 0;
};

struct ViTAttentionChunkPlan {
  bool requires_chunking = false;
  std::vector<ViTAttentionChunk> chunks;
};

inline ViTAttentionChunkPlan plan_vit_attention_chunks(
    const std::vector<std::array<int64_t, 3>>& grid_thw,
    int64_t max_units_per_chunk) {
  ViTAttentionChunkPlan plan;
  int64_t total_units = 0;
  for (const auto& grid : grid_thw) {
    total_units += std::max<int64_t>(grid[0], 0);
  }
  if (total_units == 0) {
    return plan;
  }
  const int64_t chunk_budget = max_units_per_chunk;
  const int64_t chunk_count = 1 + (total_units - 1) / chunk_budget;
  plan.chunks.reserve(chunk_count);
  int64_t input_offset = 0;
  int64_t chunk_units = 0;
  for (const auto& grid : grid_thw) {
    int64_t remaining_units = std::max<int64_t>(grid[0], 0);
    const int64_t spatial_height = grid[1];
    const int64_t spatial_width = grid[2];
    const int64_t inputs_per_attention_unit = spatial_height * spatial_width;
    while (remaining_units > 0) {
      if (plan.chunks.empty() || chunk_units == chunk_budget) {
        plan.chunks.emplace_back(
            ViTAttentionChunk{{}, input_offset, input_offset});
        plan.chunks.back().grid_thw.reserve(
            std::min(static_cast<int64_t>(grid_thw.size()), chunk_budget));
        chunk_units = 0;
      }
      const int64_t units =
          std::min(remaining_units, chunk_budget - chunk_units);
      auto& chunk = plan.chunks.back();
      chunk.grid_thw.emplace_back(
          std::array<int64_t, 3>{units, spatial_height, spatial_width});
      input_offset += units * inputs_per_attention_unit;
      chunk.input_end = input_offset;
      chunk_units += units;
      remaining_units -= units;
    }
  }
  plan.requires_chunking = plan.chunks.size() > 1;
  return plan;
}

}  // namespace detail

template <typename EncodeFunction>
torch::Tensor run_chunked_encoder(const torch::Tensor& pixel_values,
                                  const torch::Tensor& grid_thw,
                                  int64_t max_units_per_chunk,
                                  EncodeFunction&& encode) {
  if (max_units_per_chunk <= 0 || grid_thw.size(0) == 0) {
    return encode(pixel_values, grid_thw);
  }

  const auto grid_cpu = grid_thw.to(torch::kCPU, torch::kInt64).contiguous();
  const auto grid_accessor = grid_cpu.accessor<int64_t, 2>();
  std::vector<std::array<int64_t, 3>> grid_values;
  grid_values.reserve(grid_cpu.size(0));
  for (int64_t index = 0; index < grid_cpu.size(0); ++index) {
    grid_values.emplace_back(std::array<int64_t, 3>{grid_accessor[index][0],
                                                    grid_accessor[index][1],
                                                    grid_accessor[index][2]});
  }
  const auto plan =
      detail::plan_vit_attention_chunks(grid_values, max_units_per_chunk);
  if (!plan.requires_chunking) {
    return encode(pixel_values, grid_thw);
  }
  CHECK_EQ(pixel_values.size(0), plan.chunks.back().input_end)
      << "Pixel rows must match the patch count described by grid_thw.";

  size_t grid_count = 0;
  for (const auto& chunk : plan.chunks) {
    grid_count += chunk.grid_thw.size();
  }
  std::vector<int64_t> flat_grids;
  flat_grids.reserve(grid_count * 3);
  for (const auto& chunk : plan.chunks) {
    for (const auto& grid : chunk.grid_thw) {
      flat_grids.insert(flat_grids.end(), grid.begin(), grid.end());
    }
  }
  const auto chunk_grids =
      torch::tensor(flat_grids, grid_thw.options()).reshape({-1, 3});
  std::vector<torch::Tensor> outputs;
  outputs.reserve(plan.chunks.size());
  int64_t grid_offset = 0;
  for (const auto& chunk : plan.chunks) {
    const int64_t chunk_grid_count =
        static_cast<int64_t>(chunk.grid_thw.size());
    outputs.emplace_back(encode(
        pixel_values.slice(/*dim=*/0, chunk.input_begin, chunk.input_end),
        chunk_grids.narrow(/*dim=*/0, grid_offset, chunk_grid_count)));
    grid_offset += chunk_grid_count;
  }
  return torch::cat(outputs, /*dim=*/0);
}

}  // namespace xllm
