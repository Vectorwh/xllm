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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace xllm::minimax_h3_video {

// Execution limit, not a model parameter. Set before use; all VAE ranks
// must use the same value.
struct VideoDecodeOptions {
  int64_t max_window_clips;
};

// Two half-open ranges on the decoded tile's T axis: [begin, begin + count).
struct FrameSelection {
  int64_t first_begin = 0;
  int64_t first_count = 0;
  int64_t second_begin = 0;
  int64_t second_count = 0;
};

// Derived plan. Capacity includes padding slots and is equal on every rank.
struct TileWindow {
  int64_t clip_count = 0;
  int64_t local_tile_capacity = 0;
};

inline FrameSelection retained_frames(int64_t frame_count,
                                      int64_t section_frame_count,
                                      int64_t pre_padding) {
  CHECK_GE(frame_count, 0);
  CHECK_GT(section_frame_count, 0);
  CHECK_GE(pre_padding, 0);
  CHECK_LE(section_frame_count, std::numeric_limits<int64_t>::max() / 2);
  const int64_t first_end = std::min(frame_count, section_frame_count);
  const int64_t second_end = std::min(frame_count, 2 * section_frame_count);
  const int64_t first_begin = std::min(pre_padding, first_end);
  const int64_t second_span =
      second_end - std::min(section_frame_count, second_end);
  const int64_t second_count = std::max<int64_t>(0, second_span - pre_padding);
  const int64_t second_begin = second_end - second_count;
  return FrameSelection{
      .first_begin = first_begin,
      .first_count = first_end - first_begin,
      .second_begin = second_begin,
      .second_count = second_count,
  };
}

inline int64_t rank_major_tile_capacity(int64_t task_count,
                                        int64_t world_size) {
  CHECK_GT(task_count, 0);
  CHECK_GT(world_size, 0);
  return task_count / world_size + (task_count % world_size != 0 ? 1 : 0);
}

inline TileWindow plan_tile_window(int64_t remaining_clip_count,
                                   int64_t tiles_per_clip,
                                   int64_t world_size,
                                   const VideoDecodeOptions& options) {
  CHECK_GT(remaining_clip_count, 0);
  CHECK_GT(tiles_per_clip, 0);
  CHECK_GT(world_size, 0);
  CHECK_GT(options.max_window_clips, 0);

  const int64_t window_clip_count =
      std::min(remaining_clip_count, options.max_window_clips);
  CHECK_LE(window_clip_count,
           std::numeric_limits<int64_t>::max() / tiles_per_clip);
  const int64_t task_count = window_clip_count * tiles_per_clip;
  return TileWindow{
      .clip_count = window_clip_count,
      .local_tile_capacity = rank_major_tile_capacity(task_count, world_size),
  };
}

inline void copy_retained_frames(const torch::Tensor& source,
                                 const torch::Tensor& output,
                                 const FrameSelection& selection) {
  CHECK_EQ(source.dim(), 5) << "Expected [B,C,T,H,W] tiles";
  CHECK_EQ(output.dim(), 5);
  CHECK(source.device() == output.device());
  CHECK(source.scalar_type() == output.scalar_type());
  CHECK(!source.is_alias_of(output));
  CHECK_GE(selection.first_begin, 0);
  CHECK_GE(selection.first_count, 0);
  CHECK_GE(selection.second_begin, 0);
  CHECK_GE(selection.second_count, 0);
  CHECK_LE(selection.first_begin, source.size(/*dim=*/2));
  CHECK_LE(selection.first_count,
           source.size(/*dim=*/2) - selection.first_begin);
  CHECK_LE(selection.second_begin, source.size(/*dim=*/2));
  CHECK_LE(selection.second_count,
           source.size(/*dim=*/2) - selection.second_begin);
  CHECK_EQ(output.size(/*dim=*/2),
           selection.first_count + selection.second_count);
  for (int64_t dimension : {0, 1, 3, 4}) {
    CHECK_EQ(source.size(dimension), output.size(dimension));
  }
  if (selection.first_count > 0) {
    output.narrow(/*dim=*/2, /*start=*/0, selection.first_count)
        .copy_(source.narrow(
            /*dim=*/2, selection.first_begin, selection.first_count));
  }
  if (selection.second_count > 0) {
    output.narrow(/*dim=*/2, selection.first_count, selection.second_count)
        .copy_(source.narrow(
            /*dim=*/2, selection.second_begin, selection.second_count));
  }
}

// Shared by full encoder/decoder tiles and frame-packed decoder tiles.
// select_frames chooses the T ranges once; every rank must have a task.
template <typename ProduceTile, typename SelectFrames, typename GatherTiles>
torch::Tensor gather_rank_major_tiles(int64_t task_count,
                                      int64_t local_tile_capacity,
                                      int64_t world_size,
                                      int64_t rank,
                                      const ProduceTile& produce_tile,
                                      const SelectFrames& select_frames,
                                      const GatherTiles& gather_tiles) {
  CHECK_GE(task_count, world_size);
  CHECK_EQ(local_tile_capacity,
           rank_major_tile_capacity(task_count, world_size));
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world_size);
  torch::Tensor local_tiles;
  FrameSelection selection;
  int64_t local_tile_count = 0;
  // Owner = task % world_size, slot = task / world_size.
  for (int64_t task_index = rank; task_index < task_count;
       task_index += world_size) {
    torch::Tensor tile = produce_tile(task_index);
    if (!local_tiles.defined()) {
      CHECK_EQ(tile.dim(), 5);
      selection = select_frames(tile);
      std::vector<int64_t> shape = tile.sizes().vec();
      shape[2] = selection.first_count + selection.second_count;
      shape.insert(shape.begin(), local_tile_capacity);
      local_tiles = torch::empty(shape, tile.options());
    }
    copy_retained_frames(tile, local_tiles[local_tile_count], selection);
    ++local_tile_count;
  }
  if (local_tile_count < local_tile_capacity) {
    local_tiles.slice(/*dim=*/0, local_tile_count).zero_();
  }

  torch::Tensor gathered_tiles = gather_tiles(local_tiles);
  std::vector<int64_t> expected_shape = local_tiles.sizes().vec();
  expected_shape.insert(expected_shape.begin(), world_size);
  CHECK(gathered_tiles.sizes().vec() == expected_shape);
  CHECK(gathered_tiles.device() == local_tiles.device());
  CHECK(gathered_tiles.scalar_type() == local_tiles.scalar_type());
  return gathered_tiles;
}

inline torch::Tensor rank_major_tile(const torch::Tensor& gathered_tiles,
                                     int64_t task_index) {
  const int64_t world_size = gathered_tiles.size(/*dim=*/0);
  return gathered_tiles[task_index % world_size][task_index / world_size];
}

// Window driver. Each callback decodes one original tile,
// gathers a rank-major buffer, or assembles one clip in its original order.
// The caller selects supported execution before entering this function.
template <typename DecodeTile, typename GatherTiles, typename AssembleClip>
void visit_global_tile_windows(int64_t clip_count,
                               int64_t tiles_per_clip,
                               int64_t world_size,
                               int64_t rank,
                               const FrameSelection& selection,
                               const VideoDecodeOptions& options,
                               const DecodeTile& decode_tile,
                               const GatherTiles& gather_tiles,
                               const AssembleClip& assemble_clip) {
  CHECK_GT(clip_count, 0);
  CHECK_GT(world_size, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world_size);
  CHECK_GE(tiles_per_clip, world_size);

  for (int64_t first_clip_index = 0; first_clip_index < clip_count;) {
    const TileWindow window = plan_tile_window(
        clip_count - first_clip_index, tiles_per_clip, world_size, options);
    const int64_t task_count = window.clip_count * tiles_per_clip;
    const auto produce_tile = [&](int64_t task_index) {
      return decode_tile(first_clip_index + task_index / tiles_per_clip,
                         task_index % tiles_per_clip);
    };
    const auto select_frames = [&selection](const torch::Tensor& /*tile*/) {
      return selection;
    };
    torch::Tensor gathered_tiles =
        gather_rank_major_tiles(task_count,
                                window.local_tile_capacity,
                                world_size,
                                rank,
                                produce_tile,
                                select_frames,
                                gather_tiles);
    for (int64_t window_clip_index = 0; window_clip_index < window.clip_count;
         ++window_clip_index) {
      std::vector<torch::Tensor> tiles;
      tiles.reserve(static_cast<size_t>(tiles_per_clip));
      for (int64_t tile_index = 0; tile_index < tiles_per_clip; ++tile_index) {
        const int64_t task_index =
            window_clip_index * tiles_per_clip + tile_index;
        tiles.emplace_back(rank_major_tile(gathered_tiles, task_index));
      }
      assemble_clip(first_clip_index + window_clip_index, tiles);
    }
    first_clip_index += window.clip_count;
  }
}

}  // namespace xllm::minimax_h3_video
