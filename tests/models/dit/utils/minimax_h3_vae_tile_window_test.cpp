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

#include "models/dit/utils/minimax_h3_vae_tile_window.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace xllm::minimax_h3_video {
namespace {

TEST(MinimaxH3VaeTileWindowTest, WindowUsesConfiguredClipLimit) {
  const VideoDecodeOptions options{.max_window_clips = 8};
  const TileWindow window = plan_tile_window(/*remaining_clip_count=*/11,
                                             /*tiles_per_clip=*/4096,
                                             /*world_size=*/8,
                                             options);
  EXPECT_EQ(window.clip_count, 8);
  EXPECT_EQ(window.local_tile_capacity, 4096);
}

TEST(MinimaxH3VaeTileWindowTest, RemainingClipsUseEqualRankCapacity) {
  const VideoDecodeOptions options{.max_window_clips = 8};
  const TileWindow window = plan_tile_window(/*remaining_clip_count=*/3,
                                             /*tiles_per_clip=*/28,
                                             /*world_size=*/8,
                                             options);
  EXPECT_EQ(window.clip_count, 3);
  EXPECT_EQ(window.local_tile_capacity, 11);

  constexpr int64_t kClipCount = 11;
  constexpr int64_t kTilesPerClip = 28;
  constexpr int64_t kWorldSize = 8;
  const torch::TensorOptions tensor_options = torch::dtype(torch::kInt64);
  const torch::Tensor source =
      torch::arange(kClipCount * kTilesPerClip * 28, tensor_options)
          .reshape({kClipCount, kTilesPerClip, 1, 1, 28, 1, 1});
  // Oracle: the original two slices, independent of the packing helper.
  const torch::Tensor expected_tiles =
      torch::cat({source.slice(/*dim=*/4, /*start=*/3, /*end=*/20),
                  source.slice(/*dim=*/4, /*start=*/23, /*end=*/28)},
                 /*dim=*/4);
  const FrameSelection selection = retained_frames(/*frame_count=*/28,
                                                   /*section_frame_count=*/20,
                                                   /*pre_padding=*/3);

  using TileIndex = std::pair<int64_t, int64_t>;
  using RankTasks = std::vector<std::vector<TileIndex>>;
  const std::array<int64_t, 2> window_clip_counts{8, 3};
  const std::array<int64_t, 2> window_capacities{28, 11};
  std::vector<RankTasks> expected_tasks;
  std::vector<torch::Tensor> gathered_windows;
  expected_tasks.reserve(window_clip_counts.size());
  gathered_windows.reserve(window_clip_counts.size());
  int64_t first_clip_index = 0;
  for (size_t window_index = 0; window_index < window_clip_counts.size();
       ++window_index) {
    RankTasks rank_tasks(static_cast<size_t>(kWorldSize));
    for (std::vector<TileIndex>& tasks : rank_tasks) {
      tasks.reserve(static_cast<size_t>(window_capacities[window_index]));
    }
    torch::Tensor gathered = torch::zeros(
        {kWorldSize, window_capacities[window_index], 1, 1, 22, 1, 1},
        tensor_options);
    // Deal tasks cyclically into per-rank queues rather than repeating the
    // driver's task-index modulo/division mapping.
    int64_t owner = 0;
    const int64_t last_clip_index =
        first_clip_index + window_clip_counts[window_index];
    for (int64_t clip_index = first_clip_index; clip_index < last_clip_index;
         ++clip_index) {
      for (int64_t tile_index = 0; tile_index < kTilesPerClip; ++tile_index) {
        std::vector<TileIndex>& tasks = rank_tasks[static_cast<size_t>(owner)];
        const int64_t slot = static_cast<int64_t>(tasks.size());
        tasks.emplace_back(clip_index, tile_index);
        gathered[owner][slot].copy_(expected_tiles[clip_index][tile_index]);
        ++owner;
        if (owner == kWorldSize) {
          owner = 0;
        }
      }
    }
    expected_tasks.emplace_back(std::move(rank_tasks));
    gathered_windows.emplace_back(std::move(gathered));
    first_clip_index = last_clip_index;
  }

  for (int64_t rank = 0; rank < kWorldSize; ++rank) {
    SCOPED_TRACE(rank);
    size_t window_index = 0;
    int64_t next_clip_index = 0;
    std::vector<TileIndex> decoded_tasks;
    decoded_tasks.reserve(static_cast<size_t>(window_capacities.front()));
    const auto decode_tile = [&source, &decoded_tasks](int64_t clip_index,
                                                       int64_t tile_index) {
      decoded_tasks.emplace_back(clip_index, tile_index);
      return source[clip_index][tile_index];
    };
    const auto gather_tiles = [&expected_tasks,
                               &gathered_windows,
                               &decoded_tasks,
                               &window_index,
                               rank](const torch::Tensor& local_tiles) {
      EXPECT_EQ(decoded_tasks,
                expected_tasks.at(window_index)[static_cast<size_t>(rank)]);
      // Includes zero padding on ranks 4-7 in the final 84-task window.
      EXPECT_TRUE(
          torch::equal(local_tiles, gathered_windows.at(window_index)[rank]));
      decoded_tasks.clear();
      return gathered_windows.at(window_index++);
    };
    const auto assemble_clip = [&expected_tiles, &next_clip_index](
                                   int64_t clip_index,
                                   const std::vector<torch::Tensor>& tiles) {
      ASSERT_EQ(clip_index, next_clip_index);
      ASSERT_EQ(tiles.size(), static_cast<size_t>(kTilesPerClip));
      for (int64_t tile_index = 0; tile_index < kTilesPerClip; ++tile_index) {
        EXPECT_TRUE(torch::equal(tiles[static_cast<size_t>(tile_index)],
                                 expected_tiles[clip_index][tile_index]));
      }
      ++next_clip_index;
    };
    visit_global_tile_windows(kClipCount,
                              kTilesPerClip,
                              kWorldSize,
                              rank,
                              selection,
                              options,
                              decode_tile,
                              gather_tiles,
                              assemble_clip);
    EXPECT_EQ(window_index, window_clip_counts.size());
    EXPECT_EQ(next_clip_index, kClipCount);
    EXPECT_TRUE(decoded_tasks.empty());
  }
}

TEST(MinimaxH3VaeTileWindowTest, FullTilesPreserveContentsAndPadding) {
  constexpr int64_t kTaskCount = 7;
  constexpr int64_t kWorldSize = 4;
  constexpr int64_t kCapacity = 2;
  const torch::TensorOptions tensor_options = torch::dtype(torch::kInt64);
  // Full encoder/decoder tiles need not have RGB channels or 28 frames.
  const torch::Tensor source =
      torch::arange(kTaskCount * 2 * 5 * 3 * 2 * 2, tensor_options)
          .reshape({kTaskCount, 2, 5, 3, 2, 2});
  const std::array<std::vector<int64_t>, 4> rank_tasks{{
      {0, 4},
      {1, 5},
      {2, 6},
      {3},
  }};
  const torch::Tensor expected_gathered =
      torch::zeros({kWorldSize, kCapacity, 2, 5, 3, 2, 2}, tensor_options);
  for (int64_t rank = 0; rank < kWorldSize; ++rank) {
    const auto& tasks = rank_tasks[static_cast<size_t>(rank)];
    for (size_t slot = 0; slot < tasks.size(); ++slot) {
      expected_gathered[rank][static_cast<int64_t>(slot)].copy_(
          source[tasks[slot]]);
    }
  }
  for (int64_t rank = 0; rank < kWorldSize; ++rank) {
    std::vector<int64_t> produced_tasks;
    produced_tasks.reserve(kCapacity);
    const auto produce_tile = [&](int64_t task_index) {
      produced_tasks.emplace_back(task_index);
      return source[task_index];
    };
    const auto select_frames = [](const torch::Tensor& tile) {
      return FrameSelection{.first_count = tile.size(/*dim=*/2)};
    };
    const auto gather_tiles = [&](const torch::Tensor& local_tiles) {
      EXPECT_EQ(produced_tasks, rank_tasks[static_cast<size_t>(rank)]);
      EXPECT_TRUE(torch::equal(local_tiles, expected_gathered[rank]));
      return expected_gathered.clone();
    };
    std::vector<torch::Tensor> restored_tiles;
    restored_tiles.reserve(kTaskCount);
    {
      const torch::Tensor gathered = gather_rank_major_tiles(kTaskCount,
                                                             kCapacity,
                                                             kWorldSize,
                                                             rank,
                                                             produce_tile,
                                                             select_frames,
                                                             gather_tiles);
      for (int64_t task_index = 0; task_index < kTaskCount; ++task_index) {
        restored_tiles.emplace_back(rank_major_tile(gathered, task_index));
      }
    }
    // Views must retain the receive storage after the collector goes out of
    // scope.
    for (int64_t task_index = 0; task_index < kTaskCount; ++task_index) {
      EXPECT_TRUE(torch::equal(restored_tiles[static_cast<size_t>(task_index)],
                               source[task_index]));
    }
  }
}

TEST(MinimaxH3VaeTileWindowTest, RetainedFramesPackNormalAndBoundaryInputs) {
  struct FrameCase {
    int64_t frame_count;
    int64_t pre_padding;
    FrameSelection expected_selection;
    std::vector<int64_t> frame_indices;
  };
  // Normal 17+5 packing, a short clip with no second range, and padding
  // beyond a section (both retained ranges empty) are distinct contracts.
  const std::array<FrameCase, 3> cases{{
      {28, 3, {3, 17, 23, 5}, {3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                               14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27}},
      {5, 3, {3, 2, 5, 0}, {3, 4}},
      {28, 21, {20, 0, 28, 0}, {}},
  }};
  const torch::TensorOptions tensor_options = torch::dtype(torch::kInt64);
  for (const FrameCase& test_case : cases) {
    SCOPED_TRACE(::testing::Message()
                 << "frame_count=" << test_case.frame_count
                 << ", pre_padding=" << test_case.pre_padding);
    const FrameSelection selection = retained_frames(test_case.frame_count,
                                                     /*section_frame_count=*/20,
                                                     test_case.pre_padding);
    EXPECT_EQ(selection.first_begin, test_case.expected_selection.first_begin);
    EXPECT_EQ(selection.first_count, test_case.expected_selection.first_count);
    EXPECT_EQ(selection.second_begin,
              test_case.expected_selection.second_begin);
    EXPECT_EQ(selection.second_count,
              test_case.expected_selection.second_count);

    const torch::Tensor source =
        torch::arange(2 * 3 * test_case.frame_count * 2 * 3, tensor_options)
            .reshape({2, 3, test_case.frame_count, 2, 3});
    const int64_t retained_count =
        static_cast<int64_t>(test_case.frame_indices.size());
    const torch::Tensor packed = torch::full(
        {2, 3, retained_count, 2, 3}, /*fill_value=*/-1, tensor_options);
    copy_retained_frames(source, packed, selection);
    // Literal frame IDs are an oracle independent of selection arithmetic.
    const torch::Tensor expected = source.index_select(
        /*dim=*/2, torch::tensor(test_case.frame_indices, tensor_options));
    EXPECT_TRUE(torch::equal(packed, expected));
  }
}

}  // namespace
}  // namespace xllm::minimax_h3_video
