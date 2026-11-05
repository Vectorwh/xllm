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

#include "models/vlm/utils/multimodal_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace xllm {
namespace {

class EncoderBroadcastWork final : public c10d::Work {
 public:
  EncoderBroadcastWork(torch::Tensor output,
                       torch::Tensor reference,
                       const int32_t* posted_count,
                       int32_t expected_count,
                       int32_t* waited_count)
      : output_(std::move(output)),
        reference_(std::move(reference)),
        posted_count_(posted_count),
        expected_count_(expected_count),
        waited_count_(waited_count) {}

  bool wait(std::chrono::milliseconds /*timeout*/ = kNoTimeout) override {
    EXPECT_EQ(*posted_count_, expected_count_);
    output_.copy_(reference_);
    ++*waited_count_;
    return true;
  }

 private:
  torch::Tensor output_;
  torch::Tensor reference_;
  const int32_t* posted_count_;
  int32_t expected_count_;
  int32_t* waited_count_;
};

class EncoderProcessGroup final : public ProcessGroup {
 public:
  EncoderProcessGroup(int32_t rank,
                      std::vector<std::vector<int32_t>> assignment,
                      std::vector<torch::Tensor> embeddings)
      : ProcessGroup(rank,
                     static_cast<int32_t>(assignment.size()),
                     torch::Device(torch::kCPU)),
        assignment_(std::move(assignment)),
        embeddings_(std::move(embeddings)) {}

  c10::intrusive_ptr<c10d::Work> broadcast_async(torch::Tensor& input,
                                                 int32_t root_rank) override {
    std::vector<torch::Tensor> expected;
    expected.reserve(assignment_[root_rank].size());
    for (int32_t item : assignment_[root_rank]) {
      expected.emplace_back(embeddings_[item]);
    }
    torch::Tensor reference = torch::cat(expected, 0);
    EXPECT_EQ(input.sizes(), reference.sizes());
    if (rank() == root_rank) {
      EXPECT_TRUE(torch::equal(input, reference));
    }
    ++broadcast_count_;
    if (world_size() == 1) {
      return nullptr;
    }
    const int32_t active_ranks = static_cast<int32_t>(std::count_if(
        assignment_.begin(), assignment_.end(), [](const auto& items) {
          return !items.empty();
        }));
    return c10::make_intrusive<EncoderBroadcastWork>(input,
                                                     std::move(reference),
                                                     &broadcast_count_,
                                                     active_ranks,
                                                     &waited_count_);
  }

  int32_t broadcast_count() const { return broadcast_count_; }
  int32_t waited_count() const { return waited_count_; }

 private:
  std::vector<std::vector<int32_t>> assignment_;
  std::vector<torch::Tensor> embeddings_;
  int32_t broadcast_count_ = 0;
  int32_t waited_count_ = 0;
};

void check_encoder_outputs(const std::vector<int32_t>& token_nums,
                           int32_t world_size) {
  constexpr int64_t kFeatureDim = 4;
  std::vector<torch::Tensor> pixels;
  std::vector<torch::Tensor> grids;
  std::vector<torch::Tensor> expected;
  std::vector<int32_t> patch_nums;
  pixels.reserve(token_nums.size());
  grids.reserve(token_nums.size());
  expected.reserve(token_nums.size());
  patch_nums.reserve(token_nums.size());
  int64_t total_tokens = 0;
  for (size_t item = 0; item < token_nums.size(); ++item) {
    pixels.emplace_back(torch::full({token_nums[item] * 4, kFeatureDim},
                                    static_cast<float>(item + 1)));
    grids.emplace_back(torch::tensor({{1, 2, token_nums[item] * 2}}));
    expected.emplace_back(torch::full({token_nums[item], kFeatureDim},
                                      static_cast<float>(item + 1)));
    patch_nums.emplace_back(token_nums[item] * 4);
    total_tokens += token_nums[item];
  }
  const auto assignment = build_lpt_assignment(patch_nums, world_size);
  const int32_t active_ranks = static_cast<int32_t>(std::count_if(
      assignment.begin(), assignment.end(), [](const auto& items) {
        return !items.empty();
      }));
  torch::Tensor pixel_values = torch::cat(pixels, 0);
  torch::Tensor grid_thw = torch::cat(grids, 0);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    EncoderProcessGroup group(rank, assignment, expected);
    int32_t encoder_calls = 0;
    auto encoder = [&encoder_calls](const torch::Tensor& input,
                                    const torch::Tensor& grid) {
      ++encoder_calls;
      EXPECT_GT(grid.size(0), 0);
      return input.reshape({-1, 4, kFeatureDim}).select(1, 0).clone();
    };
    auto outputs = run_dp_encoder(
        encoder, pixel_values, grid_thw, token_nums, kFeatureDim, &group);
    ASSERT_EQ(outputs.size(), expected.size());
    std::unordered_set<const void*> storages;
    int64_t retained_bytes = 0;
    for (size_t item = 0; item < outputs.size(); ++item) {
      EXPECT_TRUE(torch::equal(outputs[item], expected[item]));
      const void* storage = outputs[item].storage().data_ptr().get();
      if (storages.insert(storage).second) {
        retained_bytes += outputs[item].storage().nbytes();
      }
    }
    EXPECT_EQ(retained_bytes, total_tokens * kFeatureDim * sizeof(float));
    EXPECT_EQ(encoder_calls, assignment[rank].empty() ? 0 : 1);
    EXPECT_EQ(group.broadcast_count(), active_ranks);
    EXPECT_EQ(group.waited_count(), world_size == 1 ? 0 : active_ranks);
  }
}

TEST(MultimodalEncoderDpTest, SingleImageDoesNotRetainEmptyRankPadding) {
  check_encoder_outputs({8}, 8);
}

TEST(MultimodalEncoderDpTest, UnevenImagesKeepOriginalOrderWithoutPadding) {
  check_encoder_outputs({1, 16, 2, 3, 4}, 8);
}

TEST(MultimodalEncoderDpTest, FourteenImagesOnEightRanks) {
  check_encoder_outputs(std::vector<int32_t>(14, 4), 8);
}

TEST(MultimodalEncoderDpTest, FourteenUnevenImagesOnEightRanks) {
  check_encoder_outputs({1, 32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 64}, 8);
}

TEST(MultimodalEncoderDpTest, SingleRankPreservesOutputs) {
  check_encoder_outputs({3, 1, 2}, 1);
}

TEST(MultimodalEmbeddingTest, MergeReleasesBatchCopyAndPreservesItemOutputs) {
  torch::Tensor image =
      torch::tensor({{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f, 40.0f, 41.0f}});
  torch::Tensor video =
      torch::tensor({{12.0f, 13.0f, 22.0f, 23.0f, 32.0f, 33.0f, 42.0f, 43.0f}});
  MMDataItem item(MMType::IMAGE, {{"image|embedding", image}});
  MMBatchData data({MMData(MMType::IMAGE, MMItemVec{item})});
  data.replace({{"image|embedding", image.clone()},
                {"image|mask", torch::tensor({false, true, false, false})},
                {"video|embedding", video.clone()},
                {"video|mask", torch::tensor({false, false, true, false})}});
  const auto image_storage = data.get<torch::Tensor>("image|embedding")
                                 ->storage()
                                 .getWeakStorageImpl();
  const auto video_storage = data.get<torch::Tensor>("video|embedding")
                                 ->storage()
                                 .getWeakStorageImpl();
  torch::Tensor inputs =
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});
  auto deepstacks = merge_multimodal_embeddings(inputs, data, 3);

  EXPECT_TRUE(torch::equal(
      inputs,
      torch::tensor(
          {{1.0f, 2.0f}, {10.0f, 11.0f}, {12.0f, 13.0f}, {7.0f, 8.0f}})));
  ASSERT_EQ(deepstacks.size(), 3);
  for (int64_t index = 0; index < 3; ++index) {
    const float value = static_cast<float>((index + 2) * 10);
    EXPECT_TRUE(torch::equal(deepstacks[index],
                             torch::tensor({{0.0f, 0.0f},
                                            {value, value + 1},
                                            {value + 2, value + 3},
                                            {0.0f, 0.0f}})));
  }
  EXPECT_TRUE(data.data().empty());
  EXPECT_TRUE(image_storage.expired());
  EXPECT_TRUE(video_storage.expired());
  ASSERT_EQ(data.mm_data_vec().size(), 1);
  EXPECT_TRUE(torch::equal(data.mm_data_vec()[0]
                               .items<MMItemVec>()[0]
                               .get<torch::Tensor>("image|embedding")
                               .value(),
                           image));
}

TEST(MultimodalEmbeddingDeathTest, RejectsIncompletePackedFeatures) {
  for (const int64_t width : {2, 7}) {
    torch::Tensor inputs = torch::zeros({2, 2});
    MMBatchData data(MMType::IMAGE,
                     {{"image|embedding", torch::zeros({1, width})},
                      {"image|mask", torch::tensor({true, false})}});
    EXPECT_DEATH(
        merge_multimodal_embeddings(inputs, data, /*num_deepstacks=*/3),
        "Packed multimodal width must match the main and DeepStack widths");
  }
}

TEST(MultimodalEmbeddingTest, TextOnlyDoesNotAllocateDeepstacks) {
  torch::Tensor inputs = torch::ones({4, 2});
  MMBatchData data;
  EXPECT_TRUE(merge_multimodal_embeddings(inputs, data, 3).empty());
  EXPECT_TRUE(torch::equal(inputs, torch::ones({4, 2})));
}

TEST(MultimodalEmbeddingTest, ScheduledChunkDoesNotConsumeFutureImageTokens) {
  const torch::Tensor full = torch::arange(32, torch::kFloat32).view({4, 8});
  MMDataItem item(MMType::IMAGE, {{"image|embedding", full}});
  MMBatchData data({MMData(MMType::IMAGE, MMItemVec{item})});
  for (int64_t offset : {0, 2}) {
    data.replace(
        {{"image|embedding", full.slice(0, offset, offset + 2).clone()},
         {"image|mask", torch::tensor({true, true})}});
    torch::Tensor inputs = torch::zeros({2, 2});
    const auto deepstacks = merge_multimodal_embeddings(inputs, data, 3);
    EXPECT_TRUE(
        torch::equal(inputs, full.slice(0, offset, offset + 2).slice(1, 0, 2)));
    for (int64_t index = 0; index < 3; ++index) {
      EXPECT_TRUE(
          torch::equal(deepstacks[index],
                       full.slice(0, offset, offset + 2)
                           .slice(1, (index + 1) * 2, (index + 2) * 2)));
    }
    EXPECT_TRUE(torch::equal(data.mm_data_vec()[0]
                                 .items<MMItemVec>()[0]
                                 .get<torch::Tensor>("image|embedding")
                                 .value(),
                             full));
  }
}

}  // namespace
}  // namespace xllm
