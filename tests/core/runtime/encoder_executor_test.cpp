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

#include "core/runtime/encoder_executor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "processors/processor_cache.h"

namespace xllm {
namespace {

MMDataItem make_item(MMType modality,
                     int32_t offset,
                     const torch::Tensor& mask,
                     uint8_t key) {
  MMDataItem item(modality);
  auto& state = item.mutable_state();
  state.mutable_token_pos() = {offset, static_cast<int32_t>(mask.numel())};
  state.mutable_mm_token_mask() = mask;
  state.mutable_mm_token_num() = mask.sum().item<int32_t>();
  state.mutable_schedule_data().key.data[0] = key;
  return item;
}

ModelInputParams schedule(MMData data, int32_t computed, int32_t query_length) {
  UpdateMMItemScheduleStateVisitor visitor(computed, query_length, 0);
  EXPECT_TRUE(data.foreach (visitor));
  ModelInputParams params;
  params.multimodal.mm_data =
      MMBatchData({MMData(visitor.scheduled_type_, visitor.mm_data_items_)});
  params.execution_batch.num_reqs = 1;
  params.execution_batch.num_tokens = query_length;
  params.execution_batch.num_scheduled_tokens = {query_length};
  params.execution_batch.num_computed_tokens = {computed};
  params.execution_batch.query_start_loc = {0, query_length};
  params.execution_batch.is_prefilling = {1};
  return params;
}

double mm_cache_lookup_count(const char* cache, const char* outcome) {
  return MULTI_COUNTER_mm_cache_lookups_total.get_stats({cache, outcome})
      ->get_value();
}

class MMCacheLookupMetricsTest : public ::testing::Test {
 protected:
  void expect_lookup_counts(double processor_hits,
                            double processor_misses,
                            double encoder_hits,
                            double encoder_misses,
                            int64_t encoder_samples) const {
    EXPECT_EQ(mm_cache_lookup_count(/*cache=*/"processor", /*outcome=*/"hit"),
              processor_hits_ + processor_hits);
    EXPECT_EQ(mm_cache_lookup_count(/*cache=*/"processor", /*outcome=*/"miss"),
              processor_misses_ + processor_misses);
    EXPECT_EQ(mm_cache_lookup_count(/*cache=*/"encoder", /*outcome=*/"hit"),
              encoder_hits_ + encoder_hits);
    EXPECT_EQ(mm_cache_lookup_count(/*cache=*/"encoder", /*outcome=*/"miss"),
              encoder_misses_ + encoder_misses);
    EXPECT_EQ(HISTOGRAM_encoder_cache_hit_rate.count(),
              encoder_samples_ + encoder_samples);
  }

 private:
  // Use deltas because metrics are process-global and other tests update them.
  const double processor_hits_ =
      mm_cache_lookup_count(/*cache=*/"processor", /*outcome=*/"hit");
  const double processor_misses_ =
      mm_cache_lookup_count(/*cache=*/"processor", /*outcome=*/"miss");
  const double encoder_hits_ =
      mm_cache_lookup_count(/*cache=*/"encoder", /*outcome=*/"hit");
  const double encoder_misses_ =
      mm_cache_lookup_count(/*cache=*/"encoder", /*outcome=*/"miss");
  const int64_t encoder_samples_ = HISTOGRAM_encoder_cache_hit_rate.count();
};

TEST_F(MMCacheLookupMetricsTest, CountsProcessorMissAndHit) {
  ProcessorCache cache(/*max_items=*/1);
  MMInputItem input;
  input.type = MMType::IMAGE;
  input.hash_key = XXH3Key{};
  ProcessorCacheLookupVisitor lookup(cache, /*item_count=*/2);

  ASSERT_TRUE(lookup.visit(input));
  ASSERT_EQ(lookup.cache_hits_.size(), 1);
  EXPECT_FALSE(lookup.cache_hits_[0].has_value());
  EXPECT_EQ(lookup.miss_inputs_.size(), 1);
  expect_lookup_counts(/*processor_hits=*/0,
                       /*processor_misses=*/1,
                       /*encoder_hits=*/0,
                       /*encoder_misses=*/0,
                       /*encoder_samples=*/0);

  MMDataItem processed(MMType::IMAGE);
  const torch::Tensor pixels = torch::tensor({{1.0f, 2.0f}});
  processed.add(/*key=*/"pixel_values", pixels);
  cache.insert(input.hash_key.value(), processed);
  ASSERT_TRUE(lookup.visit(input));
  ASSERT_EQ(lookup.cache_hits_.size(), 2);
  ASSERT_TRUE(lookup.cache_hits_[1].has_value());
  EXPECT_TRUE(torch::equal(
      lookup.cache_hits_[1]->get<torch::Tensor>(/*key=*/"pixel_values").value(),
      pixels));
  EXPECT_EQ(lookup.miss_inputs_.size(), 1);
  expect_lookup_counts(/*processor_hits=*/1,
                       /*processor_misses=*/1,
                       /*encoder_hits=*/0,
                       /*encoder_misses=*/0,
                       /*encoder_samples=*/0);
}

TEST_F(MMCacheLookupMetricsTest, SkipsProcessorBypasses) {
  ProcessorCache cache(/*max_items=*/1);
  ProcessorCacheLookupVisitor lookup(cache, /*item_count=*/2);
  MMInputItem unhashed;
  unhashed.type = MMType::IMAGE;
  unhashed.decode_image = torch::zeros({3, 1, 1}, torch::kUInt8);
  ASSERT_TRUE(lookup.visit(unhashed));

  MMInputItem preembedded;
  preembedded.type = MMType::IMAGE;
  preembedded.hash_key = XXH3Key{};
  preembedded.embedding.embedding = torch::tensor({{3.0f, 4.0f}});
  ASSERT_TRUE(lookup.visit(preembedded));

  // Both inputs need processing, but neither performed a cache lookup.
  ASSERT_EQ(lookup.cache_hits_.size(), 2);
  EXPECT_FALSE(lookup.cache_hits_[0].has_value());
  EXPECT_FALSE(lookup.cache_hits_[1].has_value());
  EXPECT_EQ(lookup.miss_inputs_.size(), 2);
  expect_lookup_counts(/*processor_hits=*/0,
                       /*processor_misses=*/0,
                       /*encoder_hits=*/0,
                       /*encoder_misses=*/0,
                       /*encoder_samples=*/0);
}

TEST_F(MMCacheLookupMetricsTest, CountsEncoderHitAndMiss) {
  EncoderCache cache(/*max_size=*/1024);
  EncoderCacheLookupVisitor lookup(&cache);
  MMDataItem item =
      make_item(MMType::IMAGE, /*offset=*/0, torch::tensor({true}), /*key=*/21);
  ASSERT_TRUE(lookup.visit(item));
  EXPECT_FALSE(item.is_embedded());
  expect_lookup_counts(/*processor_hits=*/0,
                       /*processor_misses=*/0,
                       /*encoder_hits=*/0,
                       /*encoder_misses=*/1,
                       /*encoder_samples=*/1);

  const torch::Tensor embedding = torch::tensor({{5.0f, 6.0f}});
  cache.insert(item.state().schedule_data().key, embedding);
  ASSERT_TRUE(lookup.visit(item));
  EXPECT_TRUE(item.is_embedded());
  EXPECT_TRUE(torch::equal(
      item.get<torch::Tensor>(/*key=*/"image|embedding").value(), embedding));
  expect_lookup_counts(/*processor_hits=*/0,
                       /*processor_misses=*/0,
                       /*encoder_hits=*/1,
                       /*encoder_misses=*/1,
                       /*encoder_samples=*/2);

  ASSERT_TRUE(lookup.visit(item));
  expect_lookup_counts(/*processor_hits=*/0,
                       /*processor_misses=*/0,
                       /*encoder_hits=*/1,
                       /*encoder_misses=*/1,
                       /*encoder_samples=*/2);
}

TEST_F(MMCacheLookupMetricsTest, SkipsPrecomputedEncoderEmbedding) {
  MMDataItem item =
      make_item(MMType::VIDEO, /*offset=*/0, torch::tensor({true}), /*key=*/22);
  const torch::Tensor embedding = torch::tensor({{7.0f, 8.0f}});
  item.add(/*key=*/"video|embedding", embedding);

  auto params = schedule(MMData(MMType::VIDEO, MMItemVec{item}),
                         /*computed=*/0,
                         /*query_length=*/1);
  EncoderExecutor executor(/*max_cache_bytes=*/1024);
  int32_t encoder_calls = 0;
  executor.prepare(params,
                   [&encoder_calls](const ModelInputParams& inputs) -> MMDict {
                     ++encoder_calls;
                     EXPECT_TRUE(inputs.multimodal.mm_data.data().empty());
                     return {};
                   });
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      params.multimodal.mm_data.get<torch::Tensor>(/*key=*/"video|embedding")
          .value(),
      embedding));
  expect_lookup_counts(/*processor_hits=*/0,
                       /*processor_misses=*/0,
                       /*encoder_hits=*/0,
                       /*encoder_misses=*/0,
                       /*encoder_samples=*/0);
}

TEST_F(MMCacheLookupMetricsTest, DisabledEncoderCacheDoesNotCountLookups) {
  MMDataItem item =
      make_item(MMType::IMAGE, /*offset=*/0, torch::tensor({true}), /*key=*/23);
  const torch::Tensor pixels = torch::tensor({{9.0f, 10.0f}});
  item.add(/*key=*/"pixel_values", pixels);
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{item}),
                         /*computed=*/0,
                         /*query_length=*/1);
  EncoderExecutor executor(/*max_cache_bytes=*/0);
  int32_t encoder_calls = 0;
  executor.prepare(
      params, [&encoder_calls](const ModelInputParams& inputs) -> MMDict {
        ++encoder_calls;
        const torch::Tensor pixels =
            inputs.multimodal.mm_data.get<torch::Tensor>(/*key=*/"pixel_values")
                .value();
        return {{"image|embedding", std::vector<torch::Tensor>{pixels}}};
      });
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      params.multimodal.mm_data.get<torch::Tensor>(/*key=*/"image|embedding")
          .value(),
      pixels));
  expect_lookup_counts(/*processor_hits=*/0,
                       /*processor_misses=*/0,
                       /*encoder_hits=*/0,
                       /*encoder_misses=*/0,
                       /*encoder_samples=*/0);
}

TEST(EncoderExecutorTest, CacheHitStillCallsEncode) {
  MMDataItem item =
      make_item(MMType::IMAGE, /*offset=*/0, torch::tensor({true}), /*key=*/31);
  const torch::Tensor pixels = torch::tensor({{1.0f, 2.0f}});
  item.add(/*key=*/"pixel_values", pixels);
  EncoderExecutor executor(/*max_cache_bytes=*/1024);
  int32_t calls = 0;
  for (int32_t iteration = 0; iteration < 2; ++iteration) {
    auto params = schedule(MMData(MMType::IMAGE, MMItemVec{item}),
                           /*computed=*/0,
                           /*query_length=*/1);
    executor.prepare(
        params, [&calls](const ModelInputParams& inputs) -> MMDict {
          ++calls;
          const auto pixels =
              inputs.multimodal.mm_data.get<torch::Tensor>("pixel_values");
          if (!pixels.has_value()) {
            return {};
          }
          return {
              {"image|embedding", std::vector<torch::Tensor>{pixels.value()}}};
        });
    EXPECT_TRUE(torch::equal(
        params.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
        pixels));
  }
  EXPECT_EQ(calls, 2);
}

TEST(EncoderExecutorTest, MissingMultimodalDataSkipsEncode) {
  EncoderExecutor executor(/*max_cache_bytes=*/0);
  int32_t calls = 0;
  executor.prepare(ModelInputParams{},
                   [&calls](const ModelInputParams& /*inputs*/) -> MMDict {
                     ++calls;
                     return {};
                   });
  EXPECT_EQ(calls, 0);
}

TEST(EncoderExecutorTest, UsesLogicalWindowsInsteadOfAttentionLayout) {
  MMDataItem first = make_item(MMType::IMAGE, 2, torch::tensor({true}), 11);
  MMDataItem second = make_item(MMType::IMAGE, 3, torch::tensor({true}), 12);
  first.add("pixel_values", torch::tensor({{1.0f, 2.0f}}));
  second.add("pixel_values", torch::tensor({{3.0f, 4.0f}}));
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{first, second}), 2, 2);
  // Attention rows can describe padding/expansion instead of logical requests.
  params.meta.num_sequences = 64;
  params.attention.host.kv_seq_lens = {0, 4};
  params.attention.host.q_seq_lens = {0, 4};
  EncoderExecutor executor(0);
  int32_t encoder_calls = 0;
  executor.prepare(
      params, [&encoder_calls](const ModelInputParams& inputs) -> MMDict {
        ++encoder_calls;
        EXPECT_EQ(inputs.execution_batch.num_reqs, 1);
        EXPECT_EQ(inputs.execution_batch.num_scheduled_tokens,
                  (std::vector<int32_t>{2}));
        EXPECT_EQ(inputs.execution_batch.num_computed_tokens,
                  (std::vector<int32_t>{2}));
        const torch::Tensor pixels =
            inputs.multimodal.mm_data.get<torch::Tensor>("pixel_values")
                .value();
        EXPECT_TRUE(
            torch::equal(pixels, torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}})));
        return {{"image|embedding", pixels.mul(10).split(1)}};
      });
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      params.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      torch::tensor({{10.0f, 20.0f}, {30.0f, 40.0f}})));
  EXPECT_TRUE(torch::equal(
      params.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({true, true})));
}

class EncoderExecutorDeathTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(EncoderExecutorDeathTest, RejectsMissingExecutionCounts) {
  MMDataItem item = make_item(MMType::IMAGE, 0, torch::tensor({true}), 18);
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 1);
  EncoderExecutor executor(0);
  params.execution_batch = {};
  EXPECT_DEATH(
      executor.prepare(params,
                       [](const ModelInputParams&) -> MMDict { return {}; }),
      "multimodal encoder prepare requires execution_batch");

  params = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 1);
  params.execution_batch.num_computed_tokens.clear();
  EXPECT_DEATH(
      executor.prepare(params,
                       [](const ModelInputParams&) -> MMDict { return {}; }),
      "computed token counts do not match logical request count");
}

TEST_F(EncoderExecutorDeathTest, RejectsInvalidWindowCounts) {
  EXPECT_DEATH(EncoderEmbeddingGatherVisitor(/*num_computed_tokens=*/{-1},
                                             /*num_scheduled_tokens=*/{2}),
               "negative computed token count");
  EXPECT_DEATH(EncoderEmbeddingGatherVisitor(/*num_computed_tokens=*/{0},
                                             /*num_scheduled_tokens=*/{-1}),
               "negative scheduled token count");
  EXPECT_DEATH(EncoderEmbeddingGatherVisitor(/*num_computed_tokens=*/{0},
                                             /*num_scheduled_tokens=*/{}),
               "computed and scheduled token counts do not match");
  EXPECT_DEATH(
      EncoderEmbeddingGatherVisitor(
          /*num_computed_tokens=*/{0, 0},
          /*num_scheduled_tokens=*/{std::numeric_limits<int32_t>::max(), 1}),
      "scheduled token count exceeds multimodal mask capacity");
}

TEST_F(EncoderExecutorDeathTest, RejectsPrecomputedEmbeddingWithMissingRows) {
  MMDataItem item =
      make_item(MMType::IMAGE, 0, torch::tensor({true, true}), 13);
  item.add("image|embedding", torch::ones({1, 2}));
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 2);
  EncoderExecutor executor(0);
  EXPECT_DEATH(
      executor.prepare(params,
                       [](const ModelInputParams& inputs) -> MMDict {
                         EXPECT_TRUE(inputs.multimodal.mm_data.data().empty());
                         return {};
                       }),
      "embedding rows do not match");
}

TEST_F(EncoderExecutorDeathTest, RejectsEncoderOutputWithMissingRows) {
  MMDataItem item =
      make_item(MMType::IMAGE, 0, torch::tensor({true, true}), 14);
  item.add("pixel_values", torch::ones({2, 2}));
  auto params = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 2);
  EncoderExecutor executor(0);
  EXPECT_DEATH(
      executor.prepare(
          params,
          [](const ModelInputParams& inputs) -> MMDict {
            const torch::Tensor pixels =
                inputs.multimodal.mm_data.get<torch::Tensor>("pixel_values")
                    .value();
            return {{"image|embedding",
                     std::vector<torch::Tensor>{pixels.narrow(0, 0, 1)}}};
          }),
      "embedding rows do not match");
}

TEST(EncoderExecutorTest, ReusesFullItemEncodingAcrossChunks) {
  MMDataItem item =
      make_item(MMType::IMAGE, 1, torch::ones({4}, torch::kBool), 1);
  item.add("pixel_values", torch::ones({4, 2}));
  const MMData request(MMType::IMAGE, MMItemVec{item});
  EncoderExecutor executor(1024);
  int32_t encoder_calls = 0;
  const torch::Tensor full = torch::tensor(
      {{10.0f, 11.0f}, {20.0f, 21.0f}, {30.0f, 31.0f}, {40.0f, 41.0f}});
  const auto encode = [&encoder_calls,
                       &full](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    if (!inputs.has("pixel_values")) {
      return {};
    }
    ++encoder_calls;
    return {{"image|embedding", std::vector<torch::Tensor>{full}}};
  };
  auto first = schedule(request, 0, 3);
  executor.prepare(first, encode);
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      first.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      full.slice(0, 0, 2)));
  EXPECT_TRUE(torch::equal(
      first.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({false, true, true})));

  auto second = schedule(request, 3, 3);
  executor.prepare(second, encode);
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(
      second.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      full.slice(0, 2, 4)));
  EXPECT_TRUE(torch::equal(
      second.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({true, true, false})));
}

TEST(EncoderExecutorTest, EncodesOnlyCacheMissesInMixedModalityBatch) {
  MMDataItem image = make_item(MMType::IMAGE, 0, torch::tensor({true}), 7);
  image.add("pixel_values", torch::tensor({{1.0f, 2.0f}}));
  MMDataItem video =
      make_item(MMType::VIDEO, 1, torch::tensor({true, false, true}), 8);
  video.add("pixel_values_videos", torch::tensor({{3.0f, 4.0f}, {5.0f, 6.0f}}));
  EncoderExecutor executor(1024);
  int32_t image_encodes = 0;
  int32_t video_encodes = 0;
  const auto encode = [&image_encodes, &video_encodes](
                          const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    MMDict outputs;
    if (inputs.has("pixel_values")) {
      ++image_encodes;
      outputs["image|embedding"] = std::vector<torch::Tensor>{
          inputs.get<torch::Tensor>("pixel_values").value() * 10};
    }
    if (inputs.has("pixel_values_videos")) {
      ++video_encodes;
      outputs["video|embedding"] = std::vector<torch::Tensor>{
          inputs.get<torch::Tensor>("pixel_values_videos").value() * 10};
    }
    return outputs;
  };
  auto warmup = schedule(MMData(MMType::IMAGE, MMItemVec{image}), 0, 1);
  executor.prepare(warmup, encode);
  EXPECT_EQ(image_encodes, 1);
  EXPECT_EQ(video_encodes, 0);

  const MMData request(MMType::IMAGE | MMType::VIDEO, MMItemVec{image, video});
  auto mixed = schedule(request, 0, 4);
  executor.prepare(mixed, encode);
  EXPECT_EQ(image_encodes, 1);
  EXPECT_EQ(video_encodes, 1);
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("image|embedding").value(),
      torch::tensor({{10.0f, 20.0f}})));
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("video|embedding").value(),
      torch::tensor({{30.0f, 40.0f}, {50.0f, 60.0f}})));
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({true, false, false, false})));
  EXPECT_TRUE(torch::equal(
      mixed.multimodal.mm_data.get<torch::Tensor>("video|mask").value(),
      torch::tensor({false, true, false, true})));

  auto cached = schedule(request, 0, 4);
  executor.prepare(cached, encode);
  EXPECT_EQ(image_encodes, 1);
  EXPECT_EQ(video_encodes, 1);
  EXPECT_TRUE(torch::equal(
      cached.multimodal.mm_data.get<torch::Tensor>("video|embedding").value(),
      mixed.multimodal.mm_data.get<torch::Tensor>("video|embedding").value()));
}

TEST(EncoderExecutorTest, ReencodesEvictedItem) {
  const torch::Tensor first_features = torch::tensor({{1.0f, 2.0f}});
  const torch::Tensor second_features = torch::tensor({{3.0f, 4.0f}});
  MMDataItem first = make_item(MMType::IMAGE, 0, torch::tensor({true}), 9);
  first.add("pixel_values", first_features);
  MMDataItem second = make_item(MMType::IMAGE, 0, torch::tensor({true}), 10);
  second.add("pixel_values", second_features);
  EncoderExecutor executor(static_cast<int64_t>(first_features.nbytes()));
  int32_t encoder_calls = 0;
  const auto encode =
      [&encoder_calls](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    if (!inputs.has("pixel_values")) {
      return {};
    }
    ++encoder_calls;
    return {{"image|embedding",
             std::vector<torch::Tensor>{
                 inputs.get<torch::Tensor>("pixel_values").value() * 10}}};
  };
  const auto prepare = [&executor, &encode](const MMDataItem& item) {
    auto batch = schedule(MMData(MMType::IMAGE, MMItemVec{item}), 0, 1);
    executor.prepare(batch, encode);
    return batch.multimodal.mm_data.get<torch::Tensor>("image|embedding")
        .value();
  };
  EXPECT_TRUE(torch::equal(prepare(first), first_features * 10));
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(prepare(first), first_features * 10));
  EXPECT_EQ(encoder_calls, 1);
  EXPECT_TRUE(torch::equal(prepare(second), second_features * 10));
  EXPECT_EQ(encoder_calls, 2);
  EXPECT_TRUE(torch::equal(prepare(first), first_features * 10));
  EXPECT_EQ(encoder_calls, 3);
}

TEST(EncoderExecutorTest, CombinesPrecomputedAndEncodedItemsInPromptOrder) {
  MMDataItem video =
      make_item(MMType::VIDEO, 0, torch::tensor({true, false, true}), 2);
  video.add("video|embedding", torch::tensor({{21.0f, 22.0f}, {23.0f, 24.0f}}));
  MMDataItem image = make_item(MMType::IMAGE, 3, torch::tensor({true}), 3);
  image.add("pixel_values", torch::ones({1, 2}));
  auto batch = schedule(
      MMData(MMType::IMAGE | MMType::VIDEO, MMItemVec{video, image}), 0, 4);
  EncoderExecutor executor(0);
  executor.prepare(batch, [](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    EXPECT_FALSE(inputs.has("video|embedding"));
    EXPECT_TRUE(inputs.has("pixel_values"));
    return {{"image|embedding",
             std::vector<torch::Tensor>{torch::tensor({{11.0f, 12.0f}})}}};
  });
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("video|mask").value(),
      torch::tensor({true, false, true, false})));
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("image|mask").value(),
      torch::tensor({false, false, false, true})));
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("video|embedding").value(),
      torch::tensor({{21.0f, 22.0f}, {23.0f, 24.0f}})));
}

TEST(EncoderExecutorTest, AudioGapChunkHasNoReplacementRows) {
  MMDataItem audio =
      make_item(MMType::AUDIO, 0, torch::tensor({true, false, false, true}), 4);
  audio.add("audio|embedding", torch::tensor({{10.0f, 11.0f}, {20.0f, 21.0f}}));
  auto batch = schedule(MMData(MMType::AUDIO, MMItemVec{audio}), 1, 2);
  EncoderExecutor executor(0);
  executor.prepare(batch, [](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    EXPECT_TRUE(inputs.data().empty());
    return {};
  });
  EXPECT_EQ(batch.multimodal.mm_data.get<torch::Tensor>("audio|embedding")
                .value()
                .size(0),
            0);
  EXPECT_TRUE(torch::equal(
      batch.multimodal.mm_data.get<torch::Tensor>("audio|mask").value(),
      torch::tensor({false, false})));
}

TEST(EncoderExecutorTest, PreservesMaskRowsAcrossItemsOfTheSameModality) {
  MMDataItem first =
      make_item(MMType::IMAGE, 0, torch::tensor({true, false}), 5);
  MMDataItem second = make_item(MMType::IMAGE, 2, torch::tensor({true}), 6);
  first.add("image|embedding", torch::tensor({{1.0f, 2.0f}}));
  second.add("image|embedding", torch::tensor({{3.0f, 4.0f}}));
  auto batch = schedule(MMData(MMType::IMAGE, MMItemVec{first, second}), 0, 3);
  EncoderExecutor executor(0);
  executor.prepare(batch, [](const ModelInputParams& params) -> MMDict {
    const auto& inputs = params.multimodal.mm_data;
    EXPECT_TRUE(inputs.data().empty());
    return {};
  });
  const auto mask =
      batch.multimodal.mm_data.get<torch::Tensor>("image|mask").value();
  const auto embeddings =
      batch.multimodal.mm_data.get<torch::Tensor>("image|embedding").value();
  EXPECT_EQ(mask.device(), embeddings.device());
  EXPECT_EQ(mask.scalar_type(), torch::kBool);
  EXPECT_TRUE(torch::equal(mask, torch::tensor({true, false, true})));
  EXPECT_TRUE(
      torch::equal(embeddings, torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}})));
}

}  // namespace
}  // namespace xllm
