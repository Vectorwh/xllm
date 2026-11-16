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

#include "core/framework/batch/dit_batch.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace {

std::shared_ptr<DiTRequest> make_dit_request(
    const std::string& request_id,
    DiTRequestKind kind,
    DiTGenerationParams generation_params) {
  DiTInputParams inputs;
  inputs.prompt = request_id;
  DiTRequestState state(std::move(inputs),
                        std::move(generation_params),
                        DiTOutputFunc{},
                        DiTOutputsFunc{},
                        kind);
  return std::make_shared<DiTRequest>(
      request_id, "rid", "rtime", std::move(state));
}

TEST(DiTBatchTest, SingleRequestSourcesUseBatchViews) {
  DiTInputParams input_params;
  torch::Tensor image =
      torch::arange(12, torch::dtype(torch::kUInt8)).reshape({3, 2, 2});
  torch::Tensor prompt_embed = torch::randn({4, 8});
  torch::Tensor prompt_audio = torch::randn({1, 32});
  input_params.media_sources.add("image", "image", image);
  input_params.tensor_sources.add("prompt_embed", prompt_embed);
  input_params.tensor_sources.add("prompt_audio", prompt_audio);

  DiTGenerationParams generation_params;
  DiTOutputFunc output_func;
  DiTOutputsFunc outputs_func;
  DiTRequestState state(std::move(input_params),
                        std::move(generation_params),
                        std::move(output_func),
                        std::move(outputs_func),
                        DiTRequestKind::kImage);
  auto request =
      std::make_shared<DiTRequest>("request", "rid", "rtime", std::move(state));

  DiTBatch batch;
  batch.add(request);
  DiTForwardInput forward_input = batch.prepare_forward_input();

  const torch::Tensor& batched_image = forward_input.media_sources.at(0).tensor;
  const torch::Tensor batched_embed =
      *forward_input.tensor_sources.get("prompt_embed");
  const torch::Tensor batched_audio =
      *forward_input.tensor_sources.get("prompt_audio");

  EXPECT_EQ(batched_image.sizes(), torch::IntArrayRef({1, 3, 2, 2}));
  EXPECT_EQ(batched_embed.sizes(), torch::IntArrayRef({1, 4, 8}));
  EXPECT_EQ(batched_audio.sizes(), torch::IntArrayRef({1, 1, 32}));
  EXPECT_EQ(batched_image.data_ptr(), image.data_ptr());
  EXPECT_EQ(batched_embed.data_ptr(), prompt_embed.data_ptr());
  EXPECT_EQ(batched_audio.data_ptr(), prompt_audio.data_ptr());
}

TEST(DiTBatchTest, ImageBatchChecksWorkerParametersNotResponseEncoding) {
  DiTGenerationParams first;
  first.num_inference_steps = 20;
  DiTGenerationParams second = first;
  second.output_type = "binary";
  DiTBatch compatible;
  compatible.add(make_dit_request("first", DiTRequestKind::kImage, first));
  compatible.add(make_dit_request("second", DiTRequestKind::kImage, second));
  const DiTForwardInput input = compatible.prepare_forward_input();
  EXPECT_EQ(input.batch_size, 2);
  EXPECT_EQ(input.prompts, (std::vector<std::string>{"first", "second"}));
  EXPECT_EQ(input.generation_params.num_inference_steps, 20);

  second.num_inference_steps = 21;
  DiTBatch incompatible;
  incompatible.add(make_dit_request("first", DiTRequestKind::kImage, first));
  incompatible.add(make_dit_request("second", DiTRequestKind::kImage, second));
  EXPECT_DEATH(incompatible.prepare_forward_input(),
               "DiT generation params must be equal");
}

TEST(DiTBatchTest, RejectsMultipleTextRequests) {
  DiTGenerationParams params;
  DiTBatch single;
  single.add(make_dit_request("text", DiTRequestKind::kText, params));
  const DiTForwardInput input = single.prepare_forward_input();
  EXPECT_EQ(input.batch_size, 1);
  EXPECT_EQ(input.prompts, (std::vector<std::string>{"text"}));

  single.add(make_dit_request("other", DiTRequestKind::kText, params));
  EXPECT_DEATH(single.prepare_forward_input(),
               "text generation supports batch_size=1 only");
}

}  // namespace
}  // namespace xllm
