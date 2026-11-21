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

#include "layers/musa/qwen3_vision_attention.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/quant_args.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/musa/linear_block_fp8.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kHiddenSize = 32;
constexpr int64_t kNumHeads = 4;
constexpr int64_t kHeadDim = 8;

torch::Tensor ApplyNeoxRotary(const torch::Tensor& input,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin) {
  const int64_t half = cos.size(-1) / 2;
  const torch::Tensor cos_half = cos.slice(/*dim=*/-1, 0, half).unsqueeze(1);
  const torch::Tensor sin_half = sin.slice(/*dim=*/-1, 0, half).unsqueeze(1);
  const torch::Tensor left = input.slice(/*dim=*/-1, 0, half);
  const torch::Tensor right = input.slice(/*dim=*/-1, half, half * 2);
  return torch::cat(
      {left * cos_half - right * sin_half, right * cos_half + left * sin_half},
      /*dim=*/-1);
}

torch::Tensor SegmentSoftmaxAttention(const torch::Tensor& query,
                                      const torch::Tensor& key,
                                      const torch::Tensor& value,
                                      const std::vector<int32_t>& cu_seq_len,
                                      float scale) {
  torch::Tensor output = torch::zeros_like(query);
  const int32_t num_sequences = static_cast<int32_t>(cu_seq_len.size()) - 1;
  for (int32_t index = 0; index < num_sequences; ++index) {
    const int32_t start = cu_seq_len[index];
    const int32_t end = cu_seq_len[index + 1];
    if (end <= start) {
      continue;
    }
    const torch::Tensor query_i =
        query.slice(/*dim=*/0, start, end).permute({1, 0, 2});
    const torch::Tensor key_i =
        key.slice(/*dim=*/0, start, end).permute({1, 2, 0});
    const torch::Tensor value_i =
        value.slice(/*dim=*/0, start, end).permute({1, 0, 2});
    const torch::Tensor scores = torch::matmul(query_i, key_i) * scale;
    const torch::Tensor probs = torch::softmax(scores, /*dim=*/-1);
    const torch::Tensor out_i =
        torch::matmul(probs, value_i).permute({1, 0, 2});
    output.slice(/*dim=*/0, start, end).copy_(out_i);
  }
  return output;
}

class Qwen3VisionAttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "MUSA device is required";
    }
    device_ = torch::Device(Platform::type_torch(), 0);
    Device xllm_device(device_);
    xllm_device.set_seed(7);
    options_ = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    ModelArgs model_args;
    model_args.mm_hidden_size() = kHiddenSize;
    model_args.mm_num_attention_heads() = kNumHeads;
    model_args.mm_head_dim() = kHeadDim;
    process_group_ = std::make_unique<ProcessGroup>(0, 1, device_);
    parallel_args_ = ParallelArgs(0, 1, process_group_.get());
    parallel_args_.tp_group_ = process_group_.get();
    context_ = ModelContext(parallel_args_, model_args, QuantArgs(), options_);

    const int64_t qkv_size = kNumHeads * kHeadDim * 3;
    qkv_weight_ = torch::randn({qkv_size, kHiddenSize}, options_) * 0.02;
    qkv_bias_ = torch::randn({qkv_size}, options_) * 0.02;
    proj_weight_ = torch::randn({kHiddenSize, kHiddenSize}, options_) * 0.02;
    proj_bias_ = torch::randn({kHiddenSize}, options_) * 0.02;
    attention_ = Qwen2VisionAttention(
        std::make_shared<Qwen3VisionAttentionImpl>(context_));
    std::unordered_map<std::string, torch::Tensor> weights = {
        {"qkv.weight", qkv_weight_},
        {"qkv.bias", qkv_bias_},
        {"proj.weight", proj_weight_},
        {"proj.bias", proj_bias_},
    };
    attention_->load_state_dict(StateDict(weights));
    scale_ = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  }

  torch::Tensor Reference(torch::Tensor hidden,
                          const std::vector<int32_t>& cu_seq_len) {
    torch::Tensor buffer;
    torch::Tensor qkv =
        musa::matmul_forward(hidden, qkv_weight_, qkv_bias_, buffer);
    const int64_t seq_len = qkv.size(0);
    const int64_t batch = qkv.dim() == 3 ? qkv.size(1) : 1;
    std::vector<torch::Tensor> parts = qkv.chunk(3, /*dim=*/-1);
    const std::vector<int64_t> split_shape = {
        seq_len, batch, kNumHeads, kHeadDim};
    torch::Tensor query = parts[0].reshape(split_shape).transpose(0, 1);
    torch::Tensor key = parts[1].reshape(split_shape).transpose(0, 1);
    torch::Tensor value = parts[2].reshape(split_shape).transpose(0, 1);
    query = query.reshape({batch * query.size(1), kNumHeads, kHeadDim})
                .to(torch::kCPU)
                .to(torch::kFloat32);
    key = key.reshape({batch * key.size(1), kNumHeads, kHeadDim})
              .to(torch::kCPU)
              .to(torch::kFloat32);
    value = value.reshape({batch * value.size(1), kNumHeads, kHeadDim})
                .to(torch::kCPU)
                .to(torch::kFloat32);
    const torch::Tensor cos = cos_.to(torch::kCPU).to(torch::kFloat32);
    const torch::Tensor sin = sin_.to(torch::kCPU).to(torch::kFloat32);
    query = ApplyNeoxRotary(query, cos, sin);
    key = ApplyNeoxRotary(key, cos, sin);

    torch::Tensor attended =
        SegmentSoftmaxAttention(query, key, value, cu_seq_len, scale_)
            .to(hidden.device())
            .to(hidden.scalar_type());
    attended = attended.view({batch, query.size(0) / batch, -1})
                   .transpose(0, 1)
                   .reshape({-1, attended.size(1) * attended.size(2)});
    torch::Tensor proj_buffer;
    return musa::matmul_forward(
        attended, proj_weight_, proj_bias_, proj_buffer);
  }

  void ExpectMatchesReference(const std::vector<int64_t>& hidden_shape,
                              const std::vector<int32_t>& cu_seq_len) {
    torch::Tensor hidden = torch::randn(hidden_shape, options_) * 0.02;
    const int64_t total_tokens = hidden.numel() / kHiddenSize;
    cos_ = torch::randn({total_tokens, kHeadDim}, options_);
    sin_ = torch::randn({total_tokens, kHeadDim}, options_);
    torch::Tensor cu_seq =
        torch::tensor(cu_seq_len, options_.dtype(torch::kInt32));
    std::vector<int32_t> cu_seq_copy = cu_seq_len;

    torch::Tensor actual =
        attention_->forward(hidden, cos_, sin_, cu_seq, cu_seq_copy);
    Device(device_).synchronize_default_stream();
    torch::Tensor expected = Reference(hidden, cu_seq_len);
    Device(device_).synchronize_default_stream();

    ASSERT_EQ(actual.sizes(), expected.sizes());
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
    EXPECT_TRUE(
        torch::allclose(actual, expected, /*rtol=*/1e-4, /*atol=*/1e-4));
  }

  torch::Device device_ = torch::kCPU;
  torch::TensorOptions options_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  std::unique_ptr<ProcessGroup> process_group_;
  ModelContext context_;
  Qwen2VisionAttention attention_{nullptr};
  torch::Tensor qkv_weight_;
  torch::Tensor qkv_bias_;
  torch::Tensor proj_weight_;
  torch::Tensor proj_bias_;
  torch::Tensor cos_;
  torch::Tensor sin_;
  float scale_ = 1.0f;
};

TEST_F(Qwen3VisionAttentionTest, SingleImageMatchesSegmentSdpa) {
  ExpectMatchesReference({6, kHiddenSize}, {0, 6});
}

TEST_F(Qwen3VisionAttentionTest, EqualImagesMatchSegmentSdpa) {
  ExpectMatchesReference({8, kHiddenSize}, {0, 4, 8});
}

TEST_F(Qwen3VisionAttentionTest, UnequalImagesWithEmptySegmentMatchFallback) {
  ExpectMatchesReference({7, kHiddenSize}, {0, 3, 3, 7});
}

TEST_F(Qwen3VisionAttentionTest, BatchGreaterThanOneMatchesSegmentSdpa) {
  ExpectMatchesReference({4, 2, kHiddenSize}, {0, 8});
}

TEST_F(Qwen3VisionAttentionTest, BatchGreaterThanOneEqualSegments) {
  ExpectMatchesReference({4, 2, kHiddenSize}, {0, 4, 8});
}

TEST_F(Qwen3VisionAttentionTest, BatchGreaterThanOneUnequalSegments) {
  ExpectMatchesReference({4, 2, kHiddenSize}, {0, 3, 8});
}

}  // namespace
}  // namespace layer
}  // namespace xllm
