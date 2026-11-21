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

#include <glog/logging.h>

#include <optional>

#include "core/kernels/ops_api.h"

namespace xllm {
namespace layer {
namespace {

torch::Tensor vision_sdpa(const torch::Tensor& query,
                          const torch::Tensor& key,
                          const torch::Tensor& value,
                          float scale) {
  return torch::scaled_dot_product_attention(query,
                                             key,
                                             value,
                                             /*attn_mask=*/std::nullopt,
                                             /*dropout_p=*/0.0,
                                             /*is_causal=*/false,
                                             /*scale=*/scale);
}

}  // namespace

torch::Tensor Qwen3VisionAttentionImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec) {
  (void)cu_seq_len;
  torch::Tensor qkv = qkv_proj_->forward(hidden_states);
  std::vector<torch::Tensor> qkv_split = split_qkv(qkv);
  for (torch::Tensor& tensor : qkv_split) {
    tensor = tensor.transpose(0, 1).contiguous();
  }

  torch::Tensor query = qkv_split[0];
  torch::Tensor key = qkv_split[1];
  torch::Tensor value = qkv_split[2];
  const int64_t batch = query.size(0);
  const int64_t sequence = query.size(1);
  const int64_t head_dim = query.size(3);
  CHECK_EQ(head_dim, hidden_size_per_attention_head_)
      << "Qwen3 vision head_dim mismatch";

  const int64_t total_tokens = batch * sequence;
  const int32_t cu_count = static_cast<int32_t>(cu_seq_len_vec.size());
  CHECK_GE(cu_count, 2) << "Qwen3 vision cu_seq_len is incomplete";
  CHECK_EQ(cu_seq_len_vec.front(), 0)
      << "Qwen3 vision cu_seq_len must start at 0";
  CHECK_EQ(static_cast<int64_t>(cu_seq_len_vec.back()), total_tokens)
      << "Qwen3 vision cu_seq_len must cover every token";
  for (int32_t index = 1; index < cu_count; ++index) {
    CHECK_GE(cu_seq_len_vec[index], cu_seq_len_vec[index - 1])
        << "Qwen3 vision cu_seq_len must be non-decreasing";
  }
  const int32_t max_seqlen = get_max_sequence_length(cu_seq_len_vec);
  query = query.reshape({total_tokens,
                         num_attention_heads_per_partition_,
                         hidden_size_per_attention_head_});
  key = key.reshape({total_tokens,
                     num_attention_heads_per_partition_,
                     hidden_size_per_attention_head_});

  CHECK_EQ(m_cos_pos.size(0), total_tokens)
      << "Qwen3 vision cos token count mismatch";
  CHECK_EQ(m_sin_pos.size(0), total_tokens)
      << "Qwen3 vision sin token count mismatch";

  const torch::ScalarType dtype = hidden_states.scalar_type();
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = query.to(torch::kFloat32);
  rotary_params.k = key.to(torch::kFloat32);
  rotary_params.sin = m_sin_pos.to(torch::kFloat32);
  rotary_params.cos = m_cos_pos.to(torch::kFloat32);
  rotary_params.interleaved = false;
  rotary_params.discrete = false;
  rotary_params.cu_query_lens = std::nullopt;
  rotary_params.max_query_len = max_seqlen;
  xllm::kernel::apply_rotary(rotary_params);
  query = rotary_params.q.to(dtype);
  key = rotary_params.k.to(dtype);

  value = value.view({batch * sequence, value.size(2), value.size(3)});
  torch::Tensor output;
  const int32_t num_sequences = static_cast<int32_t>(cu_seq_len_vec.size()) - 1;
  bool batched_sdpa_done = false;
  if (batch == 1 && num_sequences == 1 && cu_seq_len_vec[0] == 0 &&
      cu_seq_len_vec[1] == total_tokens && total_tokens > 0) {
    output = vision_sdpa(query.permute({1, 0, 2}).unsqueeze(0),
                         key.permute({1, 0, 2}).unsqueeze(0),
                         value.permute({1, 0, 2}).unsqueeze(0),
                         scale_)
                 .squeeze(/*dim=*/0)
                 .permute({1, 0, 2})
                 .contiguous();
    batched_sdpa_done = true;
  } else if (batch == 1 && num_sequences > 1 && cu_seq_len_vec[0] == 0 &&
             cu_seq_len_vec.back() == total_tokens) {
    const int32_t first_len = cu_seq_len_vec[1] - cu_seq_len_vec[0];
    bool equal_lengths = first_len > 0;
    for (int32_t index = 1; index < num_sequences && equal_lengths; ++index) {
      if (cu_seq_len_vec[index + 1] - cu_seq_len_vec[index] != first_len) {
        equal_lengths = false;
      }
    }
    if (equal_lengths) {
      const int64_t sequence_count = num_sequences;
      const int64_t sequence_len = first_len;
      const torch::Tensor query_b =
          query
              .reshape({sequence_count,
                        sequence_len,
                        num_attention_heads_per_partition_,
                        head_dim})
              .transpose(1, 2);
      const torch::Tensor key_b =
          key.reshape({sequence_count,
                       sequence_len,
                       num_attention_heads_per_partition_,
                       head_dim})
              .transpose(1, 2);
      const torch::Tensor value_b =
          value
              .reshape({sequence_count,
                        sequence_len,
                        num_attention_heads_per_partition_,
                        head_dim})
              .transpose(1, 2);
      output =
          vision_sdpa(query_b, key_b, value_b, scale_)
              .transpose(1, 2)
              .reshape(
                  {total_tokens, num_attention_heads_per_partition_, head_dim});
      batched_sdpa_done = true;
    }
  }
  if (!batched_sdpa_done) {
    output = torch::zeros_like(query);
    for (int32_t index = 0; index < num_sequences; ++index) {
      const int32_t start = cu_seq_len_vec[index];
      const int32_t end = cu_seq_len_vec[index + 1];
      if (end <= start) {
        continue;
      }
      const torch::Tensor out_i = vision_sdpa(
          query.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0),
          key.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0),
          value.slice(/*dim=*/0, start, end).permute({1, 0, 2}).unsqueeze(0),
          scale_);
      output.slice(/*dim=*/0, start, end)
          .copy_(out_i.squeeze(/*dim=*/0).permute({1, 0, 2}));
    }
  }

  output = output.view({batch, sequence, -1});
  output = output.transpose(0, 1).reshape({-1, output.size(-1)});
  return proj_->forward(output);
}

}  // namespace layer
}  // namespace xllm
