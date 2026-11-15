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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "core/kernels/musa/gdn_ops.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/musa_graph_executor_impl.h"

namespace xllm::runtime::musa {
namespace {

ModelInputParams make_decode_params(const std::vector<int32_t>& ids,
                                    const torch::Device& device) {
  const int64_t batch = static_cast<int64_t>(ids.size());
  const auto ints = torch::TensorOptions().dtype(torch::kInt32).device(device);
  ModelInputParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.num_sequences = batch;
  params.meta.actual_num_sequences = batch;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len = 10;
  params.attention.device.q_seq_lens = torch::arange(batch + 1, ints);
  params.attention.device.q_cu_seq_lens = torch::arange(batch + 1, ints);
  params.attention.device.kv_seq_lens = torch::arange(batch + 1, ints) * 10;
  params.attention.device.new_cache_slots = torch::arange(batch, ints) * 64;
  params.attention.device.block_tables =
      torch::arange(batch, ints).unsqueeze(1);
  params.attention.device.paged_kv_indptr = torch::arange(batch + 1, ints);
  params.attention.device.paged_kv_indices = torch::arange(batch, ints);
  params.attention.device.paged_kv_last_page_len =
      torch::full({batch}, 10, ints);
  params.attention.device.kv_cache_tokens_nums = torch::full({batch}, 9, ints);
  params.embedding.linear_state_ids = ids;
  params.embedding.linear_state_indices = torch::tensor(ids, ints);
  params.attention.host.q_seq_lens.reserve(batch + 1);
  params.attention.host.kv_seq_lens.reserve(batch + 1);
  for (int32_t i = 0; i <= batch; ++i) {
    params.attention.host.q_seq_lens.emplace_back(i);
    params.attention.host.kv_seq_lens.emplace_back(i * 10);
  }
  return params;
}

TEST(MusaGraphExecutorTest, ShrinkingDecodeReplayDoesNotUpdatePaddingState) {
  if (Platform::device_count() == 0) {
    GTEST_SKIP() << "No MUSA device available";
  }
  Device device(/*device_index=*/0);
  device.set_device();
  device.init_device_context();
  torch::NoGradGuard no_grad;
  const auto bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  const auto ints = bf16.dtype(torch::kInt32);
  ModelArgs args;
  args.model_type("qwen3_5_text");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.head_dim(256);
  args.n_heads(12);
  args.n_kv_heads(2);
  args.max_position_embeddings(128);
  runtime::Options options;
  options.max_seqs_per_batch(32);
  options.max_tokens_per_batch(16);
  options.block_size(64);
  options.world_size(1);
  MusaGraphPersistentParam persistent(args, device.unwrap(), options);
  const torch::Tensor k_cache = torch::zeros({16, 64, 2, 256}, bf16);
  const torch::Tensor v_cache = torch::zeros_like(k_cache);
  const auto update = [&](const ModelInputParams& input) {
    const int64_t batch = input.meta.actual_num_sequences;
    return persistent.update(torch::ones({batch}, ints),
                             k_cache,
                             v_cache,
                             torch::ones({batch}, ints),
                             input,
                             /*padded_num_tokens=*/8,
                             /*return_capture_params=*/true);
  };
  const auto full =
      make_decode_params({0, 1, 2, 3, 4, 5, 6, 7}, device.unwrap());
  const auto captured = update(full);
  ASSERT_TRUE(captured.has_value());
  const torch::Tensor captured_indices =
      captured->embedding.linear_state_indices;

  kernel::musa::MateGatedDeltaRuleDecodeParams params;
  params.num_k_heads = 1;
  params.num_v_heads = 2;
  params.head_k_dim = 16;
  params.head_v_dim = 16;
  params.scale = 0.25;
  params.use_qk_l2norm = true;
  params.mixed_qkv = torch::full({8, 64}, 0.125, bf16);
  params.mixed_qkv.narrow(0, 5, 3).zero_();
  params.A_log = torch::zeros({2}, bf16.dtype(torch::kFloat32));
  params.dt_bias = torch::zeros_like(params.A_log);
  params.a = torch::zeros({8, 2}, bf16);
  params.b = torch::zeros({8, 2}, bf16);
  const torch::Tensor initial =
      torch::ones({16, 2, 16, 16}, bf16.dtype(torch::kFloat32));
  params.state = initial.clone();
  params.state_indices = captured_indices;
  params.decode_output = torch::empty({8, 2, 16}, bf16);
  kernel::musa::fused_gated_delta_rule_decode(params);
  params.state.copy_(initial);
  device.synchronize_default_stream();
  at::musa::MUSAGraph graph;
  const c10::musa::MUSAStream capture_stream = c10::musa::getStreamFromPool();
  {
    c10::StreamGuard stream_guard(capture_stream.unwrap());
    graph.capture_begin({0, 0}, musaStreamCaptureModeThreadLocal);
    kernel::musa::fused_gated_delta_rule_decode(params);
    graph.capture_end();
  }
  capture_stream.synchronize();
  auto small = make_decode_params({0, 1, 2, 3, 7}, device.unwrap());
  auto reference = params;
  reference.state = initial.clone();
  reference.state_indices = small.embedding.linear_state_indices;
  reference.mixed_qkv = params.mixed_qkv.narrow(0, 0, 5);
  reference.a = params.a.narrow(0, 0, 5);
  reference.b = params.b.narrow(0, 0, 5);
  reference.decode_output = torch::empty({5, 2, 16}, bf16);
  kernel::musa::fused_gated_delta_rule_decode(reference);
  device.synchronize_default_stream();

  // Exercise both sources with stale capacity rows beyond the live prefix.
  small.meta.num_sequences = 8;
  small.embedding.linear_state_ids = {0, 1, 2, 3, 7, 5, 6, 7};
  small.embedding.linear_state_indices =
      torch::tensor(small.embedding.linear_state_ids, ints);
  const torch::Tensor expected_indices =
      torch::tensor({0, 1, 2, 3, 7, -1, -1, -1}, torch::kInt32);
  for (const bool tensor_source : {true, false}) {
    SCOPED_TRACE(tensor_source);
    update(full);
    if (!tensor_source) {
      small.embedding.linear_state_indices = torch::Tensor();
    }
    const auto next = update(small);
    ASSERT_TRUE(next.has_value());
    ASSERT_EQ(next->embedding.linear_state_indices.numel(), 8);
    EXPECT_TRUE(torch::equal(captured_indices.cpu(), expected_indices));
    params.state.copy_(initial);
    graph.replay();
    device.synchronize_default_stream();
    EXPECT_TRUE(torch::equal(params.state.cpu(), reference.state.cpu()))
        << "state max_abs="
        << (params.state - reference.state).abs().max().item<float>();
  }
}

}  // namespace
}  // namespace xllm::runtime::musa
