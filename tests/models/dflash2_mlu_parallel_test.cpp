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

#include <optional>
#include <string>
#include <unordered_map>

#include "core/framework/parallel_state/collective_communicator.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/layers/mlu/deepseek_v32/multi_device_test_utils.h"
#include "core/layers/mlu/dflash2_decoder_layer.h"
#include "core/platform/device.h"
#include "core/runtime/dflash/dflash_worker_impl.h"
#include "core/runtime/llm_worker_impl.h"

namespace xllm {

class DFlashWorkerImplTestPeer final {
 public:
  static void set_draft(DFlashWorkerImpl& worker,
                        const ParallelArgs& draft,
                        const torch::Device& device,
                        const runtime::Options& options) {
    worker.draft_impl_ =
        std::make_unique<LLMWorkerImpl>(draft, device, options);
  }

  static void synchronize(DFlashWorkerImpl& worker, torch::Tensor& tokens) {
    worker.maybe_broadcast_spec_tokens(tokens);
  }
};

namespace {

ModelArgs make_args() {
  ModelArgs args;
  args.model_type("DFlash2DraftModel")
      .n_layers(1)
      .hidden_size(128)
      .intermediate_size(256)
      .hidden_act("silu")
      .head_dim(128)
      .n_heads(8)
      .n_kv_heads(2)
      .attention_bias(false)
      .max_position_embeddings(32)
      .rope_theta(10000)
      .rms_norm_eps(1e-5)
      .sliding_window(16)
      .dflash2_block_size(4)
      .dflash2_conv_group_size(16)
      .dflash2_conv_kernel_size(2);
  return args;
}

torch::Tensor make_weight(int64_t rows, int64_t cols, int64_t offset) {
  return ((torch::arange(rows * cols, torch::kFloat32) + offset).remainder(31) -
          15)
             .reshape({rows, cols}) /
         256;
}

StateDict make_weights(const torch::TensorOptions& options) {
  std::unordered_map<std::string, torch::Tensor> weights;
  weights.emplace("self_attn.q_proj.weight", make_weight(1024, 128, 1));
  weights.emplace("self_attn.k_proj.weight", make_weight(256, 128, 3));
  weights.emplace("self_attn.v_proj.weight", make_weight(256, 128, 7));
  weights.emplace("self_attn.o_proj.weight", make_weight(128, 1024, 11));
  weights.emplace("mlp.gate_proj.weight", make_weight(256, 128, 13));
  weights.emplace("mlp.up_proj.weight", make_weight(256, 128, 17));
  weights.emplace("mlp.down_proj.weight", make_weight(128, 256, 19));
  for (const std::string& norm : {"self_attn.q_norm",
                                  "self_attn.k_norm",
                                  "input_layernorm",
                                  "post_attention_layernorm"}) {
    weights.emplace(norm + ".weight", torch::ones({128}, options));
  }
  for (const std::string& conv : {"attention_conv", "mlp_conv"}) {
    weights.emplace(conv + ".base_kernel",
                    torch::full({2, 2, 128}, 0.25, options));
    weights.emplace(conv + ".kernel_projection.weight",
                    make_weight(32, 128, 23));
  }
  for (auto& [name, weight] : weights) {
    weight = weight.to(options);
  }
  return StateDict(weights);
}

void compare_decoder(int32_t rank, int32_t port, int32_t cp_size) {
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  CollectiveCommunicator communicator(rank,
                                      /*world_size=*/8,
                                      /*dp_size=*/1,
                                      /*ep_size=*/1,
                                      cp_size,
                                      /*buffer_offset=*/0,
                                      /*needs_tp_cp_group=*/true);
  communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                     device.unwrap());
  const ParallelArgs& target = *communicator.parallel_args();
  const ParallelArgs draft =
      parallel_state::get_cp_independent_tp_parallel_args(target);
  ASSERT_EQ(target.cp_size(), cp_size);
  ASSERT_EQ(target.tp_group_->world_size(), 8 / cp_size);
  ASSERT_EQ(draft.cp_size(), 1);
  ASSERT_EQ(draft.tp_group_->world_size(), 8);
  ASSERT_EQ(draft.tp_group_->rank(), rank);

  const ModelArgs args = make_args();
  ModelContext context(draft, args, QuantArgs(), options);
  layer::mlu::DFlash2DecoderLayer decoder(context, /*layer_id=*/0);
  const StateDict weights = make_weights(options);
  decoder->load_state_dict(weights);
  decoder->verify_loaded_weights("layers.0.");

  ProcessGroup single(/*rank=*/0, /*world_size=*/1, device.unwrap());
  ParallelArgs reference_parallel(/*rank=*/0, /*world_size=*/1, &single);
  reference_parallel.tp_group_ = &single;
  ModelContext reference_context(
      reference_parallel, args, QuantArgs(), options);
  layer::mlu::DFlash2DecoderLayer reference(reference_context, /*layer_id=*/0);
  reference->load_state_dict(weights);
  reference->verify_loaded_weights("layers.0.");

  const auto ints = options.dtype(torch::kInt32);
  layer::AttentionMetadata metadata{};
  metadata.is_dummy = false;
  metadata.is_chunked_prefill = true;
  metadata.max_query_len = 4;
  metadata.max_seq_len = 4;
  metadata.q_cu_seq_lens = torch::tensor({0, 4}, ints);
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.block_table = torch::tensor({0}, ints).reshape({1, 1});
  metadata.slot_mapping = torch::arange(4, ints);
  metadata.is_causal = false;
  metadata.non_causal_window_right = 3;
  KVCache cache(KVCacheTensors{torch::zeros({1, 1, 16, 128}, options),
                               torch::zeros({1, 1, 16, 128}, options)});
  KVCache reference_cache(
      KVCacheTensors{torch::zeros({1, 2, 16, 128}, options),
                     torch::zeros({1, 2, 16, 128}, options)});
  torch::Tensor positions = torch::arange(4, ints);
  torch::Tensor actual = make_weight(4, 128, 29).to(options);
  torch::Tensor expected = actual.clone();
  std::optional<torch::Tensor> actual_residual;
  std::optional<torch::Tensor> expected_residual;
  ModelInputParams input;
  actual = decoder->forward(
      actual, actual_residual, positions, metadata, cache, input);
  expected = reference->forward(
      expected, expected_residual, positions, metadata, reference_cache, input);
  ASSERT_TRUE(actual_residual.has_value());
  ASSERT_TRUE(expected_residual.has_value());
  EXPECT_TRUE(torch::allclose(actual, expected, 0.03, 0.003));
  // BF16 rounds the eight partial o-projection matmuls before their sum.
  // Bound both aggregate and peak error relative to the reference scale;
  // elementwise relative error is unstable near residual cancellations.
  const torch::Tensor reference_residual =
      expected_residual->to(torch::kFloat32);
  const torch::Tensor residual_error =
      actual_residual->to(torch::kFloat32) - reference_residual;
  const float reference_norm = reference_residual.norm().item<float>();
  ASSERT_GT(reference_norm, 0.0f);
  EXPECT_LT(residual_error.norm().item<float>() / reference_norm, 0.01f);
  EXPECT_LT(residual_error.abs().max().item<float>(),
            0.01f * reference_residual.abs().max().item<float>());
  // TP8 replicates each of the two KV heads across four adjacent ranks.
  EXPECT_TRUE(torch::allclose(
      cache.get_k_cache(),
      reference_cache.get_k_cache().slice(/*dim=*/1, rank / 4, rank / 4 + 1),
      0.03,
      0.003));
  EXPECT_TRUE(torch::allclose(
      cache.get_v_cache(),
      reference_cache.get_v_cache().slice(/*dim=*/1, rank / 4, rank / 4 + 1),
      0.03,
      0.003));

  // The draft sampler must share noise over the whole draft TP domain.
  torch::Tensor noise = torch::full({2, 3, 4}, rank, options);
  draft.tp_group_->broadcast(noise, /*root_rank=*/0);
  EXPECT_EQ(noise.abs().max().item<float>(), 0.0f);
}

void check_cp_dp_groups(int32_t rank, int32_t port, bool needs_tp_cp_group) {
  Device device(rank);
  device.set_device();
  CollectiveCommunicator communicator(rank,
                                      /*world_size=*/4,
                                      /*dp_size=*/2,
                                      /*ep_size=*/4,
                                      /*cp_size=*/2,
                                      /*buffer_offset=*/0,
                                      needs_tp_cp_group);
  communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                     device.unwrap());
  const ParallelArgs& target = *communicator.parallel_args();
  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device.unwrap());
  if (needs_tp_cp_group) {
    const ParallelArgs draft =
        parallel_state::get_cp_independent_tp_parallel_args(target);
    EXPECT_EQ(draft.tp_size(), 2);
    EXPECT_EQ(draft.cp_size(), 1);
    torch::Tensor value = torch::full({1}, rank + 1, options);
    draft.tp_group_->allreduce(value);
    EXPECT_EQ(value.item<float>(), rank < 2 ? 3.0f : 7.0f);
  } else {
    EXPECT_EQ(target.tp_cp_group_, nullptr);
  }
  // Both port layouts must leave DP and the subsequent EP group usable.
  torch::Tensor dp_value = torch::ones({1}, options);
  target.dp_local_process_group_->allreduce(dp_value);
  EXPECT_EQ(dp_value.item<float>(), 2.0f);
  torch::Tensor ep_value = torch::full({1}, rank + 1, options);
  target.moe_ep_group_->allreduce(ep_value);
  EXPECT_EQ(ep_value.item<float>(), 10.0f);
}

TEST(DFlash2MluParallelTest, CpDpTargetWithoutDraftSkipsDraftTpGroup) {
  test::run_multi_device(/*ranks=*/4, [](int32_t rank, int32_t port) {
    check_cp_dp_groups(rank, port, /*needs_tp_cp_group=*/false);
  });
}

TEST(DFlash2MluParallelTest, CpDpDraftTpGroupKeepsDpBatchesIsolated) {
  test::run_multi_device(/*ranks=*/4, [](int32_t rank, int32_t port) {
    check_cp_dp_groups(rank, port, /*needs_tp_cp_group=*/true);
  });
}

TEST(DFlash2MluParallelTest, ValidationTokensAgreeAcrossFormerCpRanks) {
  test::run_multi_device(/*ranks=*/8, [](int32_t rank, int32_t port) {
    Device device(rank);
    device.set_device();
    CollectiveCommunicator communicator(rank,
                                        /*world_size=*/8,
                                        /*dp_size=*/1,
                                        /*ep_size=*/1,
                                        /*cp_size=*/4,
                                        /*buffer_offset=*/0,
                                        /*needs_tp_cp_group=*/true);
    communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                       device.unwrap());
    const ParallelArgs target = *communicator.parallel_args();
    const ParallelArgs draft =
        parallel_state::get_cp_independent_tp_parallel_args(target);
    // Construction requires no checkpoint for CP1; inject the resolved draft
    // worker to exercise the production validation broadcast independently of
    // model loading. Keep the target's two-rank group to detect the regression.
    ParallelArgs construction = target;
    construction.cp_size(1);
    runtime::Options options;
    options.speculative_algorithm("DFlash2").num_speculative_tokens(3);
    DFlashWorkerImpl worker(
        construction, device.unwrap(), options, WorkerType::LLM);
    worker.context_ = ModelContext(
        target,
        make_args(),
        QuantArgs(),
        torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap()));
    DFlashWorkerImplTestPeer::set_draft(
        worker, draft, device.unwrap(), options);
    // Different rejection decisions encode both accepted length and bonus ID.
    torch::Tensor tokens =
        torch::tensor({rank, rank % 2 == 0 ? -1 : rank + 10}, torch::kInt32)
            .to(device.unwrap());
    DFlashWorkerImplTestPeer::synchronize(worker, tokens);
    EXPECT_TRUE(torch::equal(tokens.to(torch::kCPU),
                             torch::tensor({0, -1}, torch::kInt32)));
  });
}

TEST(DFlash2MluParallelTest, TargetCp8DraftTp8MatchesUnshardedDecoder) {
  test::run_multi_device(/*ranks=*/8, [](int32_t rank, int32_t port) {
    compare_decoder(rank, port, /*cp_size=*/8);
  });
}

TEST(DFlash2MluParallelTest, TargetCp4Tp2DraftTp8MatchesUnshardedDecoder) {
  test::run_multi_device(/*ranks=*/8, [](int32_t rank, int32_t port) {
    compare_decoder(rank, port, /*cp_size=*/4);
  });
}

}  // namespace
}  // namespace xllm
