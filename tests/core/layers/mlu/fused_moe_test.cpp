/* Copyright 2025-2026 The xLLM Authors.

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

#include "layers/mlu/fused_moe.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>

#include "framework/model/model_args.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/ops_api.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/model_stream_registry.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
namespace {

class CountingReduceProcessGroup : public test::MockProcessGroup {
 public:
  CountingReduceProcessGroup(const torch::Device& device, int64_t size)
      : test::MockProcessGroup(device, 0, size) {}
  void allreduce(torch::Tensor& input) override {
    ++calls;
    input.mul_(world_size());
  }
  int32_t calls = 0;
};

TEST(FusedMoEBufferTest, ExpandWritesOnlyRequestedOutputSlice) {
  auto options = torch::TensorOptions()
                     .device(Platform::type_torch(), 0)
                     .dtype(torch::kBFloat16);
  auto input = torch::arange(8 * 256, options).view({8, 256});
  auto indices = torch::tensor({7, 0, 3, 3, 1, 5, 2, 6}, torch::kInt32)
                     .to(options.device());
  auto storage = torch::full({10, 256}, -1.0, options);
  auto output = storage.narrow(0, 1, 8);
  kernel::MoeExpandInputParams params;
  params.input = input;
  params.gather_index = indices;
  params.cusum_token_count = std::nullopt;
  params.start_expert_id = 0;
  params.expert_size = 256;
  auto reference = kernel::moe_expand_input(params);
  params.output = output;
  auto actual = kernel::moe_expand_input(params);
  EXPECT_EQ(actual.data_ptr(), output.data_ptr());
  test::verify_tensor_close(actual, reference);
  test::verify_tensor_close(actual,
                            input.index_select(0, indices.to(torch::kLong)));
  EXPECT_TRUE(storage[0].eq(-1).all().item<bool>());
  EXPECT_TRUE(storage[9].eq(-1).all().item<bool>());
}

class FusedMoETest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "MLU device is required";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    device_guard_ = std::make_unique<c10::DeviceGuard>(
        torch::Device(Platform::type_torch(), 0));
    model_args_ = test::create_default_model_args();
    quant_args_ = test::create_default_quant_args();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
    model_context_ =
        ModelContext(parallel_args_, model_args_, quant_args_, options_);
    routed_comm_stream_ = model_context_.stream_registry()->get(
        ExecutionStreamRole::COMMUNICATION);
    shared_compute_stream_ = model_context_.stream_registry()->get(
        ExecutionStreamRole::AUXILIARY_COMPUTE);
  }

  std::unordered_map<std::string, torch::Tensor> create_default_test_weights(
      int64_t num_experts,
      int64_t hidden_size,
      int64_t intermediate_size) {
    std::unordered_map<std::string, torch::Tensor> weights;
    auto projection = [&](const std::string& prefix, bool down, int8_t value) {
      const int64_t rows = down ? hidden_size : intermediate_size;
      const int64_t columns = down ? intermediate_size : hidden_size;
      weights[prefix + "qweight"] =
          torch::full({rows, columns}, value, options_.dtype(torch::kInt8));
      weights[prefix + "per_channel_scale"] =
          torch::full({rows}, 0.1f, options_.dtype(torch::kFloat32));
      weights[prefix + "smooth"] =
          torch::full({columns}, 0.05f, options_.dtype(torch::kFloat32));
    };
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      const std::string prefix = "experts." + std::to_string(expert) + ".";
      projection(prefix + "gate_proj.", false, 2);
      projection(prefix + "up_proj.", false, 2);
      projection(prefix + "down_proj.", true, 3);
    }
    projection("shared_experts.gate_proj.", false, 1);
    projection("shared_experts.up_proj.", false, 1);
    projection("shared_experts.down_proj.", true, 1);
    weights["gate.weight"] =
        torch::full({num_experts, hidden_size}, 5.0f, options_);
    weights["gate.e_score_correction_bias"] =
        torch::full({num_experts}, 0.1f, options_);
    return weights;
  }

  std::unordered_map<std::string, torch::Tensor>
  create_w4a8_groupwise_test_weights(int64_t num_experts,
                                     int64_t hidden_size,
                                     int64_t intermediate_size,
                                     int64_t group_size) {
    std::unordered_map<std::string, torch::Tensor> weights;
    for (int64_t expert = 0; expert < num_experts; ++expert) {
      for (const std::string name : {"gate_proj", "up_proj", "down_proj"}) {
        const bool down = name == "down_proj";
        const int64_t rows = down ? hidden_size : intermediate_size;
        const int64_t columns = down ? intermediate_size : hidden_size;
        const std::string prefix =
            "experts." + std::to_string(expert) + "." + name + ".";
        // Two equal signed nibbles per byte; group scales, not packed order,
        // distinguish columns. Gate/up smooth factors must be paired.
        weights[prefix + "qweight"] =
            torch::full({rows, columns / 2},
                        name == "up_proj" ? 0x22 : 0x11,
                        options_.dtype(torch::kInt8));
        const auto scale = torch::tensor(down ? std::vector<float>{0.3f, 0.4f}
                                              : std::vector<float>{0.1f, 0.2f},
                                         options_.dtype(torch::kFloat32));
        weights[prefix + "per_channel_scale"] =
            scale.unsqueeze(0).repeat({rows, 1});
        weights[prefix + "smooth"] =
            torch::full({columns}, 0.05f, options_.dtype(torch::kFloat32));
        CHECK_EQ(columns / group_size, 2);
      }
    }
    weights["gate.weight"] =
        torch::full({num_experts, hidden_size}, 5.0f, options_);
    weights["gate.e_score_correction_bias"] =
        torch::full({num_experts}, 0.1f, options_);
    return weights;
  }

  static float bf16(float value) {
    return torch::tensor(value).to(torch::kBFloat16).item<float>();
  }

  float uniform_expert_reference(int64_t hidden,
                                 int64_t intermediate,
                                 float input,
                                 int32_t gate_q,
                                 int32_t down_q,
                                 std::optional<float> limit = std::nullopt) {
    float projected = bf16(hidden * bf16(input) * bf16(0.05f) * gate_q * 0.1f);
    if (limit) {
      projected = std::min(projected, *limit);
    }
    const auto gate = torch::tensor(projected);
    const float activation = bf16((torch::silu(gate) * gate).item<float>());
    return bf16(intermediate * activation * 0.05f * down_q * 0.1f);
  }

  FusedMoE create_fused_moe(int64_t num_experts,
                            int64_t top_k,
                            int64_t num_expert_group,
                            int64_t topk_group,
                            double route_scale,
                            int64_t hidden_size,
                            int64_t intermediate_size,
                            int64_t n_shared_experts = 1,
                            bool is_gated = true,
                            int64_t renormalize = 0,
                            bool enable_result_reduction = true,
                            const std::string& hidden_act = "silu",
                            const std::string& scoring_func = "sigmoid",
                            const std::string& topk_method = "noaux_tc") {
    ModelArgs args = model_args_;
    args.n_routed_experts() = static_cast<int32_t>(num_experts);
    args.num_experts_per_tok() = static_cast<int32_t>(top_k);
    args.n_group() = static_cast<int32_t>(num_expert_group);
    args.topk_group() = static_cast<int32_t>(topk_group);
    args.routed_scaling_factor() = static_cast<float>(route_scale);
    args.hidden_size() = hidden_size;
    args.moe_intermediate_size() = static_cast<int32_t>(intermediate_size);
    args.n_shared_experts() = static_cast<int32_t>(n_shared_experts);
    args.norm_topk_prob() = (renormalize != 0);
    args.hidden_act() = hidden_act;
    args.scoring_func() = scoring_func;
    args.topk_method() = topk_method;
    const FusedMoEArgs moe_args{
        .is_gated = is_gated,
        .enable_result_reduction = enable_result_reduction};
    return FusedMoE(FusedMoEImpl(args,
                                 moe_args,
                                 quant_args_,
                                 parallel_args_,
                                 options_,
                                 routed_comm_stream_,
                                 shared_compute_stream_));
  }

  void set_tp_ctx(int64_t world_size) {
    mock_process_group_ = std::make_unique<CountingReduceProcessGroup>(
        options_.device(), world_size);
    single_rank_pg_ = std::make_unique<test::MockProcessGroup>(
        options_.device(), /*rank=*/0, /*world_size=*/1);
    parallel_args_ = ParallelArgs(/*rank=*/0,
                                  world_size,
                                  /*dp_size=*/1,
                                  mock_process_group_.get());
    parallel_args_.process_group_ = mock_process_group_.get();
    parallel_args_.tp_group_ = mock_process_group_.get();
    parallel_args_.single_rank_group_ = single_rank_pg_.get();
  }
  // format)
  std::unordered_map<std::string, torch::Tensor> create_test_weights(
      int64_t num_experts,
      int64_t custom_hidden_size = -1,
      int64_t custom_intermediate_size = -1) {
    int64_t test_hidden_size = (custom_hidden_size > 0)
                                   ? custom_hidden_size
                                   : model_args_.hidden_size();
    int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                         ? custom_intermediate_size
                                         : model_args_.intermediate_size();

    return create_default_test_weights(
        num_experts, test_hidden_size, test_intermediate_size);
  }
  void verify_tensor_close(const torch::Tensor& actual,
                           const torch::Tensor& expected,
                           double rtol = 1e-5,
                           double atol = 1e-8) {
    test::verify_tensor_close(actual, expected, rtol, atol);
    const auto a = actual.cpu().to(torch::kFloat32);
    const auto e = expected.cpu().to(torch::kFloat32);
    const double scale = torch::norm(e).item<double>();
    if (scale == 0.0) {
      EXPECT_TRUE(torch::equal(a, e));
    } else {
      EXPECT_LE(torch::norm(a - e).item<double>() / scale, rtol);
    }
  }
  torch::Tensor create_custom_input(const std::vector<int64_t>& shape,
                                    const std::vector<float>& values) {
    return test::create_custom_input(shape, values, options_);
  }

  std::unique_ptr<c10::DeviceGuard> device_guard_;
  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  ModelContext model_context_;
  std::shared_ptr<Stream> routed_comm_stream_;
  std::shared_ptr<Stream> shared_compute_stream_;

  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::unique_ptr<xllm::ProcessGroup> single_rank_pg_;
};

TEST_F(FusedMoETest, ContextConstructorResolvesModelScopedRoleStreams) {
  ModelArgs args = model_args_;
  args.n_routed_experts() = 4;
  args.num_experts_per_tok() = 2;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.hidden_size() = 256;
  args.moe_intermediate_size() = 128;
  args.n_shared_experts() = 0;
  ModelContext context(parallel_args_, args, quant_args_, options_);
  const FusedMoEArgs moe_args{.is_gated = true};
  FusedMoE first(context, moe_args), second(context, moe_args);
  EXPECT_EQ(
      first->routed_stream(),
      context.stream_registry()->get(ExecutionStreamRole::COMMUNICATION).get());
  EXPECT_EQ(first->shared_stream(),
            context.stream_registry()
                ->get(ExecutionStreamRole::AUXILIARY_COMPUTE)
                .get());
  EXPECT_EQ(first->routed_stream(), second->routed_stream());
  EXPECT_EQ(first->shared_stream(), second->shared_stream());
  EXPECT_NE(first->routed_stream(), first->shared_stream());
}

TEST_F(FusedMoETest, ModelContextReusesStreamsByExecutionRole) {
  const ModelContext context(
      parallel_args_, model_args_, quant_args_, options_);
  const std::shared_ptr<ModelStreamRegistry>& registry =
      context.stream_registry();

  std::shared_ptr<Stream> first_communication =
      registry->get(ExecutionStreamRole::COMMUNICATION);
  std::shared_ptr<Stream> second_communication =
      registry->get(ExecutionStreamRole::COMMUNICATION);
  std::shared_ptr<Stream> auxiliary_compute =
      registry->get(ExecutionStreamRole::AUXILIARY_COMPUTE);

  EXPECT_EQ(first_communication, second_communication);
  EXPECT_NE(first_communication, auxiliary_compute);
}

TEST_F(FusedMoETest, SeparateModelContextsOwnSeparateStreamRegistries) {
  const ModelContext first_context(
      parallel_args_, model_args_, quant_args_, options_);
  const ModelContext second_context(
      parallel_args_, model_args_, quant_args_, options_);

  EXPECT_NE(first_context.stream_registry(), second_context.stream_registry());
  EXPECT_NE(
      first_context.stream_registry()->get(ExecutionStreamRole::COMMUNICATION),
      second_context.stream_registry()->get(
          ExecutionStreamRole::COMMUNICATION));
}

TEST_F(FusedMoETest, CopiedAndDerivedContextsShareStreamRegistry) {
  const ModelContext context(
      parallel_args_, model_args_, quant_args_, options_);
  const ModelContext copied_context = context;
  const ModelContext parallel_context =
      context.with_parallel_args(parallel_args_);
  const ModelContext quant_context = context.with_quant_args(quant_args_);

  EXPECT_EQ(context.stream_registry(), copied_context.stream_registry());
  EXPECT_EQ(context.stream_registry(), parallel_context.stream_registry());
  EXPECT_EQ(context.stream_registry(), quant_context.stream_registry());
}

TEST_F(FusedMoETest, ConcurrentRoleRequestsReturnOneStream) {
  constexpr int32_t kThreadCount = 8;
  const ModelContext context(
      parallel_args_, model_args_, quant_args_, options_);
  const std::shared_ptr<ModelStreamRegistry>& registry =
      context.stream_registry();
  std::mutex mutex;
  std::condition_variable cv;
  int arrived = 0;
  std::promise<void> start;
  const auto ready = start.get_future().share();
  std::vector<std::future<std::shared_ptr<Stream>>> futures;
  futures.reserve(kThreadCount);
  for (int32_t i = 0; i < kThreadCount; ++i) {
    futures.emplace_back(std::async(
        std::launch::async, [registry, ready, &mutex, &cv, &arrived]() {
          {
            std::lock_guard<std::mutex> lock(mutex);
            ++arrived;
          }
          cv.notify_one();
          ready.wait();
          return registry->get(ExecutionStreamRole::COMMUNICATION);
        }));
  }

  bool all_arrived;
  {
    std::unique_lock<std::mutex> lock(mutex);
    all_arrived = cv.wait_for(lock, std::chrono::seconds(30), [&] {
      return arrived == kThreadCount;
    });
  }
  start.set_value();
  ASSERT_TRUE(all_arrived);
  std::shared_ptr<Stream> expected = futures.front().get();
  for (int32_t i = 1; i < kThreadCount; ++i) {
    EXPECT_EQ(expected, futures[i].get());
  }
}

TEST_F(FusedMoETest, PrecisionVerificationTest) {
  constexpr int64_t kHidden = 256, kIntermediate = 256, kExperts = 4;
  auto moe = create_fused_moe(
      kExperts, 2, 1, 1, 2.5, kHidden, kIntermediate, 1, true, 1);
  moe->load_state_dict(
      StateDict(create_test_weights(kExperts, kHidden, kIntermediate)));
  const auto actual =
      moe->forward_experts(torch::full({2, kHidden}, 0.05f, options_), false);
  // The uniform router is normalized: sum(route weights) = 2.5. Model BF16
  // projection/activation/output boundaries; this reproduces old 7168/2048
  // gold 992 independently without allocating a multi-GiB fixture.
  const float routed =
      uniform_expert_reference(kHidden, kIntermediate, 0.05f, 2, 3);
  const float shared =
      uniform_expert_reference(kHidden, kIntermediate, 0.05f, 1, 1);
  const auto expected = torch::full(
      {2, kHidden}, bf16(bf16(routed * 2.5f) + shared), torch::kBFloat16);
  verify_tensor_close(actual.cpu(), expected, 0.02, 0.001);
}

TEST_F(FusedMoETest, W4A8GroupwiseSmokeTest) {
  const int64_t batch_size = 4;
  const int64_t seq_len = 8;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t group_size = 128;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  quant_args_.moe_weight_bits() = 4;
  quant_args_.group_size() = group_size;

  auto fused_moe = create_fused_moe(num_experts,
                                    top_k,
                                    num_expert_group,
                                    topk_group,
                                    route_scale,
                                    hidden_size,
                                    intermediate_size,
                                    /*n_shared_experts=*/0,
                                    /*is_gated=*/true,
                                    /*renormalize=*/1);

  auto weight_dict = create_w4a8_groupwise_test_weights(
      num_experts, hidden_size, intermediate_size, group_size);
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  auto output =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  EXPECT_EQ(output.dim(), 2);
  EXPECT_EQ(output.size(0), batch_size * seq_len);
  EXPECT_EQ(output.size(1), hidden_size);
  const float g = bf16(hidden_size * bf16(0.05f) * bf16(0.05f) * 0.15f);
  const float u = bf16(hidden_size * bf16(0.05f) * bf16(0.05f) * 2.0f * 0.15f);
  const float activation =
      bf16((torch::silu(torch::tensor(g)) * u).item<float>());
  const float value =
      bf16(bf16(intermediate_size * activation * 0.05f * 0.35f) * route_scale);
  verify_tensor_close(
      output.cpu(),
      torch::full({batch_size * seq_len, hidden_size}, value, torch::kBFloat16),
      0.03,
      0.001);
}

TEST_F(FusedMoETest, SelectedRouteMatchesGateRouteWith3DInput) {
  const int64_t batch_size = 4;
  const int64_t seq_len = 8;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  FusedMoE fused_moe = create_fused_moe(num_experts,
                                        top_k,
                                        num_expert_group,
                                        topk_group,
                                        route_scale,
                                        hidden_size,
                                        intermediate_size,
                                        /*n_shared_experts=*/0);

  std::unordered_map<std::string, torch::Tensor> weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  torch::Tensor hidden_states = create_custom_input(
      {batch_size, seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({batch_size * seq_len, hidden_size});
  FusedMoEImpl::RouteInfo route_info = fused_moe->prep_route(hidden_states_2d);
  route_info.expert_id = route_info.expert_id.to(torch::kLong);

  torch::Tensor expected =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false);
  torch::Tensor actual = fused_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false, route_info);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  EXPECT_EQ(actual.dim(), hidden_states.dim());
  EXPECT_EQ(actual.size(0), batch_size);
  EXPECT_EQ(actual.size(1), seq_len);
  EXPECT_EQ(actual.size(2), hidden_size);
  verify_tensor_close(actual, expected, 1e-3, 1e-4);
}

TEST_F(FusedMoETest, SelectedRouteRejectsInvalidRouteShape) {
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  FusedMoE fused_moe = create_fused_moe(num_experts,
                                        top_k,
                                        num_expert_group,
                                        topk_group,
                                        route_scale,
                                        hidden_size,
                                        intermediate_size,
                                        /*n_shared_experts=*/0);

  torch::Tensor hidden_states = create_custom_input(
      {3, hidden_size}, std::vector<float>(3 * hidden_size, 0.05f));
  FusedMoEImpl::RouteInfo bad_topk_route{
      .reduce_weight = torch::ones({3, top_k + 1}, options_),
      .expert_id = torch::zeros({3, top_k + 1}, options_.dtype(torch::kInt32))};
  FusedMoEImpl::RouteInfo bad_token_route{
      .reduce_weight = torch::ones({2, top_k}, options_),
      .expert_id = torch::zeros({2, top_k}, options_.dtype(torch::kInt32))};

  EXPECT_DEATH(
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false,
                                 bad_topk_route),
      "topk mismatch");

  EXPECT_DEATH(
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false,
                                 bad_token_route),
      "token count mismatch");
}

TEST_F(FusedMoETest, NoReductionModeMatchesExternalReduce) {
  set_tp_ctx(/*world_size=*/2);

  const int64_t batch_size = 4;
  const int64_t seq_len = 8;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 256;
  const int64_t num_experts = 8;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;

  auto reduced_moe = create_fused_moe(num_experts,
                                      top_k,
                                      num_expert_group,
                                      topk_group,
                                      route_scale,
                                      hidden_size,
                                      intermediate_size,
                                      /*n_shared_experts=*/1,
                                      /*is_gated=*/true,
                                      /*renormalize=*/0,
                                      /*enable_result_reduction=*/true);
  auto raw_moe = create_fused_moe(num_experts,
                                  top_k,
                                  num_expert_group,
                                  topk_group,
                                  route_scale,
                                  hidden_size,
                                  intermediate_size,
                                  /*n_shared_experts=*/1,
                                  /*is_gated=*/true,
                                  /*renormalize=*/0,
                                  /*enable_result_reduction=*/false);

  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  StateDict state_dict(weight_dict);
  reduced_moe->load_state_dict(state_dict);
  raw_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  auto expected = reduced_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  auto* counted =
      static_cast<CountingReduceProcessGroup*>(mock_process_group_.get());
  ASSERT_EQ(counted->calls, 1);
  auto actual = raw_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);
  EXPECT_EQ(counted->calls, 1);
  actual = parallel_state::reduce(actual, parallel_args_.tp_group_);
  EXPECT_EQ(counted->calls, 2);

  auto shared = raw_moe->forward_shared(hidden_states);
  ASSERT_TRUE(shared.defined());
  ASSERT_EQ(raw_moe->shared_pg(), parallel_args_.single_rank_group_);
  shared = parallel_state::reduce(shared, raw_moe->shared_pg());
  actual = actual + shared;

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  verify_tensor_close(actual, expected, 1e-3, 1e-4);
}

TEST_F(FusedMoETest, DeepSeekV4MoETest) {
  const int64_t batch_size = 2;
  const int64_t seq_len = 4;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 128;
  const int64_t num_experts = 4;
  const int64_t num_expert_group = 2;
  const int64_t topk_group = 2;
  const int64_t top_k = 2;
  const double route_scale = 1.0;
  const float swiglu_limit = 10.0f;

  ModelArgs args = model_args_;
  args.n_routed_experts() = static_cast<int32_t>(num_experts);
  args.num_experts_per_tok() = static_cast<int32_t>(top_k);
  args.n_group() = static_cast<int32_t>(num_expert_group);
  args.topk_group() = static_cast<int32_t>(topk_group);
  args.routed_scaling_factor() = static_cast<float>(route_scale);
  args.hidden_size() = hidden_size;
  args.moe_intermediate_size() = static_cast<int32_t>(intermediate_size);
  args.n_shared_experts() = 1;
  args.hidden_act() = "silu";
  args.model_type() = "deepseek_v4";
  args.scoring_func() = "sqrtsoftplus";
  args.swiglu_limit() = swiglu_limit;

  const FusedMoEArgs moe_args{
      .is_gated = true, .enable_result_reduction = true, .use_hash = false};

  auto fused_moe = FusedMoE(FusedMoEImpl(args,
                                         moe_args,
                                         quant_args_,
                                         parallel_args_,
                                         options_,
                                         routed_comm_stream_,
                                         shared_compute_stream_));

  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  weight_dict["gate.bias"] =
      torch::zeros({num_experts}, options_.dtype(torch::kFloat32));
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 10.0f));

  auto output = fused_moe->forward_experts(
      hidden_states, /*enable_all2all_communication=*/false);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  ASSERT_EQ(output.dim(), 2);
  ASSERT_EQ(output.size(0), batch_size * seq_len);
  ASSERT_EQ(output.size(1), hidden_size);

  const float routed = uniform_expert_reference(
      hidden_size, intermediate_size, 10.0f, 2, 3, swiglu_limit);
  const float shared = uniform_expert_reference(
      hidden_size, intermediate_size, 10.0f, 1, 1, swiglu_limit);
  verify_tensor_close(output.cpu(),
                      torch::full({batch_size * seq_len, hidden_size},
                                  bf16(routed + shared),
                                  torch::kBFloat16),
                      0.02,
                      0.02);
}

TEST_F(FusedMoETest, DeepSeekV4HashMoETest) {
  const int64_t batch_size = 2;
  const int64_t seq_len = 4;
  const int64_t hidden_size = 256;
  const int64_t intermediate_size = 128;
  const int64_t num_experts = 4;
  const int64_t num_expert_group = 2;
  const int64_t topk_group = 2;
  const int64_t top_k = 2;
  const int64_t vocab_size = 1024;
  const double route_scale = 1.0;
  const float swiglu_limit = 10.0f;

  ModelArgs args = model_args_;
  args.n_routed_experts() = static_cast<int32_t>(num_experts);
  args.num_experts_per_tok() = static_cast<int32_t>(top_k);
  args.n_group() = static_cast<int32_t>(num_expert_group);
  args.topk_group() = static_cast<int32_t>(topk_group);
  args.routed_scaling_factor() = static_cast<float>(route_scale);
  args.hidden_size() = hidden_size;
  args.moe_intermediate_size() = static_cast<int32_t>(intermediate_size);
  args.n_shared_experts() = 1;
  args.hidden_act() = "silu";
  args.model_type() = "deepseek_v4";
  args.scoring_func() = "sqrtsoftplus";
  args.swiglu_limit() = swiglu_limit;
  args.vocab_size() = vocab_size;

  const FusedMoEArgs moe_args{
      .is_gated = true, .enable_result_reduction = true, .use_hash = true};

  auto fused_moe = FusedMoE(FusedMoEImpl(args,
                                         moe_args,
                                         quant_args_,
                                         parallel_args_,
                                         options_,
                                         routed_comm_stream_,
                                         shared_compute_stream_));

  auto weight_dict =
      create_test_weights(num_experts, hidden_size, intermediate_size);
  const auto token = torch::arange(vocab_size, options_.dtype(torch::kInt32));
  weight_dict["gate.tid2eid"] =
      torch::stack({token % num_experts, (token + 1) % num_experts}, 1);
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    weight_dict["experts." + std::to_string(expert) + ".down_proj.qweight"]
        .fill_(expert + 3);
  }
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 10.0f));
  const auto input_ids =
      torch::arange(batch_size * seq_len, options_.dtype(torch::kInt32));
  auto output =
      fused_moe->forward_experts(hidden_states,
                                 /*enable_all2all_communication=*/false,
                                 /*route_info=*/std::nullopt,
                                 input_ids);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  ASSERT_EQ(output.dim(), 2);
  ASSERT_EQ(output.size(0), batch_size * seq_len);
  ASSERT_EQ(output.size(1), hidden_size);

  const float shared = uniform_expert_reference(
      hidden_size, intermediate_size, 10.0f, 1, 1, swiglu_limit);
  auto expected =
      torch::empty({batch_size * seq_len, hidden_size}, torch::kBFloat16);
  for (int64_t row = 0; row < batch_size * seq_len; ++row) {
    const int32_t first = row % num_experts, second = (row + 1) % num_experts;
    const float a = uniform_expert_reference(
        hidden_size, intermediate_size, 10.0f, 2, first + 3, swiglu_limit);
    const float b = uniform_expert_reference(
        hidden_size, intermediate_size, 10.0f, 2, second + 3, swiglu_limit);
    expected[row].fill_(bf16(bf16((a + b) * 0.5f) + shared));
  }
  verify_tensor_close(output.cpu(), expected, 0.02, 0.02);
}
}  // namespace
}  // namespace layer
}  // namespace xllm
