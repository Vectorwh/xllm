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

#include "layers/mlu/dense_mlp.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

class DenseMLPTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() == 0) {
      GTEST_SKIP() << "Accelerator device is required";
    }
    quant_args_ = test::create_default_quant_args();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Platform::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
  }

  std::unordered_map<std::string, torch::Tensor> create_default_test_weights(
      int64_t hidden_size,
      int64_t intermediate_size) {
    auto gate_up_weight =
        torch::full({intermediate_size * 2, hidden_size}, 5.0f, options_);
    auto down_weight =
        torch::full({hidden_size, intermediate_size}, 3.0f, options_);
    auto gate_up_qweight = gate_up_weight.to(torch::kInt8);
    auto down_qweight = down_weight.to(torch::kInt8);
    auto gate_up_scale = torch::full({intermediate_size},
                                     0.1f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));
    auto up_scale = torch::full({intermediate_size},
                                0.1f,
                                torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(options_.device()));
    auto down_scale = torch::full({hidden_size},
                                  0.1f,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(options_.device()));
    auto gate_up_smooth = torch::full({hidden_size},
                                      0.05f,
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(options_.device()));
    auto up_smooth = torch::full({hidden_size},
                                 0.05f,
                                 torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));
    auto down_smooth = torch::full({intermediate_size},
                                   0.05f,
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    weight_dict["gate_proj.qweight"] =
        gate_up_qweight.slice(0, 0, intermediate_size);
    weight_dict["gate_proj.per_channel_scale"] = gate_up_scale;
    weight_dict["gate_proj.smooth"] = gate_up_smooth;
    weight_dict["up_proj.qweight"] =
        gate_up_qweight.slice(0, intermediate_size, intermediate_size * 2);
    weight_dict["up_proj.per_channel_scale"] = up_scale;
    weight_dict["up_proj.smooth"] = up_smooth;
    weight_dict["down_proj.qweight"] = down_qweight;
    weight_dict["down_proj.per_channel_scale"] = down_scale;
    weight_dict["down_proj.smooth"] = down_smooth;

    return weight_dict;
  }
  MluDenseMLP create_mlp(int64_t hidden_size, int64_t intermediate_size) {
    return MluDenseMLP(MluDenseMLPImpl(hidden_size,
                                       intermediate_size,
                                       /*is_gated=*/true,
                                       /*has_bias=*/false,
                                       /*hidden_act=*/"silu",
                                       /*enable_result_reduction=*/true,
                                       quant_args_,
                                       parallel_args_.tp_group_,
                                       options_));
  }

  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
};

TEST_F(DenseMLPTest, Bfloat16LoadStateDictTest) {
  constexpr int64_t kHidden = 128;
  constexpr int64_t kIntermediate = 256;
  MluDenseMLP mlp(kHidden,
                  kIntermediate,
                  true,
                  false,
                  "silu",
                  true,
                  QuantArgs(),
                  parallel_args_.tp_group_,
                  options_);
  const auto gate = torch::full({kIntermediate, kHidden}, 0.01f, options_);
  const auto up = torch::full({kIntermediate, kHidden}, 0.02f, options_);
  const auto down = torch::full({kHidden, kIntermediate}, 0.03f, options_);
  mlp->load_state_dict(StateDict({{"gate_proj.weight", gate},
                                  {"up_proj.weight", up},
                                  {"down_proj.weight", down}}));
  auto input = torch::ones({2, kHidden}, options_);
  const auto actual = mlp->forward(input);
  const auto x = input.cpu().to(torch::kFloat32);
  const auto g = torch::matmul(x, gate.cpu().to(torch::kFloat32).t())
                     .to(torch::kBFloat16)
                     .to(torch::kFloat32);
  const auto u = torch::matmul(x, up.cpu().to(torch::kFloat32).t())
                     .to(torch::kBFloat16)
                     .to(torch::kFloat32);
  const auto activated = (torch::silu(g) * u).to(torch::kBFloat16);
  const auto expected = torch::matmul(activated.to(torch::kFloat32),
                                      down.cpu().to(torch::kFloat32).t())
                            .to(torch::kBFloat16);
  test::verify_tensor_close(actual.cpu(), expected, 1e-2, 1e-3);
}

#if defined(USE_CUDA)
TEST_F(DenseMLPTest, Fp8IgnoredDownProjLoadsAsUnquantized) {
  QuantArgs fp8_quant_args;
  fp8_quant_args.quant_method() = kQuantMethodFp8;
  fp8_quant_args.bits() = 8;
  fp8_quant_args.activation_dynamic() = false;
  fp8_quant_args.ignored_modules() = {"model.layers.1.mlp.down_proj"};

  const int64_t hidden_size = 16;
  const int64_t intermediate_size = 32;
  auto mlp = MluDenseMLP(MluDenseMLPImpl(hidden_size,
                                         intermediate_size,
                                         /*is_gated=*/true,
                                         /*has_bias=*/false,
                                         /*hidden_act=*/"silu",
                                         /*enable_result_reduction=*/true,
                                         fp8_quant_args,
                                         parallel_args_.tp_group_,
                                         options_,
                                         "model.layers.1.mlp"));

  std::unordered_map<std::string, torch::Tensor> weight_dict;
  auto fp8_weight_options = options_.dtype(torch::kFloat8_e4m3fn);
  auto scale_options = options_.dtype(torch::kFloat32);

  weight_dict["gate_proj.weight"] =
      torch::zeros({intermediate_size, hidden_size}, fp8_weight_options);
  weight_dict["gate_proj.weight_scale"] = torch::ones({1}, scale_options);
  weight_dict["gate_proj.input_scale"] = torch::ones({1}, scale_options);

  weight_dict["up_proj.weight"] =
      torch::zeros({intermediate_size, hidden_size}, fp8_weight_options);
  weight_dict["up_proj.weight_scale"] = torch::ones({1}, scale_options);
  weight_dict["up_proj.input_scale"] = torch::ones({1}, scale_options);

  weight_dict["down_proj.weight"] =
      torch::zeros({hidden_size, intermediate_size}, options_);

  StateDict state_dict(weight_dict);
  mlp->load_state_dict(state_dict);

  const auto params = mlp->named_parameters(/*recurse=*/true);
  EXPECT_TRUE(params.contains("gate_up_proj.weight_scale"));
  EXPECT_TRUE(params.contains("gate_up_proj.input_scale"));
  EXPECT_TRUE(params.contains("down_proj.weight"));
  EXPECT_FALSE(params.contains("down_proj.weight_scale"));
  EXPECT_FALSE(params.contains("down_proj.input_scale"));
}
#endif

TEST_F(DenseMLPTest, SmoothquantPrecisionVerificationTest) {
  constexpr int64_t kHidden = 128;
  constexpr int64_t kIntermediate = 256;
  auto mlp = create_mlp(kHidden, kIntermediate);
  mlp->load_state_dict(
      StateDict(create_default_test_weights(kHidden, kIntermediate)));
  auto input = torch::full({2, kHidden}, 0.5f, options_);
  const auto actual = mlp->forward(input);
  const auto gate = torch::tensor(kHidden * 0.5f * 0.05f * 5.0f * 0.1f)
                        .to(torch::kBFloat16)
                        .to(torch::kFloat32);
  const float activated =
      (torch::silu(gate) * gate).to(torch::kBFloat16).item<float>();
  const float value = kIntermediate * activated * 0.05f * 3.0f * 0.1f;
  const auto expected = torch::full({2, kHidden}, value, torch::kBFloat16);
  test::verify_tensor_close(actual.cpu(), expected, 1e-2, 1e-3);
}

}  // namespace layer
}  // namespace xllm
