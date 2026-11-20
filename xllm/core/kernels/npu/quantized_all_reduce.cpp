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

#include <ATen/ops/linalg_vector_norm.h>
#include <glog/logging.h>

#include <atomic>
#include <limits>
#include <optional>

#include "core/common/global_flags.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kMaxQuantizedWorldSize = 8;
constexpr int64_t kMinLastDimension = 64;
constexpr int64_t kMaxInt8Value = 127;
constexpr double kMinQuantizationScale = 1e-8;

bool should_quantize(const torch::Tensor& input,
                     int32_t world_size,
                     int64_t min_numel) {
  return input.scalar_type() == torch::kBFloat16 && input.dim() >= 2 &&
         input.size(-1) >= kMinLastDimension && input.numel() >= min_numel &&
         world_size <= kMaxQuantizedWorldSize;
}

torch::Tensor& anti_quantize_per_channel(const torch::Tensor& input,
                                         const torch::Tensor& scales,
                                         torch::Tensor& output) {
  std::optional<torch::Tensor> offset = std::nullopt;
  const int64_t output_dtype = static_cast<int64_t>(ACL_BF16);
  constexpr bool sqrt_mode = false;
  EXEC_NPU_CMD(aclnnAscendAntiQuant,
               input,
               scales,
               offset,
               output_dtype,
               sqrt_mode,
               output);
  return output;
}

}  // namespace

void quantized_all_reduce(torch::Tensor& input,
                          ProcessGroup& process_group,
                          int64_t min_numel,
                          c10::intrusive_ptr<c10d::Work>* work) {
  const int32_t world_size = process_group.world_size();
  if (!should_quantize(input, world_size, min_numel)) {
    process_group.allreduce(input);
    return;
  }

  // One-shot observability: proves the quantized path actually executes at
  // runtime (as opposed to merely being enabled in the config).
  static std::atomic<bool> first_activation_logged{false};
  if (!first_activation_logged.exchange(true)) {
    LOG(INFO) << "Quantized all-reduce path activated: world_size="
              << world_size << ", min_numel=" << min_numel
              << ", first eligible shape=" << input.sizes();
  }

  torch::Tensor flat = input.reshape({-1, input.size(-1)});
  torch::Tensor amax =
      at::linalg_vector_norm(flat,
                             std::numeric_limits<double>::infinity(),
                             /*dim=*/{0},
                             /*keepdim=*/false,
                             /*dtype=*/torch::kFloat32);
  // wait() chains the dependent scale/quantize ops behind the reduced amax on
  // the device stream instead of blocking the host pipeline.
  process_group.allreduce_async(amax, c10d::ReduceOp::MAX)->wait();

  // Reserve enough INT8 headroom for the worst-case same-sign SUM on every
  // rank. Clamping each rank to 127 would overflow during the reduction.
  const int64_t max_q = kMaxInt8Value / world_size;
  torch::Tensor scales =
      (amax / static_cast<double>(max_q)).clamp_min(kMinQuantizationScale);
  torch::Tensor zero_points;
  torch::Tensor quantized = quantize_per_tensor(
      flat, scales, zero_points, at::ScalarType::QInt8, /*axis=*/1);
  // wait() keeps the in-place dequantize below stream-ordered behind the
  // reduction; the same handle is handed back to async callers (launch_reduce)
  // so one wait covers the whole quantize-reduce-dequantize chain.
  auto reduce_work = process_group.allreduce_async(quantized);
  reduce_work->wait();
  if (work != nullptr) {
    *work = std::move(reduce_work);
  }

  // Dequantize straight into the caller's buffer's flat view: this preserves
  // the ProcessGroup::allreduce in-place contract (the tensor address must not
  // change under graph capture) and saves an extra bf16 copy per collective.
  if (input.is_contiguous()) {
    anti_quantize_per_channel(quantized, scales, flat);
    return;
  }
  // A non-contiguous input cannot be written through its reshaped view, so
  // dequantize into a temporary and copy back once.
  torch::Tensor output = torch::empty(quantized.sizes().vec(), input.options());
  anti_quantize_per_channel(quantized, scales, output);
  input.copy_(output.reshape(input.sizes()));
}

bool try_quantized_all_reduce(torch::Tensor& input,
                              ProcessGroup& process_group,
                              c10::intrusive_ptr<c10d::Work>* work) {
#if defined(USE_NPU)
  // Soft gate for the native (C++ layers) path. Unlike PyCausalLM, which
  // validates at model construction, this funnel also carries non-TP groups
  // (e.g. EP), so disqualifiers degrade to a fallback with a one-shot warning
  // instead of a startup CHECK.
  static std::atomic<bool> disabled{false};
  if (disabled.load(std::memory_order_relaxed)) {
    return false;
  }

  auto& kernel_config = KernelConfig::get_instance();
  if (!kernel_config.enable_quantized_all_reduce()) {
    disabled.store(true, std::memory_order_relaxed);
    return false;
  }
  static std::atomic<bool> gate_warned{false};
  if (!is_ascend_a2()) {
    if (!gate_warned.exchange(true)) {
      LOG(WARNING) << "Quantized TP AllReduce requires Atlas A2; disabling.";
    }
    disabled.store(true, std::memory_order_relaxed);
    return false;
  }
  if (FLAGS_enable_flashcomm1) {
    if (!gate_warned.exchange(true)) {
      LOG(WARNING) << "Quantized TP AllReduce is incompatible with FlashComm1; "
                      "disabling quantized TP AllReduce.";
    }
    disabled.store(true, std::memory_order_relaxed);
    return false;
  }

  const int64_t min_numel = kernel_config.quantized_all_reduce_min_numel();
  if (!should_quantize(input, process_group.world_size(), min_numel)) {
    // Per-call eligibility (bf16, last-dim, numel, world size): any
    // unqualified collective falls back to the plain bf16 all-reduce, which
    // is the caller's job now.
    return false;
  }
  quantized_all_reduce(input, process_group, min_numel, work);
  return true;
#else
  (void)input;
  (void)process_group;
  (void)work;
  return false;
#endif
}

}  // namespace xllm::kernel::npu
