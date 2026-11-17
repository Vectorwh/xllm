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

#pragma once

#include <cnrt.h>

#include <cstdint>

namespace xllm::kernel::mlu::bangc {

// Kernel dtype codes selected by launch_prefill: 0 float32, 1 float16,
// 2 bfloat16.
constexpr int32_t kPrefillFloat32 = 0;
constexpr int32_t kPrefillFloat16 = 1;
constexpr int32_t kPrefillBFloat16 = 2;

// Device-side launch parameters for the causal Conv1d prefill kernel. Strides
// are element strides; the metadata pointers reference device Int32/Bool arrays
// that follow the prefill state contract.
struct PrefillParams {
  const void* x;
  const void* weight;
  void* state;
  void* output;
  const void* bias;
  const int32_t* query;
  const int32_t* batch;
  const int32_t* chunks;
  const int32_t* indices;
  const int32_t* read_indices;
  const bool* initial;
  int64_t dim;
  int64_t nt;
  int64_t x_channel_stride;
  int64_t x_token_stride;
  int64_t weight_channel_stride;
  int64_t weight_width_stride;
  int64_t state_slot_stride;
  int64_t state_channel_stride;
  int64_t state_history_stride;
  int64_t output_channel_stride;
  int64_t output_token_stride;
  int64_t index_stride;
  int64_t read_index_stride;
  int64_t bias_stride;
  int32_t width;
};

cnrtRet_t launch_prefill(PrefillParams params,
                         int32_t dtype,
                         cnrtQueue_t queue);

}  // namespace xllm::kernel::mlu::bangc
