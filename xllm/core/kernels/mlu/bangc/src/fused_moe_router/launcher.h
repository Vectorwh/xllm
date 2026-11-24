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

#include <cstdint>

namespace xllm::kernel::mlu::bangc {
struct RouterBias {
  const float* text;
  const float* visual;
  const void* input_ids;
  int64_t image_token_id;
  bool int64_ids;
};

inline constexpr int32_t kRouterHidden = 5120;
inline constexpr int32_t kRouterBlock = 320;
inline constexpr int32_t kRouterResidentBlock = 640;
inline constexpr int32_t kRouterResidentExperts = 128;

// Each K partition writes one partial score per token and expert.
inline int32_t router_k_partitions(int32_t rows) {
  if (rows <= 48) {
    return 16;
  }
  if (rows <= 200) {
    return 8;
  }
  if (rows <= 450) {
    return 4;
  }
  return rows <= 950 ? 2 : 1;
}

void fused_moe_router_launch(void* queue,
                             const void* hidden,
                             const void* weight,
                             RouterBias bias,
                             float* out_weights,
                             int32_t* out_ids,
                             int32_t rows,
                             int32_t experts,
                             int32_t topk,
                             float scale,
                             int32_t partitions,
                             float* workspace);

void fused_moe_router_resident_launch(void* queue,
                                      const void* hidden,
                                      const void* weight,
                                      RouterBias bias,
                                      float* out_weights,
                                      int32_t* out_ids,
                                      int32_t rows,
                                      int32_t topk,
                                      float scale);
}  // namespace xllm::kernel::mlu::bangc
