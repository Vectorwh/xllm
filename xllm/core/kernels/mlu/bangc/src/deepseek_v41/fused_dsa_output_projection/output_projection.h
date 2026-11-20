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

#pragma once

#include <cstdint>

#include "cnrt.h"

namespace xllm::kernel::mlu::bangc::output_projection {

constexpr int32_t kFusedDecodeMaxRows = 768;
constexpr int32_t kGroupWidth = 4096;
constexpr int32_t kGroupOut = 1024;

struct WorkspaceLayout {
  int64_t qa_offset = 0;
  int64_t sa_offset = 0;
  int64_t y_offset = 0;
  int64_t total_bytes = 0;
};

inline int64_t align256(int64_t bytes) { return (bytes + 255) & ~int64_t{255}; }

inline WorkspaceLayout get_workspace_layout(int64_t m, int64_t g) {
  WorkspaceLayout layout{};
  if (g == 1 && m <= kFusedDecodeMaxRows) {
    return layout;
  }
  const int64_t y_bytes =
      align256(m * kGroupOut * g * static_cast<int64_t>(sizeof(uint16_t)));
  if (m * g <= kFusedDecodeMaxRows) {
    layout.y_offset = 0;
    layout.total_bytes = y_bytes;
    return layout;
  }
  const int64_t qa_bytes =
      align256(g * m * kGroupWidth * static_cast<int64_t>(sizeof(int8_t)));
  const int64_t sa_bytes =
      align256(g * m * static_cast<int64_t>(sizeof(float)));
  layout.qa_offset = 0;
  layout.sa_offset = qa_bytes;
  layout.y_offset = qa_bytes + sa_bytes;
  layout.total_bytes = qa_bytes + sa_bytes + y_bytes;
  return layout;
}

cnrtRet_t launch(cnrtQueue_t queue,
                 const void* x,
                 const int32_t* pos,
                 const float* rope,
                 int32_t last_pos,
                 const int8_t* weight,
                 const float* scale,
                 const float* smooth_a,
                 const float* smooth_b,
                 void* workspace,
                 int8_t* out_q,
                 float* out_scale,
                 int32_t m,
                 int32_t g);

}  // namespace xllm::kernel::mlu::bangc::output_projection
