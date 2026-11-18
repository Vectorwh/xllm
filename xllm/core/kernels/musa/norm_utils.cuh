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

#include <cuda_runtime.h>

namespace xllm::kernel::musa {

__device__ __forceinline__ float fast_rsqrt(float value) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  const float half_value = 0.5f * value;
  float y = __frsqrt_rn(value);
  y = y * (1.5f - half_value * y * y);
  return y;
#else
  return rsqrtf(value);
#endif
}

}  // namespace xllm::kernel::musa
