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

constexpr int32_t kKdaFloat32 = 0;
constexpr int32_t kKdaBFloat16 = 2;

cnrtRet_t launch_kda_state_writeback(const void* source,
                                     void* cache,
                                     const int32_t* indices,
                                     int64_t rows,
                                     int64_t width,
                                     int64_t source_stride,
                                     int64_t cache_stride,
                                     int64_t cache_rows,
                                     int32_t source_dtype,
                                     int32_t cache_dtype,
                                     cnrtQueue_t queue);

}  // namespace xllm::kernel::mlu::bangc
