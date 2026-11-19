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

extern "C" void engram_dequantize_launch(void* queue,
                                         const void* bytes,
                                         const void* scales,
                                         const void* hashes,
                                         const void* fp8_values,
                                         void* output,
                                         int64_t rows,
                                         int32_t dim,
                                         int32_t scale_dim,
                                         int64_t table_start,
                                         int64_t table_end,
                                         bool all_owned);

extern "C" void engram_dequantize_packed_launch(void* queue,
                                                const void* packed,
                                                const void* hashes,
                                                const void* fp8_values,
                                                void* output,
                                                int64_t rows,
                                                int32_t dim,
                                                int32_t scale_dim,
                                                int64_t packed_row_stride,
                                                int64_t tokens,
                                                int64_t columns,
                                                int64_t head_shard_count,
                                                int64_t table_start,
                                                int64_t table_end,
                                                bool all_owned,
                                                const void* row_indices);
