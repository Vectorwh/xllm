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

extern "C" void engram_hash_launch(void* queue,
                                   const void* tokens,
                                   bool int32_tokens,
                                   const int32_t* starts,
                                   const int64_t* history,
                                   const int64_t* token_map,
                                   const int64_t* multipliers,
                                   const int64_t* primes,
                                   const int64_t* offsets,
                                   int64_t* output,
                                   int64_t tokens_count,
                                   int32_t sequences,
                                   int32_t layers,
                                   int32_t ngram,
                                   int32_t columns,
                                   int64_t vocab,
                                   int64_t pad_token,
                                   int32_t cores);
