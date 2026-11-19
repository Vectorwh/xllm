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

#include <torch/types.h>

namespace xllm::kernel::mlu::bangc {

// All inputs reside on MLU. starts [S+1] delimits requests; history [S,N-1]
// stores newest predecessors first. Output [L,T,(N-1)*H] keeps each layer
// contiguous for the existing gather/dequantization kernels. Hash parameters
// must satisfy the nonnegative, nonoverflowing int64 contract checked at load.
torch::Tensor engram_hash(const torch::Tensor& tokens,
                          const torch::Tensor& starts,
                          const torch::Tensor& history,
                          const torch::Tensor& token_map,
                          const torch::Tensor& multipliers,
                          const torch::Tensor& primes,
                          const torch::Tensor& offsets,
                          int64_t pad_token);

}  // namespace xllm::kernel::mlu::bangc
