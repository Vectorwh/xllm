/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include "kernels/mlu/bangc/src/deepseek_v41/fused_dsa_projection/projection.h"

namespace xllm::kernel::mlu::bangc::projection {
__mlu_global__ void gemm_kernel(GemmParams p);
__mlu_global__ void norm_kernel(NormParams p);
}  // namespace xllm::kernel::mlu::bangc::projection
