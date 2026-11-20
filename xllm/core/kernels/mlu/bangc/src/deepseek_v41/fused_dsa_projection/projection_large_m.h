/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include "kernels/mlu/bangc/src/deepseek_v41/fused_dsa_projection/projection.h"

namespace xllm::kernel::mlu::bangc::projection {

constexpr int32_t kMaxLargeTiles = 16;

struct LargeMGemmParams {
  const void* x;  // [m, kHidden] BF16
  int32_t m;
  int32_t n_total;
  int32_t nseg;
  Segment seg[kMaxSegments];
  int32_t bm;
  int32_t bn;
  int32_t kb;
  int32_t row_blocks;
  int32_t col_tiles;
  int32_t tile_n0[kMaxLargeTiles];
  int32_t tile_w[kMaxLargeTiles];
  int32_t fuse_norm;
  NormParams norm;
};

int32_t launch_large_m(const ProjectionArgs& args,
                       int32_t clusters,
                       cnrtQueue_t queue);

}  // namespace xllm::kernel::mlu::bangc::projection
