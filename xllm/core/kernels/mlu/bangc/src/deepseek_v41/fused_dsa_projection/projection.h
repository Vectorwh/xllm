/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include <cnrt.h>

#include <cstdint>

namespace xllm::kernel::mlu::bangc::projection {

constexpr int32_t kHidden = 5120;
constexpr int32_t kQRank = 1280;
constexpr int32_t kKvDim = 512;
constexpr int32_t kRotary = 64;
constexpr int32_t kMaxSmallM = 4096;
constexpr int32_t kMaxLargeM = 32768;
constexpr int32_t kMaxSegments = 4;
constexpr int32_t kMaxTiles = 12;

// One contiguous slice of the concatenated output columns. All segment
// lengths are multiples of 256 so a GEMM column tile never straddles two.
struct Segment {
  const void* weight;  // [n_len, kHidden] BF16 row-major
  void* out;           // first output element of this segment
  int32_t n_begin;     // global column offset in the concatenated order
  int32_t n_len;
  int32_t ld;        // output row stride in elements
  int32_t fp32_out;  // 1: FP32 compressor output, 0: BF16 pre-norm output
};

struct NormParams {
  void* kv;              // [m, 512] BF16, holds bf16(X W_kv^T) on entry
  void* qr;              // [m, 1280] BF16, holds bf16(X W_qa^T) on entry
  const void* gamma_kv;  // [512] BF16
  const void* gamma_q;   // [1280] BF16
  const int32_t* positions;
  const float* rope;  // [rope_rows, 128] FP32
  int32_t rope_rows;
  int32_t m;
  float eps;
};

struct GemmParams {
  const void* x;  // [m, kHidden] BF16
  int32_t m;
  int32_t n_total;
  int32_t nseg;
  Segment seg[kMaxSegments];
  int32_t bm;        // rows per cluster unit
  int32_t bn;        // max columns per cluster unit (buffer sizing)
  int32_t kb;        // K step
  int32_t split_n;   // IPUs splitting the unit columns (1, 2 or 4)
  int32_t a_direct;  // 1: each IPU loads its X rows GDRAM->NRAM, 0: via SRAM
  int32_t row_blocks;
  int32_t col_tiles;
  int32_t tile_n0[kMaxTiles];  // global first column of each column tile
  int32_t tile_w[kMaxTiles];   // tile width (multiple of 64 * split_n)
  // RMSNorm/RoPE as a GEMM-kernel tail: each IPU normalises the rows it
  // computed (needs split_n == 1 and whole row blocks per cluster).
  int32_t tail_norm;
  NormParams norm;
};

struct ProjectionArgs {
  const void *x, *w_kv, *w_qa, *gamma_kv, *gamma_q, *w_ckv, *w_gate;
  const int32_t* positions;
  const float* rope;
  void *kv, *qr, *raw_scores;
  int32_t m, c, rope_rows;
  float eps;
};
// Both tilings write the same column segments and apply the same normalization.
template <typename Params>
inline void init_projection(Params& params, const ProjectionArgs& args) {
  params.x = args.x;
  params.m = args.m;
  const void* weights[4] = {args.w_kv, args.w_qa, args.w_ckv, args.w_gate};
  const int32_t lens[4] = {kKvDim, kQRank, kKvDim, kKvDim};
  const int32_t comp_ld = kKvDim * args.c;
  void* outs[4] = {
      args.kv,
      args.qr,
      args.raw_scores,
      args.c == 2 ? static_cast<float*>(args.raw_scores) + kKvDim : nullptr};
  const int32_t lds[4] = {kKvDim, kQRank, comp_ld, comp_ld};
  params.nseg = 2 + args.c;
  int32_t begin = 0;
  for (int32_t i = 0; i < params.nseg; ++i) {
    params.seg[i].weight = weights[i];
    params.seg[i].out = outs[i];
    params.seg[i].n_begin = begin;
    params.seg[i].n_len = lens[i];
    params.seg[i].ld = lds[i];
    params.seg[i].fp32_out = i >= 2 && args.c == 2 ? 1 : 0;
    begin += lens[i];
  }
  params.n_total = begin;
  auto& n = params.norm;
  n.kv = args.kv;
  n.qr = args.qr;
  n.gamma_kv = args.gamma_kv;
  n.gamma_q = args.gamma_q;
  n.positions = args.positions;
  n.rope = args.rope;
  n.rope_rows = args.rope_rows;
  n.m = args.m;
  n.eps = args.eps;
}

// Returns 0 on success, otherwise a cnrtRet_t value.
int32_t launch(const ProjectionArgs& args, cnrtQueue_t queue);
}  // namespace xllm::kernel::mlu::bangc::projection
