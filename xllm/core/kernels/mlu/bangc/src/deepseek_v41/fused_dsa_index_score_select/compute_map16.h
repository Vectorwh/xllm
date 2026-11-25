/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef INDEX_SCORE_SELECT_KERNELS_COMPUTE_MAP16_H_
#define INDEX_SCORE_SELECT_KERNELS_COMPUTE_MAP16_H_

#include <bang.h>

#include <cstdint>
#include <type_traits>

#include "kernels/mlu/bangc/src/deepseek_v41/fused_dsa_index_score_select/index_score_select.h"

namespace xllm::kernel::mlu::bangc::score_select {

constexpr int kMap64LtNum = 64;
constexpr int kMap16LtNum = 16;
#ifdef __MLU_WRAM_SIZE__
constexpr int kMap16WramBytes = __MLU_WRAM_SIZE__ * 1024;
#else
constexpr int kMap16WramBytes = 512 * 1024;
#endif
constexpr int kMap16LtStrideBytes = kMap16WramBytes / kMap16LtNum;

__mlu_func__ inline void LoadPagedDenseTile(const PageLayout& page,
                                            const bfloat16_t* paged_k,
                                            const int32_t* table_b,
                                            int seg_max_n,
                                            int max_piece_pages,
                                            int c0,
                                            bfloat16_t* kbf,
                                            int32_t* page_tmp,
                                            int* cache_base,
                                            int* cache_end) {
  const int valid_rows =
      (seg_max_n - c0) < kTileSlots ? (seg_max_n - c0) : kTileSlots;
  const int nco = (valid_rows + 63) / 64 * 64;
  const int p_col0 = c0 >> (page.block_shift + 3);
  const int need = (valid_rows + page.rows - 1) >> (page.block_shift + 3);

  if (need > 0 && (p_col0 < *cache_base || p_col0 + need > *cache_end)) {
    const int cnt =
        (max_piece_pages - p_col0) < 64 ? (max_piece_pages - p_col0) : 64;
    __memcpy(page_tmp, table_b + p_col0, cnt * sizeof(int32_t), GDRAM2NRAM);
    __sync_io();
    *cache_base = p_col0;
    *cache_end = p_col0 + cnt;
  }

  for (int first = 0; first < need;) {
    const int64_t physical = page_tmp[p_col0 - *cache_base + first];
    int next = first + 1;
    while (next < need &&
           page_tmp[p_col0 - *cache_base + next] == physical + next - first) {
      ++next;
    }
    const int begin = first * page.rows;
    const int end =
        next * page.rows < valid_rows ? next * page.rows : valid_rows;
    __memcpy_async(kbf + begin * kDim,
                   paged_k + physical * page.rows * kDim,
                   (end - begin) * kDim * sizeof(bfloat16_t),
                   GDRAM2NRAM);
    first = next;
  }
  if (nco > valid_rows) {
    __bang_write_value(kbf + valid_rows * kDim,
                       (nco - valid_rows) * kDim,
                       static_cast<bfloat16_t>(0));
  }
}

namespace mlisa {

template <typename T>
__mlu_device__ inline constexpr char type[] = "";
template <>
__mlu_device__ inline constexpr char type<float>[] = ".f32";
template <>
__mlu_device__ inline constexpr char type<bfloat16_t>[] = ".bf16";
template <>
__mlu_device__ inline constexpr char type<half>[] = ".f16";

template <bool Need>
__mlu_device__ inline constexpr char paren_pair[] = "";
template <>
__mlu_device__ inline constexpr char paren_pair<true>[] = "()";

template <bool Need>
__mlu_device__ inline constexpr char post_cvt[] = "";
template <>
__mlu_device__ inline constexpr char post_cvt<true>[] = ",.cvt.dst.rn";

template <bool Need>
__mlu_device__ inline constexpr char precvt_or_not[] = "";
template <>
__mlu_device__ inline constexpr char precvt_or_not<true>[] = ".precvt";

}  // namespace mlisa

template <typename T, int kLtStrideBytes>
__mlu_func__ void loadQToWram16(T* q_wram,
                                const T* query_address,
                                int head_num,
                                int head_size,
                                int tile_seq_q,
                                mluMemcpyDirection_t dir) {
  const int n = (head_num * tile_seq_q) / kMap64LtNum;
  T* dst = q_wram;
  const T* src = query_address;
  if (n > 0) {
    __memcpy_async(dst,
                   src,
                   sizeof(T) * head_size * 4,
                   dir,
                   /*dst_stride0*/ kLtStrideBytes,
                   kMap16LtNum - 1,
                   /*dst_stride1*/ head_size * sizeof(T) * 4,
                   n - 1,
                   /*src_stride0*/ head_size * sizeof(T) * 4,
                   n * kMap16LtNum - 1,
                   0,
                   0);
    dst += head_size * n * 4;
    src += head_size * n * kMap64LtNum;
  }
  const int co_remain = (head_num * tile_seq_q) % kMap64LtNum;
  if (co_remain > 0) {
    __memcpy_async(dst,
                   src,
                   sizeof(T) * head_size * 4,
                   dir,
                   /*dst_stride0*/ kLtStrideBytes,
                   /*src_stride0*/ head_size * sizeof(T) * 4,
                   co_remain / 4 - 1);
  }
}

// computeQK.
template <typename DST, typename SRC>
__mlu_func__ void computeQK(DST* qk,
                            SRC* q_wram,
                            const void* k_nram,
                            /*M*/ const int wi,
                            /*K*/ const int ci,
                            /*N*/ const int co) {
  constexpr bool post_cvt = !std::is_same_v<DST, float>;
  using TD = std::conditional_t<post_cvt, DST, void>;
  __asm__ __volatile__(
      "conv.nram.rn.f32 %[src_type]%[src_type] [%[dst]], [%[src]], "
      "[%[kernel]], %[ci], "
      "1, %[wi], 1, 1, 1, 1, "
      "%[co],.relu()%[post_cvt_op]%[post_cvt_type]%[paren_pair];\n\t" ::
          [src_type] "C"(mlisa::type<SRC>),
      [post_cvt_op] "C"(mlisa::post_cvt<post_cvt>),
      [dst] "r"(qk),
      [src] "r"(k_nram),
      [kernel] "r"(q_wram),
      [ci] "r"(ci),
      [wi] "r"(wi),
      [co] "r"(co),
      [post_cvt_type] "C"(mlisa::type<TD>),
      [paren_pair] "C"(mlisa::paren_pair<post_cvt>));
}

template <typename T, int kLtStrideBytes>
__mlu_func__ void moveQKMap16(T* qk_wram,
                              const void* qk_nram,
                              uint32_t head_num,
                              uint32_t seq_k,
                              uint32_t seq_q) {
  if (seq_q == 1) {
    __memcpy_async(qk_wram,
                   qk_nram,
                   head_num * sizeof(T) * 4,
                   NRAM2WRAM,
                   /*dststride0*/ kLtStrideBytes,
                   kMap16LtNum - 1,
                   /*dststride1*/ head_num * sizeof(T) * 4,
                   seq_k / kMap64LtNum - 1,
                   head_num * sizeof(T) * 4,
                   seq_k / 4 - 1,
                   0,
                   0);
  } else {
    for (uint32_t q_idx = 0; q_idx < seq_q; ++q_idx) {
      for (uint32_t k_idx = 0; k_idx < seq_k / kMap64LtNum; ++k_idx) {
        __memcpy_async(qk_wram + q_idx * head_num * (seq_k / kMap16LtNum) +
                           k_idx * head_num * 4,
                       static_cast<const uint8_t*>(qk_nram) +
                           k_idx * kMap64LtNum * head_num * sizeof(T) * seq_q +
                           q_idx * head_num * sizeof(T),
                       head_num * sizeof(T),
                       NRAM2WRAM,
                       /*dststride0*/ head_num * sizeof(T),
                       4 - 1,
                       /*dststride1*/ kLtStrideBytes,
                       kMap16LtNum - 1,
                       /*srcstride0*/ head_num * sizeof(T) * seq_q,
                       kMap64LtNum - 1,
                       0,
                       0);
      }
    }
  }
}

// torch_mlu_ops scaleAndSum / loopScaleAndSumMap16.
template <typename IN_DTYPE, typename OUT_DTYPE>
__mlu_func__ void scaleAndSum(OUT_DTYPE* output,
                              const IN_DTYPE* q_scale_nram,
                              const IN_DTYPE* qk_wram,
                              /*K=*/const uint32_t head_num,
                              /*N=*/const uint32_t seq_k) {
  if constexpr (std::is_same_v<IN_DTYPE, float> &&
                std::is_same_v<OUT_DTYPE, float>) {
    __asm__ __volatile__(
        "conv.nram.rn.f32.f32.f32 [%[dst]], [%[src]], [%[kernel]], %[ci], "
        "1, 1, 1, 1, 1, 1, %[co];\n\t" ::[dst] "r"(output),
        [src] "r"(q_scale_nram),
        [kernel] "r"(qk_wram),
        [ci] "r"(head_num),
        [co] "r"(seq_k));
  } else if constexpr (std::is_same_v<IN_DTYPE, float> &&
                       std::is_same_v<OUT_DTYPE, bfloat16_t>) {
    __asm__ __volatile__(
        "conv.nram.rn.f32.f32.f32 [%[dst]], [%[src]], [%[kernel]], %[ci], "
        "1, 1, 1, 1, 1, 1, %[co],.cvt.dst.rn.bf16();\n\t" ::[dst] "r"(output),
        [src] "r"(q_scale_nram),
        [kernel] "r"(qk_wram),
        [ci] "r"(head_num),
        [co] "r"(seq_k));
  } else if constexpr (std::is_same_v<IN_DTYPE, bfloat16_t> &&
                       std::is_same_v<OUT_DTYPE, bfloat16_t>) {
    __asm__ __volatile__(
        "conv.nram.rn.f32.bf16.bf16 [%[dst]], [%[src]], [%[kernel]], %[ci], "
        "1, 1, 1, 1, 1, 1, %[co],.cvt.dst.rn.bf16();\n\t" ::[dst] "r"(output),
        [src] "r"(q_scale_nram),
        [kernel] "r"(qk_wram),
        [ci] "r"(head_num),
        [co] "r"(seq_k));
  } else if constexpr (std::is_same_v<IN_DTYPE, bfloat16_t> &&
                       std::is_same_v<OUT_DTYPE, float>) {
    __asm__ __volatile__(
        "conv.nram.rn.f32.bf16.bf16 [%[dst]], [%[src]], [%[kernel]], %[ci], "
        "1, 1, 1, 1, 1, 1, %[co];\n\t" ::[dst] "r"(output),
        [src] "r"(q_scale_nram),
        [kernel] "r"(qk_wram),
        [ci] "r"(head_num),
        [co] "r"(seq_k));
  }
}

template <typename OUT_DTYPE, typename IN_DTYPE>
__mlu_func__ void loopScaleAndSumMap16(OUT_DTYPE* output,
                                       const IN_DTYPE* q_scale_nram,
                                       const IN_DTYPE* qk_wram,
                                       const uint32_t head_num,
                                       const uint32_t seq_k,
                                       const uint32_t seq_k_align,
                                       const uint32_t seq_q) {
  for (uint32_t q = 0; q < seq_q; ++q) {
    scaleAndSum<IN_DTYPE, OUT_DTYPE>(
        output + q * seq_k,
        q_scale_nram + q * head_num,
        qk_wram + q * head_num * (seq_k_align / kMap16LtNum),
        head_num,
        seq_k_align);
  }
}

__mlu_func__ inline void LoadQueriesMap16(bfloat16_t* q_wram,
                                          const bfloat16_t* query_addr,
                                          int nq,
                                          mluMemcpyDirection_t dir) {
  loadQToWram16<bfloat16_t, kMap16LtStrideBytes>(
      q_wram, query_addr, kHeads, kDim, nq, dir);
}

// Unified single/multi-query Map16 tile scorer parameterized by per-head
// `LogitsT` (`float` for `torch.float32`, `bfloat16_t` for `torch.bfloat16`):
// - `nco` MUST be a multiple of 64 (64, 128, 192, or 256).
//   `computeQK<LogitsT> + moveQKMap16<LogitsT> + loopScaleAndSumMap16<float,
//   LogitsT>`, using a CT||MV intra-tile pipeline when `nq > kGroupQ`.
// - Never calls `__sync_io()`, so background GDRAM->NRAM K prefetch overlaps
//   100% with scoring.
template <int kMaxQ, typename LogitsT>
__mlu_func__ inline void ScoreTileMap16Typed(float* scores_out,
                                             void* qk_nram_raw,
                                             bfloat16_t* wram_base,
                                             const bfloat16_t* k_tile_bf16,
                                             const LogitsT* weights_nram,
                                             int nq,
                                             int nco) {
  constexpr int kGroupQ = std::is_same_v<LogitsT, bfloat16_t> ? 4 : 2;
  constexpr int kQWramElemsPerLt = ((kMaxQ + 1) / 2) * 4 * kDim;
  LogitsT* qk_nram_buf = static_cast<LogitsT*>(qk_nram_raw);
  LogitsT* qk_wram0 = reinterpret_cast<LogitsT*>(wram_base + kQWramElemsPerLt);
  if constexpr (kMaxQ <= kGroupQ) {
    computeQK<LogitsT, bfloat16_t>(
        qk_nram_buf, wram_base, k_tile_bf16, nco, kDim, nq * kHeads);
    __sync_compute();
    moveQKMap16<LogitsT, kMap16LtStrideBytes>(
        qk_wram0, qk_nram_buf, kHeads, nco, nq);
    __sync_move();
    loopScaleAndSumMap16<float, LogitsT>(
        scores_out, weights_nram, qk_wram0, kHeads, nco, nco, nq);
    __sync_compute();
  } else {
    if (nq <= kGroupQ) {
      computeQK<LogitsT, bfloat16_t>(
          qk_nram_buf, wram_base, k_tile_bf16, nco, kDim, nq * kHeads);
      __sync_compute();
      moveQKMap16<LogitsT, kMap16LtStrideBytes>(
          qk_wram0, qk_nram_buf, kHeads, nco, nq);
      __sync_move();
      loopScaleAndSumMap16<float, LogitsT>(
          scores_out, weights_nram, qk_wram0, kHeads, nco, nco, nq);
      __sync_compute();
      return;
    }
    constexpr int kQKNramStride = kTileSlots * kGroupQ * kHeads;  // 64 KiB
    const int qk_wram_stride = kGroupQ * kHeads * (nco / kMap16LtNum);
    const int groups = (nq + kGroupQ - 1) / kGroupQ;
    for (int s = 0; s < groups + 2; ++s) {
      if (s >= 2) {
        const int g2 = s - 2;
        const int q0 = g2 * kGroupQ;
        const int cq = (nq - q0) < kGroupQ ? (nq - q0) : kGroupQ;
        loopScaleAndSumMap16<float, LogitsT>(
            scores_out + q0 * nco,
            weights_nram + q0 * kHeads,
            qk_wram0 + (g2 & 1) * qk_wram_stride,
            kHeads,
            nco,
            nco,
            cq);
      }
      if (s >= 1 && s - 1 < groups) {
        const int g1 = s - 1;
        const int q0 = g1 * kGroupQ;
        const int cq = (nq - q0) < kGroupQ ? (nq - q0) : kGroupQ;
        moveQKMap16<LogitsT, kMap16LtStrideBytes>(
            qk_wram0 + (g1 & 1) * qk_wram_stride,
            qk_nram_buf + (g1 & 1) * kQKNramStride,
            kHeads,
            nco,
            cq);
      }
      if (s < groups) {
        const int q0 = s * kGroupQ;
        const int cq = (nq - q0) < kGroupQ ? (nq - q0) : kGroupQ;
        computeQK<LogitsT, bfloat16_t>(qk_nram_buf + (s & 1) * kQKNramStride,
                                       wram_base + s * (2 * kGroupQ * kDim),
                                       k_tile_bf16,
                                       nco,
                                       kDim,
                                       cq * kHeads);
      }
      __sync_compute();
      __sync_move();
    }
  }
}

template <int kMaxQ>
__mlu_func__ inline void ScoreTileMap16(float* scores_out,
                                        void* qk_nram_buf,
                                        bfloat16_t* wram_base,
                                        const bfloat16_t* k_tile_bf16,
                                        const float* w_f32,
                                        const bfloat16_t* w_bf16,
                                        int nq,
                                        int nco,
                                        int logits_dtype) {
  if (logits_dtype == kLogitsBf16) {
    ScoreTileMap16Typed<kMaxQ, bfloat16_t>(
        scores_out, qk_nram_buf, wram_base, k_tile_bf16, w_bf16, nq, nco);
  } else {
    ScoreTileMap16Typed<kMaxQ, float>(
        scores_out, qk_nram_buf, wram_base, k_tile_bf16, w_f32, nq, nco);
  }
}

}  // namespace xllm::kernel::mlu::bangc::score_select

#endif  // INDEX_SCORE_SELECT_KERNELS_COMPUTE_MAP16_H_
