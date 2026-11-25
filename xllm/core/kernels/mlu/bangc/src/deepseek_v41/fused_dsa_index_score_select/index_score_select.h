/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef INDEX_SCORE_SELECT_KERNELS_INDEX_SCORE_SELECT_H_
#define INDEX_SCORE_SELECT_KERNELS_INDEX_SCORE_SELECT_H_

#include <cnrt.h>

#include <cstdint>

#if defined(__BANG__)
#define ISS_HD __mlu_func__ __mlu_host__
#else
#define ISS_HD
#endif

namespace xllm::kernel::mlu::bangc::score_select {

enum Mode : int { kNone = 0, kSource = 1, kFilter = 2 };
enum ExecutionMode : int { kExecPrefill = 0, kExecDecode = 1 };
enum LogitsDtype : int { kLogitsFp32 = 0, kLogitsBf16 = 1 };

constexpr int kHeads = 32;
constexpr int kDim = 128;
constexpr int kTopk = 512;
constexpr int kBlockSize = 8;
constexpr int kBlocks = 2048;
struct PageLayout {
  int32_t rows;
  int32_t block_shift;
  int32_t block_mask;
  int32_t bytes;
  float inverse_blocks;
};
constexpr int kPartCap =
    1024;  // entries per partial token pool (>= kTopk + kTileSlots)
constexpr int kTileSlots = 256;

ISS_HD inline int64_t Align64(int64_t x) { return (x + 63) / 64 * 64; }
ISS_HD inline int64_t Align512(int64_t x) { return (x + 511) / 512 * 512; }

// =============================================================================
// Unified Rank-3 Packed Prefill / Prefill CP Layout (mtp_592)
// =============================================================================

constexpr int kPagedMaxCores = 64;
constexpr int kPagedMaxSplitSegs = 64;
constexpr int kPagedMaxSplitPieceSlots = 128;
constexpr int kPagedBlkPoolCap = 4096;  // >= kBlocks (2048) + 32
constexpr int kPagedUnionCap =
    8192;  // max union blocks where union beats per-query

struct alignas(64) PagedSchedHeader {
  int32_t num_segs;
  int32_t num_pieces;
  int32_t num_split_segs;
  int32_t total_tiles;
  int32_t num_perq_segs;
  int32_t core_piece_begin[kPagedMaxCores];
  int32_t core_piece_end[kPagedMaxCores];
};

struct alignas(64) PagedSeg {
  int32_t b;
  int32_t nq;
  int32_t max_n;
  int32_t min_n;
  int32_t num_tiles;
  int32_t pad0;
  int32_t pad1;
  int32_t pad2;
  int32_t rows[16];
  int32_t n_read[16];
  int32_t pad_tail[8];
};

struct PagedPiece {
  int32_t seg_idx;
  int32_t part_idx;
  int32_t tile_lo;
  int32_t tile_hi;
  int32_t is_final;
  int32_t slot_idx;
  int32_t pad0;
  int32_t pad1;
};

struct PagedSplitSeg {
  int32_t seg_idx;
  int32_t first_slot;
  int32_t num_parts;
  int32_t pad0;
};

ISS_HD inline int64_t PagedUnionSegBytes() {
  return 64 + 2 * static_cast<int64_t>(kPagedUnionCap) * sizeof(int32_t);
}

ISS_HD inline int64_t MaxPagedSegments(int64_t ncap, int64_t bcap, int tq) {
  if (ncap <= 0 || bcap <= 0) return 0;
  const int64_t non_empty_b = bcap < ncap ? bcap : ncap;
  const int64_t bound = ncap / tq + non_empty_b;
  return bound < ncap ? bound : ncap;
}

struct PagedPlan {
  int tq;
  int sched_override;
  int do_split;
  int funion;
  int cores;
  int64_t max_segs;
  int64_t max_pieces;
  int64_t header_offset;
  int64_t seg_offset;
  int64_t piece_offset;
  int64_t split_seg_offset;
  int64_t plan_scratch_offset;
  int64_t tok_part_val_offset;
  int64_t tok_part_idx_offset;
  int64_t core_blk_val_offset;
  int64_t core_blk_idx_offset;
  int64_t blk_part_val_offset;
  int64_t blk_part_idx_offset;
  int64_t union_offset;
  int64_t row_seg_offset;
  int64_t bytes;
};

ISS_HD inline PagedPlan MakePagedPlan(int64_t ncap,
                                      int64_t bcap,
                                      int mode,
                                      int cores,
                                      int sched_override) {
  PagedPlan p{};
  const int eff_cores =
      cores <= 0 ? 1 : (cores > kPagedMaxCores ? kPagedMaxCores : cores);
  p.cores = eff_cores;
  p.sched_override = sched_override;
  p.tq = (sched_override == 2 || (ncap > 0 && ncap <= 8 && sched_override == 0))
             ? 8
             : 16;
  p.do_split = (sched_override == 1) ? 0 : 1;
  if (ncap <= 0 || bcap <= 0) {
    p.bytes = 4096;
    return p;
  }
  p.max_segs = MaxPagedSegments(ncap, bcap, p.tq);
  p.funion =
      (mode == kFilter && p.tq == 16 && sched_override != 2 &&
       (sched_override == 1 || ncap > 4 * static_cast<int64_t>(eff_cores)))
          ? 1
          : 0;
  if (mode == kFilter && !p.funion) {
    p.bytes = 4096;
    return p;
  }
  p.max_pieces = p.max_segs + eff_cores;

  int64_t cur = 0;
  p.header_offset = cur;
  cur += Align64(sizeof(PagedSchedHeader));

  p.seg_offset = cur;
  cur += Align64(p.max_segs * static_cast<int64_t>(sizeof(PagedSeg)));

  p.piece_offset = cur;
  cur += Align64(p.max_pieces * static_cast<int64_t>(sizeof(PagedPiece)));

  p.split_seg_offset = cur;
  cur += Align64(static_cast<int64_t>(kPagedMaxSplitSegs) *
                 static_cast<int64_t>(sizeof(PagedSplitSeg)));

  p.plan_scratch_offset = cur;
  const int64_t plan_ints = 4 * ncap + 2 * (bcap + 1);
  cur += Align64(plan_ints * static_cast<int64_t>(sizeof(int32_t)));

  const int64_t tok_entries =
      static_cast<int64_t>(kPagedMaxSplitPieceSlots) * 16 * kTopk;
  p.tok_part_val_offset = cur;
  cur += Align64(tok_entries * static_cast<int64_t>(sizeof(float)));
  p.tok_part_idx_offset = cur;
  cur += Align64(tok_entries * static_cast<int64_t>(sizeof(int32_t)));

  p.core_blk_val_offset = cur;
  p.core_blk_idx_offset = cur;
  p.blk_part_val_offset = cur;
  p.blk_part_idx_offset = cur;
  if (mode == kSource) {
    const int64_t core_blk_entries =
        static_cast<int64_t>(eff_cores) * 16 * kPagedBlkPoolCap;
    p.core_blk_val_offset = cur;
    cur += Align64(core_blk_entries * static_cast<int64_t>(sizeof(float)));
    p.core_blk_idx_offset = cur;
    cur += Align64(core_blk_entries * static_cast<int64_t>(sizeof(int32_t)));

    const int64_t part_blk_entries =
        static_cast<int64_t>(kPagedMaxSplitPieceSlots) * 16 * kBlocks;
    p.blk_part_val_offset = cur;
    cur += Align64(part_blk_entries * static_cast<int64_t>(sizeof(float)));
    p.blk_part_idx_offset = cur;
    cur += Align64(part_blk_entries * static_cast<int64_t>(sizeof(int32_t)));
  }

  p.union_offset = cur;
  p.row_seg_offset = cur;
  if (mode == kFilter) {
    if (p.funion) {
      cur += Align64(p.max_segs * PagedUnionSegBytes());
      p.row_seg_offset = cur;
      cur += Align64(ncap * static_cast<int64_t>(sizeof(int32_t)));
    }
  }

  p.bytes = cur < 4096 ? 4096 : cur;
  return p;
}

struct PagedParams {
  const void* query;                    // bf16 [Ncap, 32, 128]
  const void* weights;                  // bf16 [Ncap, 32]
  const void* paged_k;                  // bf16 [Nphys, 128]
  const int32_t* query_positions;       // int32 [Ncap]
  const int32_t* query_batch_indices;   // int32 [Ncap]
  const int32_t* num_valid_input_rows;  // int32 [1] on device
  const int32_t* paged_block_table;     // int32 [Bcap, P]
  const int32_t* history_lengths;       // int32 [Bcap]
  const int32_t* candidate_ids;         // int32 [Ncap, 2048] or nullptr
  int32_t* topk_indices;                // int32 [Ncap, 512]
  int32_t* candidate_ids_out;           // int32 [Ncap, 2048] or nullptr
  uint8_t* workspace;
  int64_t nq;     // Ncap
  int64_t batch;  // Bcap
  int64_t max_pages;
  int64_t paged_rows;
  int mode;
  PageLayout page;
  int ratio;
  int logits_dtype;
  PagedPlan plan;
};

int64_t PagedWorkspaceBytes(int64_t ncap,
                            int64_t bcap,
                            int mode,
                            int cores,
                            int sched_override,
                            int logits_dtype = kLogitsFp32);

void LaunchPagedPrefillIndexScoreSelect(cnrtQueue_t queue,
                                        const PagedParams& params,
                                        int clusters,
                                        int cores);

// =============================================================================
// Unified Rank-3 Packed Decode / Variable-Length Verify Layout (mtp_592)
// =============================================================================

constexpr int kDecodeSegQ = 8;
constexpr int kDecodeMaxSplitSegs = 64;
constexpr int kDecodeMaxSplitPieceSlots = 128;
constexpr int kDecodeFilterUnionCap = kDecodeSegQ * kBlocks;  // 16384
// Odd multiples of 128 32-bit words (odd multiples of 512B) break 32-core
// concurrent HBM pseudo-channel / LLC set bank aliasing across split slots.
constexpr int kDecodeTokSlotStride = kDecodeSegQ * kTopk + 128;    // 4224
constexpr int kDecodeBlkSlotStride = kDecodeSegQ * kBlocks + 128;  // 16512
constexpr int kDecodeCoreBlkStride =
    kDecodeSegQ * kPagedBlkPoolCap + 128;  // 32896
constexpr int kDecodeCoreUnionStride =
    2 * kDecodeFilterUnionCap + 128;  // 32896

struct alignas(64) DecodeMergeHeader {
  int32_t num_splits;
  int32_t pad[15];
};

struct DecodeSplitSeg {
  int32_t row0;
  int32_t nq;
  int32_t first_slot;
  int32_t num_parts;
};

struct DecodePlan {
  int tq;
  int sched_override;
  int do_split;
  int funion;
  int cores;
  uint32_t header_offset;
  uint32_t split_seg_offset;
  uint32_t tok_part_cnt_offset;
  uint32_t blk_part_cnt_offset;
  uint32_t tok_part_val_offset;
  uint32_t tok_part_idx_offset;
  uint32_t core_blk_val_offset;
  uint32_t core_blk_idx_offset;
  uint32_t blk_part_val_offset;
  uint32_t blk_part_idx_offset;
  uint32_t union_scratch_offset;
  int64_t bytes;
};

// Bounded-concurrency workspace for unified rank-3 packed Decode/Verify:
// O(cores) scratch only; strictly independent of P (history capacity) and
// Nphysical across ALL modes (NONE, SOURCE, FILTER).
ISS_HD inline DecodePlan MakeDecodePlan(int64_t ncap,
                                        int64_t bcap,
                                        int mode,
                                        int cores,
                                        int sched_override) {
  DecodePlan p{};
  const int eff_cores =
      cores <= 0 ? 1 : (cores > kPagedMaxCores ? kPagedMaxCores : cores);
  p.cores = eff_cores;
  p.sched_override = sched_override;
  p.tq = (sched_override == 2) ? 4 : kDecodeSegQ;
  p.funion = (mode == kFilter &&
              ((ncap > bcap && sched_override != 2) || sched_override == 3))
                 ? 1
                 : 0;
  p.do_split =
      (sched_override == 1 || (mode == kFilter && sched_override == 2)) ? 0 : 1;

  if (ncap <= 0 || bcap <= 0) {
    p.bytes = 4096;
    return p;
  }

  int64_t cur = 0;
  p.header_offset = static_cast<uint32_t>(cur);
  cur += Align64(sizeof(DecodeMergeHeader));

  p.split_seg_offset = static_cast<uint32_t>(cur);
  cur += Align512(cur + static_cast<int64_t>(kDecodeMaxSplitSegs) *
                            static_cast<int64_t>(sizeof(DecodeSplitSeg))) -
         cur;

  const int64_t part_cnt_entries =
      static_cast<int64_t>(kDecodeMaxSplitPieceSlots) * kDecodeSegQ;
  p.tok_part_cnt_offset = static_cast<uint32_t>(cur);
  cur += Align512(part_cnt_entries * static_cast<int64_t>(sizeof(int32_t)));
  p.blk_part_cnt_offset = static_cast<uint32_t>(cur);
  if (mode == kSource) {
    cur += Align512(part_cnt_entries * static_cast<int64_t>(sizeof(int32_t)));
  }

  const int64_t tok_entries =
      static_cast<int64_t>(kDecodeMaxSplitPieceSlots) * kDecodeTokSlotStride;
  p.tok_part_val_offset = static_cast<uint32_t>(cur);
  cur += Align512(tok_entries * static_cast<int64_t>(sizeof(float))) + 512;
  p.tok_part_idx_offset = static_cast<uint32_t>(cur);
  cur += Align512(tok_entries * static_cast<int64_t>(sizeof(int32_t))) + 512;

  p.core_blk_val_offset = static_cast<uint32_t>(cur);
  p.core_blk_idx_offset = static_cast<uint32_t>(cur);
  p.blk_part_val_offset = static_cast<uint32_t>(cur);
  p.blk_part_idx_offset = static_cast<uint32_t>(cur);
  if (mode == kSource) {
    if (ncap > 1) {
      const int64_t core_blk_entries =
          static_cast<int64_t>(eff_cores) * kDecodeCoreBlkStride;
      p.core_blk_val_offset = static_cast<uint32_t>(cur);
      cur += Align512(core_blk_entries * static_cast<int64_t>(sizeof(float))) +
             512;
      p.core_blk_idx_offset = static_cast<uint32_t>(cur);
      cur +=
          Align512(core_blk_entries * static_cast<int64_t>(sizeof(int32_t))) +
          512;
    }
    const int64_t part_blk_entries =
        static_cast<int64_t>(kDecodeMaxSplitPieceSlots) * kDecodeBlkSlotStride;
    p.blk_part_val_offset = static_cast<uint32_t>(cur);
    cur +=
        Align512(part_blk_entries * static_cast<int64_t>(sizeof(float))) + 512;
    p.blk_part_idx_offset = static_cast<uint32_t>(cur);
    cur += Align512(part_blk_entries * static_cast<int64_t>(sizeof(int32_t))) +
           512;
  }

  p.union_scratch_offset = static_cast<uint32_t>(cur);
  if (p.funion) {
    // Per-core [uid: kDecodeFilterUnionCap][ubits: kDecodeFilterUnionCap]
    const int64_t union_ints =
        static_cast<int64_t>(eff_cores) * kDecodeCoreUnionStride;
    cur += Align512(union_ints * static_cast<int64_t>(sizeof(int32_t)));
  }

  p.bytes = cur < 4096 ? 4096 : cur;
  return p;
}

struct DecodeParams {
  const void* query;                    // bf16 [Ncap, 32, 128]
  const void* weights;                  // bf16 [Ncap, 32]
  const void* paged_k;                  // bf16 [Nphys, 128]
  const int32_t* query_positions;       // int32 [Ncap]
  const int32_t* query_batch_indices;   // int32 [Ncap]
  const int32_t* num_valid_input_rows;  // int32 [1] on device
  const int32_t* paged_block_table;     // int32 [Bcap, P]
  const int32_t* history_lengths;       // int32 [Bcap]
  const int32_t* candidate_ids;         // int32 [Ncap, 2048] or nullptr
  int32_t* topk_indices;                // int32 [Ncap, 512]
  int32_t* candidate_ids_out;           // int32 [Ncap, 2048] or nullptr
  uint8_t* workspace;
  int64_t ncap;
  int64_t bcap;
  int64_t max_pages;
  int64_t paged_rows;
  int mode;
  PageLayout page;
  int ratio;
  int logits_dtype;
  DecodePlan plan;
};

int64_t DecodeWorkspaceBytes(int64_t ncap,
                             int64_t bcap,
                             int mode,
                             int cores,
                             int sched_override,
                             int logits_dtype = kLogitsFp32);

void LaunchPagedDecodeIndexScoreSelect(cnrtQueue_t queue,
                                       const DecodeParams& params,
                                       int clusters,
                                       int cores);

}  // namespace xllm::kernel::mlu::bangc::score_select

#endif  // INDEX_SCORE_SELECT_KERNELS_INDEX_SCORE_SELECT_H_
