/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0; see LICENSE. */
#pragma once

#include <cnrt.h>
#include <stdint.h>

namespace xllm::kernel::mlu::bangc::paged_attention {

constexpr int32_t kHeads = 64;
constexpr int32_t kDim = 512;
constexpr int32_t kWindow = 128;
constexpr int32_t kPage = 128;
constexpr int32_t kMaxTopk = 512;
// Each KV table column spans 128 original tokens at either compression ratio.
constexpr int32_t kMaxCompressedColumns = 8192;
constexpr int32_t kMaxSwaPreloadColumns = 256;

struct Params {
  const void* q;
  const void* current;
  void* swa;
  const void* compressed;
  const int32_t* swa_table;
  const int32_t* compressed_table;
  const int32_t* topk;
  const float* sink;
  const int32_t* request_ids;
  const int32_t* positions;
  const int32_t* kv_prefix;
  const int32_t* begins;
  const int32_t* num_valid_rows;
  void* output;
  const int32_t* logical_swa_columns;
  int32_t heads;
  int32_t window;
  bool draft;
  int32_t nq;
  int32_t nkv;
  int32_t kcap;
  int32_t ratio;
  int32_t swa_columns;
  int32_t compressed_columns;
  int32_t compressed_page_shift;
  int32_t requests;
  int64_t swa_row_stride;
  int64_t compressed_row_stride;
  float scale;
};

cnrtRet_t launch_generic_attention(cnrtQueue_t queue, const Params& params);
cnrtRet_t launch_prefill_attention(cnrtQueue_t queue, const Params& params);
cnrtRet_t launch_swa_commit(cnrtQueue_t queue, const Params& params);
cnrtRet_t launch_decode(cnrtQueue_t queue, const Params& params);
cnrtRet_t launch_verify(cnrtQueue_t queue, const Params& params);
cnrtRet_t launch_draft_attention(cnrtQueue_t queue, const Params& params);
}  // namespace xllm::kernel::mlu::bangc::paged_attention
