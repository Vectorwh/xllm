/* Copyright 2026 The xLLM Authors.

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

#include <ATen/MemoryOverlap.h>
#include <glog/logging.h>

#include <cmath>
#include <limits>

#include "framework/core/MLUStream.h"
#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "kernels/mlu/bangc/include/deepseek_v41_constants.h"
#include "kernels/mlu/bangc/src/deepseek_v41/fused_paged_sparsed_attention/launch.h"
#include "kernels/mlu/bangc/torch_api/tensor_checks.h"

namespace xllm::kernel::mlu::bangc {
namespace {
using namespace paged_attention;

void check_rows(const torch::Tensor& t, const torch::Device& device) {
  CHECK(detail::check_tensor(t, torch::kBFloat16, device));
  CHECK_EQ(t.dim(), 2);
  CHECK_EQ(t.size(1), kDim);
}

const int32_t* int_vector(const torch::Tensor& t,
                          int64_t rows,
                          const torch::Device& device) {
  CHECK(detail::check_tensor(t, torch::kInt32, device));
  CHECK(t.sizes() == torch::IntArrayRef({rows}));
  return t.data_ptr<int32_t>();
}

void check_table(const torch::Tensor& t,
                 int64_t rows,
                 const torch::Device& device) {
  CHECK(detail::check_tensor(t, torch::kInt32, device, false));
  CHECK_EQ(t.dim(), 2);
  CHECK_EQ(t.size(0), rows);
  CHECK_EQ(t.stride(1), 1);
  CHECK_GE(t.stride(0), t.size(1));
}
}  // namespace

torch::Tensor fused_paged_sparsed_attention(PagedAttentionStage stage,
                                            const torch::Tensor& q,
                                            const torch::Tensor& current,
                                            const torch::Tensor& swa,
                                            const torch::Tensor& compressed,
                                            const torch::Tensor& sink,
                                            const PagedAttentionMetadata& m,
                                            int32_t ratio,
                                            int32_t window) {
  using namespace paged_attention;
  CHECK(detail::check_tensor(q, torch::kBFloat16));
  const auto device = q.device();
  const torch::DeviceGuard guard(device);
  CHECK_EQ(q.dim(), 3);
  CHECK_EQ(q.size(2), kDim);
  const int64_t heads = q.size(1);
  CHECK(heads == 8 || heads == 16 || heads == 32 || heads == 64);
  check_rows(current, device);
  check_rows(swa, device);
  CHECK_EQ(swa.size(0) % kPage, 0);
  CHECK(detail::check_tensor(sink, torch::kFloat32, device));
  CHECK(sink.sizes() == torch::IntArrayRef({heads}));
  CHECK(ratio == 0 || ratio == 1 || ratio == 2);
  CHECK(stage != PagedAttentionStage::DRAFT || ratio == 0);
  CHECK(stage == PagedAttentionStage::PREFILL || current.size(0) == q.size(0));
  CHECK_LE(q.size(0), std::numeric_limits<int32_t>::max() / 2);
  CHECK_LE(current.size(0), std::numeric_limits<int32_t>::max() / 2);
  CHECK_LE(m.begins.numel(), std::numeric_limits<int32_t>::max() / 2);
  Params p{};
  CHECK_GT(window, 0);
  CHECK_LE(window, kWindow);
  p.window = window;
  p.nq = static_cast<int32_t>(q.size(0));
  p.nkv = static_cast<int32_t>(current.size(0));
  p.heads = static_cast<int32_t>(heads);
  p.requests = static_cast<int32_t>(m.begins.numel());
  p.draft = stage == PagedAttentionStage::DRAFT;
  p.ratio = ratio;
  p.scale = 1.0f / std::sqrt(static_cast<float>(kDim));
  p.q = q.data_ptr();
  p.current = current.data_ptr();
  p.swa = swa.data_ptr();
  p.sink = sink.data_ptr<float>();
  p.num_valid_rows = int_vector(m.valid_rows, 1, device);
  p.logical_swa_columns = int_vector(m.swa_columns, 1, device);
  p.kv_prefix = int_vector(m.prefix, p.requests + 1, device);
  p.begins = int_vector(m.begins, p.requests, device);
  if (stage == PagedAttentionStage::PREFILL) {
    p.request_ids = int_vector(m.requests, p.nq, device);
    p.positions = int_vector(m.positions, p.nq, device);
  }
  check_table(m.swa_table, p.requests, device);
  CHECK_GT(m.swa_table.size(1), 0);
  CHECK_LE(m.swa_table.size(1), kDeepseekV41MaxSwaColumns);
  p.swa_table = m.swa_table.data_ptr<int32_t>();
  p.swa_row_stride = m.swa_table.stride(0);
  if (ratio > 0) {
    check_rows(compressed, device);
    p.compressed_page_shift = ratio == 2 ? 6 : 7;
    CHECK_EQ(compressed.size(0) % (kPage / ratio), 0);
    check_table(m.compressed_table, p.requests, device);
    CHECK_LE(m.compressed_table.size(1), kMaxCompressedColumns);
    CHECK(detail::check_tensor(m.topk, torch::kInt32, device));
    CHECK_EQ(m.topk.dim(), 2);
    CHECK_EQ(m.topk.size(0), p.nq);
    CHECK_LE(m.topk.size(1), kMaxTopk);
    p.kcap = static_cast<int32_t>(m.topk.size(1));
    p.compressed = compressed.data_ptr();
    p.compressed_table = m.compressed_table.data_ptr<int32_t>();
    p.compressed_columns = static_cast<int32_t>(m.compressed_table.size(1));
    p.compressed_row_stride = m.compressed_table.stride(0);
    p.topk = m.topk.data_ptr<int32_t>();
  }
  if (!p.draft) {
    at::assert_no_overlap(swa, current);
    at::assert_no_overlap(swa, q);
    if (ratio > 0) {
      at::assert_no_overlap(swa, compressed);
    }
  }
  auto output = torch::empty_like(q);
  p.output = output.data_ptr();
  const auto queue = torch_mlu::getCurMLUStream(device.index());
  if (stage == PagedAttentionStage::PREFILL || heads < kHeads ||
      window != kWindow) {
    CHECK_EQ(launch_prefill_attention(queue, p), cnrtSuccess);
  } else if (p.draft) {
    CHECK_EQ(launch_draft_attention(queue, p), cnrtSuccess);
  } else if (stage == PagedAttentionStage::VERIFY) {
    CHECK_EQ(launch_verify(queue, p), cnrtSuccess);
  } else {
    CHECK_EQ(launch_decode(queue, p), cnrtSuccess);
  }
  return output;
}
}  // namespace xllm::kernel::mlu::bangc
