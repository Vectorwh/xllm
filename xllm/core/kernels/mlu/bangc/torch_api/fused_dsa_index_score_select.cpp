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

#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <limits>

#include "kernels/mlu/bangc/include/deepseek_v41.h"
#include "kernels/mlu/bangc/include/deepseek_v41_constants.h"
#include "kernels/mlu/bangc/src/deepseek_v41/fused_dsa_index_score_select/index_score_select.h"

namespace xllm::kernel::mlu::bangc {
namespace {

void check_tensor(const torch::Tensor& t,
                  const torch::Device& device,
                  torch::ScalarType dtype,
                  torch::IntArrayRef shape) {
  CHECK(t.defined() && t.device() == device && t.scalar_type() == dtype &&
        t.is_contiguous() && t.sizes() == shape);
}
void device_geometry(int32_t& clusters, int32_t& cores) {
  int32_t device = 0;
  CNRT_CHECK(cnrtGetDevice(&device));
  CNRT_CHECK(cnrtDeviceGetAttribute(&clusters, cnrtAttrClusterCount, device));
  CNRT_CHECK(cnrtDeviceGetAttribute(&cores, cnrtAttrMcorePerCluster, device));
}
template <typename Params>
void set_inputs(Params& p,
                const ScoreSelectInputs& in,
                int32_t ratio,
                ScoreSelectMode mode,
                torch::Tensor topk,
                torch::Tensor candidates_out,
                torch::Tensor workspace) {
  p.query = in.query.data_ptr();
  p.weights = in.weights.data_ptr();
  p.paged_k = in.cache.data_ptr();
  p.query_positions = in.positions.data_ptr<int32_t>();
  p.query_batch_indices = in.requests.data_ptr<int32_t>();
  p.num_valid_input_rows = in.valid_rows.data_ptr<int32_t>();
  p.paged_block_table = in.table.data_ptr<int32_t>();
  p.history_lengths = in.history.data_ptr<int32_t>();
  p.candidate_ids =
      in.candidates.defined() ? in.candidates.data_ptr<int32_t>() : nullptr;
  p.topk_indices = topk.data_ptr<int32_t>();
  p.candidate_ids_out =
      candidates_out.defined() ? candidates_out.data_ptr<int32_t>() : nullptr;
  p.workspace = workspace.data_ptr<uint8_t>();
  p.max_pages = in.table.size(1);
  p.paged_rows = in.cache.size(0);
  const int32_t rows = kDeepseekV41MluKVBlockSize / ratio;
  const int32_t blocks = rows / score_select::kBlockSize;
  p.page = {
      rows,
      blocks == 16 ? 4 : 3,
      blocks - 1,
      rows * score_select::kDim * static_cast<int32_t>(sizeof(torch::BFloat16)),
      1.0f / blocks};
  p.ratio = ratio;
  p.mode = static_cast<int32_t>(mode);
  p.logits_dtype = score_select::kLogitsBf16;
}
}  // namespace
int64_t score_select_workspace(int64_t rows,
                               int64_t batch,
                               ScoreSelectMode mode,
                               ScoreSelectPhase phase) {
  CHECK_GE(rows, 0);
  CHECK_LE(rows, std::numeric_limits<int32_t>::max());
  CHECK_GE(batch, 0);
  CHECK_LE(batch, std::numeric_limits<int32_t>::max());
  int32_t clusters, cores;
  device_geometry(clusters, cores);
  const int32_t m = static_cast<int32_t>(mode);
  return phase == ScoreSelectPhase::DECODE
             ? score_select::DecodeWorkspaceBytes(rows,
                                                  batch,
                                                  m,
                                                  clusters * cores,
                                                  0,
                                                  score_select::kLogitsBf16)
             : score_select::PagedWorkspaceBytes(rows,
                                                 batch,
                                                 m,
                                                 clusters * cores,
                                                 0,
                                                 score_select::kLogitsBf16);
}
void fused_dsa_index_score_select(const ScoreSelectInputs& in,
                                  int32_t ratio,
                                  ScoreSelectMode mode,
                                  ScoreSelectPhase phase,
                                  torch::Tensor topk,
                                  torch::Tensor candidates_out,
                                  torch::Tensor workspace) {
  const auto device = in.query.device();
  CHECK(device.type() == torch::kPrivateUse1);
  const torch::DeviceGuard guard(device);
  const int64_t rows = in.query.size(0), batch = in.table.size(0);
  CHECK_EQ(in.table.dim(), 2);
  CHECK(ratio == 1 || ratio == 2);
  CHECK_LE(in.table.size(1), (1 << 22) / kDeepseekV41MluKVBlockSize);
  check_tensor(in.query, device, torch::kBFloat16, {rows, 32, 128});
  check_tensor(in.weights, device, torch::kBFloat16, {rows, 32});
  check_tensor(in.cache, device, torch::kBFloat16, {in.cache.size(0), 128});
  check_tensor(in.positions, device, torch::kInt32, {rows});
  check_tensor(in.requests, device, torch::kInt32, {rows});
  check_tensor(in.valid_rows, device, torch::kInt32, {1});
  check_tensor(in.history, device, torch::kInt32, {batch});
  check_tensor(in.table, device, torch::kInt32, {batch, in.table.size(1)});
  check_tensor(topk, device, torch::kInt32, {rows, 512});
  CHECK_EQ(in.candidates.defined(), mode == ScoreSelectMode::FILTER);
  CHECK_EQ(candidates_out.defined(), mode == ScoreSelectMode::SOURCE);
  if (in.candidates.defined()) {
    check_tensor(in.candidates, device, torch::kInt32, {rows, 2048});
  }
  if (candidates_out.defined()) {
    check_tensor(candidates_out, device, torch::kInt32, {rows, 2048});
  }
  check_tensor(workspace, device, torch::kUInt8, {workspace.numel()});
  CHECK_GE(workspace.numel(), score_select_workspace(rows, batch, mode, phase));
  const torch::Tensor* tensors[] = {&in.query,
                                    &in.weights,
                                    &in.cache,
                                    &in.positions,
                                    &in.requests,
                                    &in.valid_rows,
                                    &in.table,
                                    &in.history,
                                    &in.candidates,
                                    &topk,
                                    &candidates_out,
                                    &workspace};
  for (size_t i = 0; i < std::size(tensors); ++i) {
    const auto& a = *tensors[i];
    if (!a.defined() || a.numel() == 0) {
      continue;
    }
    const uintptr_t start = reinterpret_cast<uintptr_t>(a.data_ptr());
    CHECK_LE(a.nbytes(), std::numeric_limits<uintptr_t>::max() - start);
    for (size_t j = 0; j < i; ++j) {
      const auto& b = *tensors[j];
      if (!b.defined() || b.numel() == 0) {
        continue;
      }
      const uintptr_t other = reinterpret_cast<uintptr_t>(b.data_ptr());
      CHECK(start + a.nbytes() <= other || other + b.nbytes() <= start)
          << "overlapping score selection tensors";
    }
  }
  if (rows == 0) {
    return;
  }
  int32_t clusters, cores;
  device_geometry(clusters, cores);
  if (phase == ScoreSelectPhase::PREFILL) {
    score_select::PagedParams p{};
    set_inputs(p, in, ratio, mode, topk, candidates_out, workspace);
    p.nq = rows;
    p.batch = batch;
    p.plan =
        score_select::MakePagedPlan(rows, batch, p.mode, clusters * cores, 0);
    score_select::LaunchPagedPrefillIndexScoreSelect(
        torch_mlu::getCurMLUStream(), p, clusters, cores);
  } else {
    score_select::DecodeParams p{};
    set_inputs(p, in, ratio, mode, topk, candidates_out, workspace);
    p.ncap = rows;
    p.bcap = batch;
    p.plan =
        score_select::MakeDecodePlan(rows, batch, p.mode, clusters * cores, 0);
    score_select::LaunchPagedDecodeIndexScoreSelect(
        torch_mlu::getCurMLUStream(), p, clusters, cores);
  }
  CHECK_EQ(cnrtGetLastError(), cnrtSuccess);
}
}  // namespace xllm::kernel::mlu::bangc
