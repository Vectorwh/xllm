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

#include "layers/mlu/deepseek_v41/deepseek_v41_indexer.h"

#include <algorithm>

#include "kernels/mlu/bangc/include/deepseek_v41.h"

namespace xllm::layer {

DeepseekV41IndexerImpl::DeepseekV41IndexerImpl(const ModelContext& context,
                                               int32_t layer_id)
    : args_(context.get_model_args()),
      layer_id_(layer_id),
      ratio_(args_.compress_ratios().at(layer_id)),
      options_(context.get_tensor_options()) {
  const auto& kv_sources = args_.kv_source_layer_ids();
  const auto& index_sources = args_.index_source_layer_ids();
  const bool writes_k = ratio_ > 0 && std::find(kv_sources.begin(),
                                                kv_sources.end(),
                                                layer_id_) != kv_sources.end();
  const bool writes_head =
      ratio_ > 0 &&
      std::find(index_sources.begin(), index_sources.end(), layer_id_) !=
          index_sources.end();
  if (writes_k || writes_head) {
    CHECK_EQ(options_.dtype().toScalarType(), torch::kBFloat16);
    CHECK_EQ(args_.index_head_dim(), 128);
    CHECK_EQ(args_.qk_rope_head_dim(), 64);
  }
  if (writes_head) {
    CHECK_EQ(args_.hidden_size(), 5120);
    CHECK_EQ(args_.index_n_heads(), 32);
    CHECK_EQ(args_.index_topk(), 512)
        << "MLU index selection at layer " << layer_id_;
    const int32_t source = args_.candidate_source_layer_id();
    if (source >= 0 && layer_id_ >= source) {
      CHECK_EQ(args_.candidate_block_size(), 8)
          << "MLU candidate selection at layer " << layer_id_;
      CHECK_EQ(args_.candidate_topk_blocks(), 2048)
          << "MLU candidate selection at layer " << layer_id_;
    }
  }
  if (writes_k) {
    CHECK_EQ(args_.head_dim(), 512);
    CHECK_EQ(args_.rms_norm_eps(), 1e-20f);
  }
}

void DeepseekV41IndexerImpl::load_state_dict(const StateDict& state) {
  for (const std::string name :
       {"wq_b.weight", "wk.weight", "k_norm.weight", "weights_proj.weight"}) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    weights_[name] = value.to(options_.device()).contiguous();
  }
}

void DeepseekV41IndexerImpl::verify_loaded_weights() const {
  const std::string prefix =
      "layers." + std::to_string(layer_id_) + ".attn.indexer.";
  const auto& index_sources = args_.index_source_layer_ids();
  if (std::find(index_sources.begin(), index_sources.end(), layer_id_) !=
      index_sources.end()) {
    for (const std::string name : {"wq_b.weight", "weights_proj.weight"}) {
      CHECK(weights_.count(name)) << "Missing weight " << prefix + name;
    }
  }
  const auto& kv_sources = args_.kv_source_layer_ids();
  if (ratio_ > 0 &&
      std::find(kv_sources.begin(), kv_sources.end(), layer_id_) !=
          kv_sources.end()) {
    for (const std::string name : {"wk.weight", "k_norm.weight"}) {
      CHECK(weights_.count(name)) << "Missing weight " << prefix + name;
    }
  }
}

torch::Tensor DeepseekV41IndexerImpl::project_query(
    const torch::Tensor& qr,
    const torch::Tensor& positions,
    const torch::Tensor& rope) {
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  return kernel::mlu::bangc::fused_indexer_q(
      qr,
      weights_.at("wq_b.weight"),
      positions.to(torch::kInt32).contiguous(),
      rope);
}

torch::Tensor DeepseekV41IndexerImpl::project_indexer(
    const torch::Tensor& x,
    const torch::Tensor& latent,
    const torch::Tensor& positions,
    const torch::Tensor& slots,
    const torch::Tensor& rope,
    const torch::Tensor& cache,
    const std::optional<torch::Tensor>& num_valid_input_rows,
    const std::optional<torch::Tensor>& num_valid_compressed_rows) {
  const bool has_head = x.defined() && x.size(0) > 0;
  const bool has_k = latent.defined() && latent.size(0) > 0;
  torch::Tensor head = torch::empty({x.defined() ? x.size(0) : 0, 32},
                                    options_.dtype(torch::kBFloat16));
  return kernel::mlu::bangc::fused_indexer_k(
      x,
      latent,
      has_head ? weights_.at("weights_proj.weight") : torch::Tensor(),
      has_k ? weights_.at("wk.weight") : torch::Tensor(),
      has_k ? weights_.at("k_norm.weight") : torch::Tensor(),
      rope,
      positions,
      slots,
      cache,
      head,
      num_valid_input_rows,
      num_valid_compressed_rows);
}

void DeepseekV41IndexerImpl::select_indices(const torch::Tensor& query,
                                            const torch::Tensor& weights,
                                            const torch::Tensor& cache,
                                            const torch::Tensor& positions,
                                            const torch::Tensor& requests,
                                            bool decode,
                                            Csa2Context& shared) {
  namespace bangc = kernel::mlu::bangc;
  using Mode = bangc::ScoreSelectMode;
  const int32_t source = args_.candidate_source_layer_id();
  const Mode mode = source < 0 || layer_id_ < source
                        ? Mode::NONE
                        : (layer_id_ == source ? Mode::SOURCE : Mode::FILTER);
  const auto phase = decode ? bangc::ScoreSelectPhase::DECODE
                            : bangc::ScoreSelectPhase::PREFILL;
  const auto& table = shared.block_tables.at(1);
  const int64_t rows = query.size(0), batch = table.size(0);
  deepseek_v41::prepare_selection(
      shared, rows, batch, source >= 0, phase, query.options());
  bangc::ScoreSelectInputs in{
      query,
      weights,
      cache.view({-1, 128}),
      positions,
      requests,
      shared.num_valid_input_rows,
      table,
      shared.index_history.at(ratio_ - 1).narrow(0, 0, batch),
      mode == Mode::FILTER ? shared.packed_candidates : torch::Tensor()};
  bangc::fused_dsa_index_score_select(
      in,
      ratio_,
      mode,
      phase,
      shared.packed_topk,
      mode == Mode::SOURCE ? shared.packed_candidates : torch::Tensor(),
      shared.select_workspace);
}
}  // namespace xllm::layer
