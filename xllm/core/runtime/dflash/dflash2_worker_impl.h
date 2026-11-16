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

#pragma once

#include <optional>

#include "runtime/dflash/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

class DFlash2WorkerImpl final : public DFlashWorkerImpl {
 public:
  DFlash2WorkerImpl(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options,
                    WorkerType worker_type);

 protected:
  DraftBlock run_decode_draft(const ForwardInput& input,
                              ForwardInput& validate_input) override;
  DraftBlock run_decode_draft_from_pending(
      const ForwardInput& input,
      const std::vector<int64_t>& rows) override;

 private:
  torch::Tensor prepare_draft_noise(const ForwardInput& input) const;
  DraftBlock execute_draft_query(const ForwardInput& input,
                                 ForwardInput query_input,
                                 const torch::Tensor& anchor_token_ids,
                                 const torch::Tensor& gumbel_noise,
                                 bool prepared);

  struct BlockSampleOutput {
    torch::Tensor token_ids;
    torch::Tensor dense_probs;
  };

  static std::optional<BlockSampleOutput> fused_sample(
      const torch::Tensor& candidate_ids,
      const torch::Tensor& edge_log_probs,
      const torch::Tensor& gumbel_noise,
      const SamplingParameters& sampling_params,
      int64_t vocab_size);

  BlockSampleOutput sample_path(const DFlash2CandidateOutput& candidates,
                                const SamplingParameters& sampling_params,
                                const torch::Tensor& gumbel_noise,
                                int64_t vocab_size) const;

  ProcessGroup* sampling_process_group_ = nullptr;
};

}  // namespace xllm
