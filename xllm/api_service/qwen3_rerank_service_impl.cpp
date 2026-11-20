/* Copyright 2025-2026 The xLLM Authors.

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

#include "api_service/qwen3_rerank_service_impl.h"

#include "api_service/request_admission.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_params.h"

namespace xllm {

Qwen3RerankServiceImpl::Qwen3RerankServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : RerankServiceImpl(master, models) {}

void Qwen3RerankServiceImpl::process_async_impl(
    std::shared_ptr<RerankCall> call) {
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();
  if (rpc_request.documents().empty()) {
    // Keep model errors ahead of empty-input errors without reserving slots.
    const Status status =
        models_.contains(model)
            ? Status::invalid_parameter("documents must not be empty",
                                        "documents")
            : api_service_internal::model_not_supported_status();
    call->finish_with_error(status);
    return;
  }
  const int64_t num_requests =
      static_cast<int64_t>(rpc_request.documents_size());
  const Status admission =
      api_service_internal::admit_rpc_request(model,
                                              models_,
                                              master_->get_rate_limiter(),
                                              /*supports_lora=*/false,
                                              num_requests);
  if (!admission.ok()) {
    call->finish_with_error(admission);
    return;
  }

  auto query = rpc_request.query();
  std::vector<std::string> documents(rpc_request.documents().begin(),
                                     rpc_request.documents().end());

  std::vector<std::string> reqs;
  reqs.reserve(documents.size());
  for (size_t i = 0; i < documents.size(); ++i) {
    reqs.emplace_back(query + documents[i]);
  }

  RequestParams request_params(
      rpc_request, call->get_x_request_id(), call->get_x_request_time());
  std::vector<RequestParams> sps(documents.size(), request_params);
  auto request_id = request_params.request_id;

  int32_t top_n = static_cast<int32_t>(documents.size());
  if (rpc_request.has_top_n()) {
    top_n = std::min(top_n, rpc_request.top_n());
  }

  // Logprobs-based score computer for Qwen3 rerank
  auto compute_scores = [](const std::vector<std::string>& documents,
                           const std::vector<RequestOutput>& req_outputs)
      -> std::vector<RerankRequestOutput> {
    std::vector<RerankRequestOutput> rerank_outputs;
    rerank_outputs.reserve(documents.size());

    for (size_t i = 0; i < documents.size(); ++i) {
      if (req_outputs[i].outputs[0].logprobs.has_value()) {
        auto score = req_outputs[i].outputs[0].logprobs.value()[0].logprob;
        rerank_outputs.emplace_back(i, documents[i], score);
      }
    }
    return rerank_outputs;
  };

  auto ctx = std::make_shared<RerankContext>(call,
                                             std::move(documents),
                                             model,
                                             request_id,
                                             top_n,
                                             sps.size(),
                                             compute_scores);

  auto batch_callback = [ctx](size_t index, RequestOutput output) -> bool {
    ctx->on_complete(index, std::move(output));
    return true;
  };

  master_->handle_batch_request(
      std::move(reqs), std::move(sps), batch_callback);
}

}  // namespace xllm
