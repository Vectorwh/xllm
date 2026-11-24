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

#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "disagg_pd.pb.h"
#include "framework/constraint/xgrammar_support.h"
#include "framework/sampling/json_object_grammar.h"
#include "runtime/xservice_client.h"

namespace xllm {

class Engine;
class Request;
class DisaggPDScheduler;

// a class to handle disagg_pd requests
class DisaggPDServiceImpl {
 public:
  explicit DisaggPDServiceImpl(DisaggPDScheduler* scheduler, Engine* engine);
  ~DisaggPDServiceImpl() = default;

  virtual void decode_recv_new_requests(const proto::DisaggRequests* request,
                                        proto::DisaggResponses* response);

  virtual void decode_recv_first_generation(
      const proto::DisaggGenerationsRequests* request,
      proto::Status* response);

  virtual void link_instance(const proto::InstanceClusterInfo* request,
                             proto::Status* response);

  virtual void unlink_instance(const proto::InstanceClusterInfo* request,
                               proto::Status* response);

  void release_reservation(const proto::ReleaseReservationRequest* request,
                           proto::ReleaseReservationResponse* response);

  void cancel_request(const proto::CancelRequestRequest* request,
                      proto::CancelRequestResponse* response);

  void mark_kv_unsafe(const proto::MarkKVUnsafeRequest* request,
                      proto::MarkKVUnsafeResponse* response);

 protected:
  // Rebuilds the request for the decode instance. On a nullptr return,
  // `constraint_rejected` (optional) reports whether the failure was a
  // structured-output constraint rebuild failure: the migration cannot be
  // retried into a valid decode, so the caller reports a permanent 422 for
  // it while every other nullptr path (config, multimodal metadata, ...)
  // stays a generic decode error.
  std::shared_ptr<Request> generate_request(
      const proto::DisaggRequest& req,
      bool* constraint_rejected = nullptr);

  std::shared_ptr<const JsonObjectGrammar> get_json_object_grammar(
      bool reasoning_enabled,
      std::string* error);

  DisaggPDScheduler* scheduler_;  // not owned
  Engine* engine_;                // not owned
  XServiceClient* xservice_client_ = nullptr;
  std::mutex json_object_grammar_mutex_;
  std::shared_ptr<const JsonObjectGrammar> json_object_grammar_;
  std::shared_ptr<const JsonObjectGrammar> json_reasoning_grammar_;
#if defined(USE_XGRAMMAR)
  // Decode-side pool of xgrammar machines for migrated requests: rebuilds
  // the exact per-request grammar from the shipped spec (G9).
  std::unique_ptr<XGrammarMachineCache> xgrammar_machines_;
#endif
};

}  // namespace xllm
