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

#include "api_service/multimodal_preprocess_service_impl.h"

#include <algorithm>
#include <utility>

#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_codec.h"
#include "core/runtime/xservice_client.h"

namespace xllm {
namespace {
proto::PreprocessCode preprocess_code(StatusCode code) {
  switch (code) {
    case StatusCode::OK:
      return proto::PREPROCESS_CODE_OK;
    case StatusCode::INVALID_ARGUMENT:
      return proto::PREPROCESS_CODE_INVALID_ARGUMENT;
    case StatusCode::RESOURCE_EXHAUSTED:
      return proto::PREPROCESS_CODE_RESOURCE_EXHAUSTED;
    case StatusCode::DEADLINE_EXCEEDED:
      return proto::PREPROCESS_CODE_DEADLINE_EXCEEDED;
    case StatusCode::UNAVAILABLE:
      return proto::PREPROCESS_CODE_UNAVAILABLE;
    case StatusCode::CANCELLED:
      return proto::PREPROCESS_CODE_CANCELLED;
    default:
      return proto::PREPROCESS_CODE_PROCESSING_FAILED;
  }
}

}  // namespace

MultimodalPreprocessServiceImpl::MultimodalPreprocessServiceImpl(
    VLMMaster* master)
    : master_(master) {}

void MultimodalPreprocessServiceImpl::process_request(
    google::protobuf::RpcController* controller,
    const proto::PreprocessRequest* request,
    proto::PreprocessResponse* response,
    google::protobuf::Closure* done) {
  proto::PreprocessResponse identity;
  identity.set_request_uid(request->request_uid());
  const auto registered =
      XServiceClient::get_instance()->registered_provider_snapshot();
  if (registered) {
    const auto& descriptor = registered->first;
    auto* key = identity.mutable_engine();
    key->set_provider_id(descriptor.identity().provider_id());
    key->set_engine_uid(descriptor.identity().engine_uid());
    key->set_profile_digest(descriptor.profile_digest());
    key->set_incarnation_id(registered->second);
  }
  const auto finish =
      [controller, response, done](proto::PreprocessResponse result) {
        if (controller != nullptr && controller->IsCanceled()) {
          result.set_code(proto::PREPROCESS_CODE_CANCELLED);
          result.clear_items();
          result.clear_mm_data();
          result.clear_token_ids();
        }
        *response = std::move(result);
        if (done != nullptr) {
          done->Run();
        }
      };
  const auto fail = [&](proto::PreprocessCode code,
                        const std::string& message) {
    identity.set_code(code);
    identity.set_message(message);
    finish(std::move(identity));
  };

  if (!registered ||
      registered->first.serving().role() != proto::ENGINE_ROLE_PREFILL ||
      std::find(registered->first.serving().api_features().begin(),
                registered->first.serving().api_features().end(),
                "multimodal_preprocess") ==
          registered->first.serving().api_features().end()) {
    fail(proto::PREPROCESS_CODE_UNAVAILABLE,
         "Preprocess provider is not registered");
    return;
  }
  if (request->items_size() > 8) {
    fail(proto::PREPROCESS_CODE_RESOURCE_EXHAUSTED,
         "Preprocess item limit exceeded");
    return;
  }
  if (controller != nullptr && controller->IsCanceled()) {
    fail(proto::PREPROCESS_CODE_CANCELLED, "Preprocess request cancelled");
    return;
  }
  constexpr size_t kMaxRequestUidLength = 256;
  const auto& engine_identity = identity.engine();
  const auto& expected_engine = request->expected_engine();
  if (engine_identity.provider_id() != proto::PROVIDER_ID_XLLM_NATIVE ||
      engine_identity.engine_uid().empty() ||
      engine_identity.incarnation_id().empty() ||
      engine_identity.profile_digest().empty() ||
      engine_identity.provider_id() != expected_engine.provider_id() ||
      engine_identity.engine_uid() != expected_engine.engine_uid() ||
      engine_identity.incarnation_id() != expected_engine.incarnation_id() ||
      engine_identity.profile_digest() != expected_engine.profile_digest()) {
    fail(proto::PREPROCESS_CODE_FENCED, "Preprocess target identity changed");
    return;
  }

  const int32_t image_limit =
      std::clamp<int32_t>(master_->get_image_limit(), 0, 8);
  if (request->request_uid().empty() ||
      request->request_uid().size() > kMaxRequestUidLength ||
      request->model_revision() != registered->first.model().model_revision() ||
      request->items_size() == 0 ||
      (!request->has_prompt_text() && request->items_size() > image_limit)) {
    fail(proto::PREPROCESS_CODE_INVALID_ARGUMENT, "Invalid preprocess request");
    return;
  }
  const bool native_prompt = request->has_prompt_text();
  MMContentVec media;
  if (native_prompt) {
    media.resize(request->items_size(), MMContent("text", ""));
    std::vector<bool> seen(request->items_size(), false);
    int32_t image_count = 0;
    for (const auto& item : request->items()) {
      MMContent content("text", "");
      const uint32_t index = item.mm_item_index();
      if (index >= media.size() || seen[index] || item.has_image() ||
          !decode_media_input(item.media(), content).ok()) {
        fail(proto::PREPROCESS_CODE_INVALID_ARGUMENT,
             "Invalid Native media occurrence");
        return;
      }
      seen[index] = true;
      image_count += content.type == "image_url" ? 1 : 0;
      media[index] = std::move(content);
    }
    if (image_count > image_limit) {
      fail(proto::PREPROCESS_CODE_INVALID_ARGUMENT, "Image limit exceeded");
      return;
    }
  }

  std::string payload;
  if (!native_prompt) {
    media.resize(request->items_size(), MMContent("image_url", ""));
    std::vector<bool> seen(request->items_size(), false);
    for (const auto& item : request->items()) {
      const uint32_t index = item.mm_item_index();
      if (index >= media.size() || seen[index] ||
          !decode_image_media_input(item.image(), media[index], payload).ok()) {
        fail(proto::PREPROCESS_CODE_INVALID_ARGUMENT,
             "Invalid image occurrence");
        return;
      }
      seen[index] = true;
    }
  }
  auto completed = [identity, finish, native_prompt](
                       Status status,
                       MMData processed,
                       std::vector<int32_t> tokens) mutable {
    identity.set_code(preprocess_code(status.code()));
    identity.set_message(status.message());
    if (status.ok() && native_prompt) {
      // No pixels or model tensors cross into Service. The loaded processor
      // owns token expansion, layout and all occurrence spans.
      if (!mmdata_entry_to_proto(processed,
                                 identity.mutable_mm_data(),
                                 /*include_tensors=*/false)) {
        identity.set_code(proto::PREPROCESS_CODE_PROCESSING_FAILED);
        identity.clear_mm_data();
      } else {
        identity.mutable_token_ids()->Add(tokens.begin(), tokens.end());
      }
    }
    if (status.ok()) {
      identity.mutable_items()->Reserve(processed.size());
      for (size_t index = 0; index < processed.size(); ++index) {
        auto* result = identity.add_items();
        result->set_mm_item_index(static_cast<uint32_t>(index));
        const auto& item = processed.items<MMItemVec>()[index];
        if (!(native_prompt ? mmdata_item_to_proto(item,
                                                   result->mutable_media_item(),
                                                   /*include_tensors=*/false)
                            : media_metadata_to_proto(
                                  item, result->mutable_media_item()))) {
          identity.set_code(proto::PREPROCESS_CODE_PROCESSING_FAILED);
          identity.clear_items();
          break;
        }
        result->set_code(proto::PREPROCESS_CODE_OK);
      }
    }
    if (identity.code() != proto::PREPROCESS_CODE_OK) {
      identity.clear_items();
      identity.clear_token_ids();
      identity.clear_mm_data();
    }
    finish(std::move(identity));
  };
  master_->preprocess(std::move(media),
                      native_prompt
                          ? std::optional<std::string>(request->prompt_text())
                          : std::nullopt,
                      BinaryPayload(payload),
                      std::move(completed));
}
}  // namespace xllm
