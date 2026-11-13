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

#include <google/protobuf/service.h>

#include "multimodal_preprocess.pb.h"

namespace xllm {
class VLMMaster;
class MultimodalPreprocessServiceImpl final {
 public:
  explicit MultimodalPreprocessServiceImpl(VLMMaster* master);
  void process_request(google::protobuf::RpcController* controller,
                       const proto::PreprocessRequest* request,
                       proto::PreprocessResponse* response,
                       google::protobuf::Closure* done);

 private:
  VLMMaster* master_;
};
}  // namespace xllm
