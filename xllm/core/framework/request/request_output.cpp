/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "request_output.h"

#include "core/common/metrics.h"

namespace xllm {

void RequestOutput::log_request_status() const {
  if (terminal_status_recorded || !status.has_value()) {
    return;
  }

  // The PD Prefill first-token response is not the request's final result.
  if (finished_on_prefill_instance && status->ok() && !cancelled) {
    return;
  }
  if (cancelled && status->ok()) {
    COUNTER_INC(request_status_total_cancelled);
    return;
  }

  record_request_status(status->code());
}

}  // namespace xllm
