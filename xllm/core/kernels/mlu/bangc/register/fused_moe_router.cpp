/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "kernels/mlu/bangc/include/fused_moe_router.h"

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(xllm_mlu, m) {
  m.def("pack_moe_router_weight(Tensor weight) -> Tensor");
  m.def(
      "fused_moe_router(Tensor hidden_states, Tensor packed_weight, Tensor "
      "bias, int topk, float route_scale, Tensor? input_ids=None, "
      "Tensor? bias_vl=None, int image_token_id=-1) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(xllm_mlu, PrivateUse1, m) {
  m.impl("pack_moe_router_weight",
         TORCH_FN(xllm::kernel::mlu::bangc::pack_moe_router_weight));
  m.impl("fused_moe_router",
         TORCH_FN(xllm::kernel::mlu::bangc::fused_moe_router));
}
