/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <cstring>

#include "core/framework/multimodal/mm_data.h"
#include "core/util/hash_util.h"

namespace xllm::tests {

inline MMData make_image_mm_data(int32_t offset,
                                 int32_t length,
                                 uint8_t key_byte) {
  MMData mm_data;
  auto& item = mm_data.add(MMType::IMAGE);
  item.mutable_state().mutable_token_pos() = {offset, length};
  std::memset(item.mutable_state().mutable_schedule_data().key.data,
              key_byte,
              XXH3_128BITS_HASH_VALUE_LEN);
  return mm_data;
}

}  // namespace xllm::tests
