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

#pragma once

#include "framework/kv_cache_transfer/mooncake_store_backend.h"

namespace xllm::internal {

std::optional<std::vector<StoreRegisteredRange>> unique_ranges(
    const std::vector<StoreRegisteredRange>& ranges);
std::optional<int32_t> get_bytes(const MultiBuffer& buffer);
std::vector<uint8_t> get_statuses(const std::vector<int32_t>& expected_bytes,
                                  const std::vector<int>& results);
bool put_succeeded(int result);
MooncakeReplicaTier replica_tier(
    const std::vector<mooncake::Replica::Descriptor>& replicas);

}  // namespace xllm::internal
