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

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "framework/kv_cache_transfer/mooncake_store_backend_internal.h"

namespace xllm {

namespace {

enum class ReplicaKind : uint8_t {
  MEMORY,
  NOF,
  DISK,
  LOCAL_DISK,
};

mooncake::Replica::Descriptor make_replica(ReplicaKind kind,
                                           mooncake::ReplicaStatus status) {
  mooncake::Replica::Descriptor replica;
  replica.id = 0;
  replica.status = status;
  switch (kind) {
    case ReplicaKind::MEMORY:
      replica.descriptor_variant = mooncake::MemoryDescriptor{};
      break;
    case ReplicaKind::NOF:
      replica.descriptor_variant = mooncake::NoFDescriptor{};
      break;
    case ReplicaKind::DISK:
      replica.descriptor_variant = mooncake::DiskDescriptor{};
      break;
    case ReplicaKind::LOCAL_DISK:
      replica.descriptor_variant = mooncake::LocalDiskDescriptor{};
      break;
  }
  return replica;
}

TEST(MooncakeStoreBackendTest, ValidatesGetBufferByteLimits) {
  int32_t value = 0;
  const size_t max_bytes =
      static_cast<size_t>(std::numeric_limits<int32_t>::max());
  EXPECT_EQ(internal::get_bytes({{}, {}}), 0);
  EXPECT_FALSE(internal::get_bytes({{&value}, {}}));
  EXPECT_FALSE(internal::get_bytes({{}, {4}}));
  EXPECT_EQ(internal::get_bytes({{&value}, {max_bytes}}),
            std::numeric_limits<int32_t>::max());
  EXPECT_FALSE(internal::get_bytes({{&value}, {max_bytes + 1}}));
  EXPECT_FALSE(internal::get_bytes({{&value, &value}, {max_bytes, 1}}));
  EXPECT_FALSE(
      internal::get_bytes({{&value}, {std::numeric_limits<size_t>::max()}}));
  EXPECT_EQ(internal::get_bytes({{&value, &value}, {4, 8}}), 12);
}

TEST(MooncakeStoreBackendTest, MapsGetResultsToExactByteCounts) {
  EXPECT_TRUE(internal::get_statuses({}, {}).empty());
  EXPECT_EQ(internal::get_statuses({32}, {32}), std::vector<uint8_t>({1}));
  EXPECT_EQ(internal::get_statuses({32}, {}), std::vector<uint8_t>({0}));
  EXPECT_EQ(internal::get_statuses({32}, {32, 32}), std::vector<uint8_t>({1}));
  EXPECT_EQ(internal::get_statuses({32, 32, 32, 32, 0}, {32, -1, 31, 33, 0}),
            std::vector<uint8_t>({1, 0, 0, 0, 1}));
  EXPECT_EQ(internal::get_statuses({32, 64}, {32}),
            std::vector<uint8_t>({1, 0}));
}

TEST(MooncakeStoreBackendTest, TreatsExistingObjectsAsPutSuccess) {
  EXPECT_TRUE(internal::put_succeeded(0));
  EXPECT_TRUE(internal::put_succeeded(
      static_cast<int>(mooncake::ErrorCode::OBJECT_ALREADY_EXISTS)));
  EXPECT_FALSE(internal::put_succeeded(-1));
}

TEST(MooncakeStoreBackendTest, ValidatesRegistrationRangesBeforeSetup) {
  const auto valid_ranges =
      internal::unique_ranges({{reinterpret_cast<void*>(0x1000), 64},
                               {reinterpret_cast<void*>(0x1000), 64},
                               {reinterpret_cast<void*>(0x2000), 32}});
  ASSERT_TRUE(valid_ranges.has_value());
  ASSERT_EQ(valid_ranges->size(), 2U);
  EXPECT_EQ((*valid_ranges)[0].address, reinterpret_cast<void*>(0x1000));
  EXPECT_EQ((*valid_ranges)[1].address, reinterpret_cast<void*>(0x2000));

  EXPECT_FALSE(internal::unique_ranges({{nullptr, 1}}).has_value());
  EXPECT_FALSE(internal::unique_ranges({{reinterpret_cast<void*>(0x1000), 0}})
                   .has_value());
  EXPECT_FALSE(internal::unique_ranges({{reinterpret_cast<void*>(0x1000), 64},
                                        {reinterpret_cast<void*>(0x1000), 32}})
                   .has_value());
  EXPECT_FALSE(internal::unique_ranges({{reinterpret_cast<void*>(0x1000), 64},
                                        {reinterpret_cast<void*>(0x1020), 64}})
                   .has_value());
  EXPECT_FALSE(
      internal::unique_ranges({{reinterpret_cast<void*>(UINTPTR_MAX - 7), 8}})
          .has_value());
}

TEST(MooncakeStoreBackendTest, ReplicaTierFollowsCompleteReadPreference) {
  constexpr mooncake::ReplicaStatus kComplete =
      mooncake::ReplicaStatus::COMPLETE;
  constexpr mooncake::ReplicaStatus kProcessing =
      mooncake::ReplicaStatus::PROCESSING;
  const struct {
    const char* name;
    std::vector<mooncake::Replica::Descriptor> replicas;
    MooncakeReplicaTier expected;
  } cases[] = {
      {"no replica", {}, MooncakeReplicaTier::MISSING},
      {"memory",
       {make_replica(ReplicaKind::MEMORY, kComplete)},
       MooncakeReplicaTier::MEMORY},
      {"nof",
       {make_replica(ReplicaKind::NOF, kComplete)},
       MooncakeReplicaTier::DISK},
      {"disk",
       {make_replica(ReplicaKind::DISK, kComplete)},
       MooncakeReplicaTier::DISK},
      {"local disk",
       {make_replica(ReplicaKind::LOCAL_DISK, kComplete)},
       MooncakeReplicaTier::DISK},
      {"memory listed after disk",
       {make_replica(ReplicaKind::DISK, kComplete),
        make_replica(ReplicaKind::MEMORY, kComplete)},
       MooncakeReplicaTier::MEMORY},
      {"memory still being written",
       {make_replica(ReplicaKind::MEMORY, kProcessing),
        make_replica(ReplicaKind::DISK, kComplete)},
       MooncakeReplicaTier::DISK},
      {"only incomplete replicas",
       {make_replica(ReplicaKind::MEMORY, kProcessing),
        make_replica(ReplicaKind::DISK, mooncake::ReplicaStatus::FAILED)},
       MooncakeReplicaTier::MISSING},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    EXPECT_EQ(internal::replica_tier(test_case.replicas), test_case.expected);
  }
}

}  // namespace
}  // namespace xllm
