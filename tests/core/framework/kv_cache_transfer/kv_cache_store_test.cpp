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

#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/mooncake_store_backend.h"
#include "tests/core/common/scoped_environment_variable.h"

namespace xllm {

namespace {

using StoreObjects = std::unordered_map<std::string, std::vector<uint8_t>>;
class MemoryStoreTransport final : public KVCacheStoreTransport {
 public:
  explicit MemoryStoreTransport(std::shared_ptr<StoreObjects> objects)
      : objects_(std::move(objects)) {}
  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<MultiBuffer>& buffers) override {
    last_put_keys_ = keys;
    std::vector<uint8_t> results(keys.size(), /*value=*/0);
    for (size_t index = 0; index < keys.size(); ++index) {
      if (objects_->count(keys[index]) != 0) {
        results[index] = 1;
        continue;
      }
      if (failed_puts_.count(keys[index]) != 0) {
        continue;
      }
      const auto& buffer = buffers[index];
      size_t bytes = 0;
      for (size_t size : buffer.sizes) {
        bytes += size;
      }
      auto& object = (*objects_)[keys[index]];
      object.resize(bytes);
      size_t offset = 0;
      for (size_t part = 0; part < buffer.sizes.size(); ++part) {
        std::memcpy(
            object.data() + offset, buffer.addresses[part], buffer.sizes[part]);
        offset += buffer.sizes[part];
      }
      results[index] = 1;
    }
    if (result_count_.has_value()) {
      results.resize(*result_count_);
    }
    return results;
  }
  std::vector<uint8_t> batch_get(
      const std::vector<std::string>& keys,
      const std::vector<MultiBuffer>& buffers) override {
    ++batch_get_calls_;
    std::vector<uint8_t> results(keys.size(), /*value=*/0);
    for (size_t index = 0; index < keys.size(); ++index) {
      if (failed_gets_.count(keys[index]) != 0) {
        continue;
      }
      const auto object = objects_->find(keys[index]);
      if (object == objects_->end()) {
        continue;
      }
      const auto& buffer = buffers[index];
      size_t bytes = 0;
      for (size_t size : buffer.sizes) {
        bytes += size;
      }
      if (object->second.size() != bytes) {
        continue;
      }
      size_t offset = 0;
      for (size_t part = 0; part < buffer.sizes.size(); ++part) {
        std::memcpy(buffer.addresses[part],
                    object->second.data() + offset,
                    buffer.sizes[part]);
        offset += buffer.sizes[part];
      }
      results[index] = 1;
    }
    if (result_count_.has_value()) {
      results.resize(*result_count_);
    }
    return results;
  }
  std::vector<MooncakeReplicaTier> batch_query_tiers(
      const std::vector<std::string>& keys) override {
    last_query_keys_ = keys;
    std::vector<MooncakeReplicaTier> tiers;
    tiers.reserve(keys.size());
    for (const std::string& key : keys) {
      const auto override_tier = tiers_.find(key);
      if (override_tier != tiers_.end()) {
        tiers.emplace_back(override_tier->second);
        continue;
      }
      tiers.emplace_back(objects_->count(key) != 0
                             ? MooncakeReplicaTier::MEMORY
                             : MooncakeReplicaTier::MISSING);
    }
    return tiers;
  }
  std::vector<std::string> last_put_keys_;
  std::vector<std::string> last_query_keys_;
  std::unordered_map<std::string, MooncakeReplicaTier> tiers_;
  MooncakeStoreBackendConfig config_;
  std::vector<StoreRegisteredRange> ranges_;
  std::optional<size_t> result_count_;
  std::unordered_set<std::string> failed_gets_;
  std::unordered_set<std::string> failed_puts_;
  int32_t batch_get_calls_ = 0;

 private:
  std::shared_ptr<StoreObjects> objects_;
};

// Only this unit-test executable supplies these backend definitions. The
// separate integration target links the real Mooncake backend. Store itself
// always runs its public init/put/get/exist paths, including tensor
// registration.
std::shared_ptr<MemoryStoreTransport> pending_transport;
std::unordered_map<const MooncakeStoreBackend*,
                   std::shared_ptr<MemoryStoreTransport>>
    backend_transports;
std::unordered_map<const KVCacheStore*, const MooncakeStoreBackend*>
    store_backends;
const MooncakeStoreBackend* initialized_backend = nullptr;

MemoryStoreTransport& transport_for(const KVCacheStore& store) {
  return *backend_transports.at(store_backends.at(&store));
}

void initialize_store(KVCacheStore* store,
                      const KVCacheStoreInitConfig& config,
                      HostCacheStoreIndex index) {
  pending_transport =
      std::make_shared<MemoryStoreTransport>(std::make_shared<StoreObjects>());
  CHECK(store->init(config, std::move(index)));
  store_backends[store] = initialized_backend;
  pending_transport.reset();
}

void use_transport(KVCacheStore* store,
                   std::unique_ptr<MemoryStoreTransport> transport) {
  transport->config_ = transport_for(*store).config_;
  transport->ranges_ = transport_for(*store).ranges_;
  backend_transports.at(store_backends.at(store)) = std::move(transport);
}

std::vector<std::string> observe_keys(KVCacheStore& store,
                                      std::vector<BlockTransferInfo> infos) {
  Slice<BlockTransferInfo> slice(infos);
  store.batch_exist(slice);
  return transport_for(store).last_query_keys_;
}

}  // namespace

MooncakeStoreBackend::~MooncakeStoreBackend() {
  backend_transports.erase(this);
}
bool MooncakeStoreBackend::init(
    const MooncakeStoreBackendConfig& config,
    const std::vector<StoreRegisteredRange>& ranges) {
  CHECK(pending_transport != nullptr);
  pending_transport->config_ = config;
  pending_transport->ranges_ = ranges;
  backend_transports[this] = pending_transport;
  initialized_backend = this;
  return true;
}
std::vector<uint8_t> MooncakeStoreBackend::batch_put(
    const std::vector<std::string>& keys,
    const std::vector<MultiBuffer>& buffers) {
  return backend_transports.at(this)->batch_put(keys, buffers);
}
std::vector<uint8_t> MooncakeStoreBackend::batch_get(
    const std::vector<std::string>& keys,
    const std::vector<MultiBuffer>& buffers) {
  return backend_transports.at(this)->batch_get(keys, buffers);
}
std::vector<MooncakeReplicaTier> MooncakeStoreBackend::batch_query_tiers(
    const std::vector<std::string>& keys) {
  return backend_transports.at(this)->batch_query_tiers(keys);
}

namespace {

KVCache make_attention_cache(int64_t host_blocks, int64_t width) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return KVCache(
      KVCacheTensors{torch::zeros({host_blocks, 2, width}, options),
                     torch::zeros({host_blocks, 2, width}, options)});
}

KVCache make_linear_cache(int64_t host_blocks, int64_t width) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return KVCache(LinearAttentionKVCacheTensors{
      torch::zeros({host_blocks, width}, options),
      torch::zeros({host_blocks, width}, options)});
}

KVCache make_compressed_kpool_cache(int64_t host_blocks,
                                    int64_t index_width,
                                    int64_t tail_blocks) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors.key_cache =
      torch::zeros({host_blocks, 2, 8}, options);
  tensors.index_cache = torch::zeros({host_blocks, 2, 1, index_width}, options);
  tensors.kpool_tail = torch::zeros({tail_blocks, 2, 4, index_width}, options);
  return KVCache(tensors);
}

BlockTransferInfo make_block_info(uint8_t hash_value,
                                  BlockType block_type = BlockType::KV,
                                  int32_t destination_block_id = 0) {
  std::array<uint8_t, XXH3_128BITS_HASH_VALUE_LEN> hash_key;
  hash_key.fill(hash_value);
  return BlockTransferInfo(/*src_id=*/0,
                           destination_block_id,
                           hash_key.data(),
                           TransferType::G2H,
                           block_type);
}

KVCacheStoreInitConfig make_store_config(
    const std::string& model_id = "target-model",
    uint32_t tp_rank = 1,
    uint32_t tp_size = 2,
    bool enable_mla = false,
    int32_t cp_index_write_mode = kCpIndexWriteModeReplicated) {
  KVCacheStoreInitConfig config;
  config.model_id = model_id;
  config.tp_rank = tp_rank;
  config.tp_size = tp_size;
  config.kv_split_full_domain_size = static_cast<int32_t>(tp_size);
  config.enable_mla = enable_mla;
  config.cp_index_write_mode = cp_index_write_mode;
  return config;
}

std::string key_for_component(const std::vector<std::string>& keys,
                              const std::string& component) {
  const auto key =
      std::find_if(keys.begin(), keys.end(), [&component](const auto& entry) {
        return entry.find(":" + component + ":") != std::string::npos;
      });
  return key == keys.end() ? "" : *key;
}

TEST(KVCacheStoreTest, SplitPutElectsOneWriterPerStoreKey) {
  struct SplitWriterCase {
    int32_t cp_size;
    int32_t cp_rank;
    uint32_t tp_rank;
    uint32_t tp_size;
    int32_t full_domain_size;
    int32_t split_size;
    int32_t split_rank;
    bool sharded_owner;
    bool replicated_owner;
  };
  const std::vector<SplitWriterCase> writers = {
      // kv_split_size < cp_size: elect one CP replica per split and TP shard.
      {4, 0, 0, 2, 8, 2, 0, true, true},
      {4, 1, 0, 2, 8, 2, 0, false, false},
      {4, 2, 0, 2, 8, 2, 1, true, true},
      {4, 3, 0, 2, 8, 2, 1, false, false},
      {4, 0, 1, 2, 8, 2, 0, true, false},
      {4, 2, 1, 2, 8, 2, 1, true, false},
      // kv_split_size == cp_size: CP ranks are distinct, MLA TP ranks
      // replicate.
      {2, 1, 0, 2, 4, 2, 1, true, true},
      {2, 1, 1, 2, 4, 2, 1, true, false},
      // Full-domain DCP: split rank already distinguishes every CP/TP worker.
      {2, 0, 0, 2, 4, 4, 0, true, true},
      {2, 0, 1, 2, 4, 4, 1, true, true},
      {2, 1, 0, 2, 4, 4, 2, true, true},
      {2, 1, 1, 2, 4, 4, 3, true, true},
      // Legacy Store TP coordinates spanning CP still select the first TP
      // group per split, rather than publishing every unique sharded key.
      {4, 0, 0, 8, 8, 2, 0, true, true},
      {4, 0, 1, 8, 8, 2, 0, true, false},
      {4, 1, 2, 8, 8, 2, 0, false, false},
      {4, 2, 4, 8, 8, 2, 1, true, true},
      {4, 3, 6, 8, 8, 2, 1, false, false},
      // Full-domain remains unique when it equals CP and Store TP spans CP.
      {4, 3, 3, 4, 4, 4, 3, true, true},
  };
  for (bool enable_mla : {false, true}) {
    for (BlockType block_type : {BlockType::KV, BlockType::LINEAR}) {
      KVCache cache = block_type == BlockType::KV
                          ? make_attention_cache(/*host_blocks=*/2, /*width=*/8)
                          : make_linear_cache(/*host_blocks=*/2, /*width=*/8);
      for (const SplitWriterCase& writer : writers) {
        SCOPED_TRACE(
            ::testing::Message()
            << "enable_mla=" << enable_mla
            << " block_type=" << static_cast<int32_t>(block_type)
            << " cp_size=" << writer.cp_size << " cp_rank=" << writer.cp_rank
            << " tp_rank=" << writer.tp_rank << " tp_size=" << writer.tp_size
            << " full_domain_size=" << writer.full_domain_size << " split_size="
            << writer.split_size << " split_rank=" << writer.split_rank);
        KVCacheStoreInitConfig config = make_store_config(
            "split-writer-test", writer.tp_rank, writer.tp_size, enable_mla);
        config.cp_size = writer.cp_size;
        config.cp_rank = writer.cp_rank;
        config.kv_split_full_domain_size = writer.full_domain_size;
        config.kv_split_size = writer.split_size;
        config.kv_split_rank = writer.split_rank;
        HostCacheStoreIndex index;
        index[block_type].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                           "main",
                                                           &cache,
                                                           std::nullopt,
                                                           /*is_target=*/true});
        KVCacheStore store;
        initialize_store(&store, config, std::move(index));

        const std::vector<BlockTransferInfo> infos = {
            make_block_info(94, block_type)};
        const bool tp_replicated = enable_mla && block_type == BlockType::KV;
        const bool linear_owner =
            writer.cp_rank == 0 &&
            (enable_mla || writer.split_size <= writer.cp_size ||
             writer.tp_rank == 0);
        const bool is_owner = block_type == BlockType::LINEAR
                                  ? linear_owner
                                  : (tp_replicated ? writer.replicated_owner
                                                   : writer.sharded_owner);
        EXPECT_EQ(store.batch_put(infos), 1U);
        EXPECT_EQ(transport_for(store).last_put_keys_.empty(), !is_owner);
      }
    }
  }
}

TEST(KVCacheStoreTest, PublishesTargetDraftAndLinearFromTheirOwnRankGroups) {
  struct OwnerCase {
    int32_t cp_size;
    uint32_t store_tp_size;
    int32_t domain_size;
    int32_t split_size;
    bool enable_mla;
    std::vector<int32_t> target_kv_owners;
    std::vector<int32_t> draft_linear_owners;
  };
  const std::vector<OwnerCase> cases = {
      {4, 2, 8, 2, true, {0, 4}, {0, 1}},
      {4, 2, 8, 2, false, {0, 1, 4, 5}, {0, 1}},
      {1, 2, 2, 2, true, {0, 1}, {0, 1}},
      // Non-MLA decode: two splits, each containing a complete TP2 group.
      {1, 4, 4, 2, false, {0, 1, 2, 3}, {0, 1}},
      {1, 4, 4, 1, true, {0}, {0, 1, 2, 3}},
      {1, 4, 4, 1, false, {0, 1, 2, 3}, {0, 1, 2, 3}},
  };
  for (const OwnerCase& owner_case : cases) {
    for (int32_t rank = 0; rank < owner_case.domain_size; ++rank) {
      SCOPED_TRACE(::testing::Message()
                   << "cp=" << owner_case.cp_size
                   << " split=" << owner_case.split_size
                   << " mla=" << owner_case.enable_mla << " rank=" << rank);
      KVCache target_kv = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
      KVCache draft_kv = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
      KVCache target_linear = make_linear_cache(/*host_blocks=*/2, /*width=*/8);
      KVCache draft_linear = make_linear_cache(/*host_blocks=*/2, /*width=*/4);
      HostCacheStoreIndex index;
      index[BlockType::KV] = {
          {0, "main", &target_kv, owner_case.enable_mla, true},
          {1, "draft", &draft_kv, false, false}};
      index[BlockType::LINEAR] = {
          {0, "main", &target_linear, owner_case.enable_mla, true},
          {1, "draft", &draft_linear, false, false}};
      const int32_t workers_per_cp =
          owner_case.domain_size / owner_case.cp_size;
      KVCacheStoreInitConfig config =
          make_store_config("owner-test",
                            rank % owner_case.store_tp_size,
                            owner_case.store_tp_size,
                            owner_case.enable_mla);
      config.cp_size = owner_case.cp_size;
      config.cp_rank = rank / workers_per_cp;
      config.kv_split_full_domain_size = owner_case.domain_size;
      config.kv_split_size = owner_case.split_size;
      config.kv_split_rank =
          rank / (owner_case.domain_size / owner_case.split_size);
      KVCacheStore store;
      initialize_store(&store, config, std::move(index));
      auto objects = std::make_shared<StoreObjects>();
      use_transport(&store, std::make_unique<MemoryStoreTransport>(objects));
      const std::vector<BlockTransferInfo> infos = {
          make_block_info(96, BlockType::KV),
          make_block_info(97, BlockType::LINEAR)};
      ASSERT_EQ(store.batch_put(infos), 2U);
      const bool target_owner =
          std::find(owner_case.target_kv_owners.begin(),
                    owner_case.target_kv_owners.end(),
                    rank) != owner_case.target_kv_owners.end();
      const bool draft_linear_owner =
          std::find(owner_case.draft_linear_owners.begin(),
                    owner_case.draft_linear_owners.end(),
                    rank) != owner_case.draft_linear_owners.end();
      for (const BlockTransferInfo& info : infos) {
        const auto keys = observe_keys(store, {info});
        EXPECT_EQ(objects->count(key_for_component(keys, "main")),
                  info.block_type == BlockType::KV ? target_owner
                                                   : draft_linear_owner);
        EXPECT_EQ(objects->count(key_for_component(keys, "draft")),
                  draft_linear_owner);
      }
    }
  }
}

TEST(KVCacheStoreTest, UnsplitPutSkipsOnlySameKeyReplicas) {
  for (bool enable_mla : {false, true}) {
    for (BlockType block_type : {BlockType::KV, BlockType::LINEAR}) {
      KVCache cache = block_type == BlockType::KV
                          ? make_attention_cache(/*host_blocks=*/2, /*width=*/8)
                          : make_linear_cache(/*host_blocks=*/2, /*width=*/8);
      for (int32_t cp_rank : {0, 1}) {
        for (uint32_t tp_rank : {0U, 1U}) {
          SCOPED_TRACE(::testing::Message()
                       << "enable_mla=" << enable_mla
                       << " block_type=" << static_cast<int32_t>(block_type)
                       << " cp_rank=" << cp_rank << " tp_rank=" << tp_rank);
          KVCacheStoreInitConfig config = make_store_config(
              "unsplit-writer-test", tp_rank, /*tp_size=*/2, enable_mla);
          config.cp_size = 2;
          config.cp_rank = cp_rank;
          config.kv_split_full_domain_size = 4;
          HostCacheStoreIndex index;
          index[block_type].emplace_back(
              HostCacheStoreEntry{/*cache_handle=*/0,
                                  "main",
                                  &cache,
                                  std::nullopt,
                                  /*is_target=*/true});
          KVCacheStore store;
          initialize_store(&store, config, std::move(index));

          const bool same_key_replica =
              cp_rank != 0 ||
              (enable_mla && block_type == BlockType::KV && tp_rank != 0);
          EXPECT_EQ(store.batch_put({make_block_info(95, block_type)}), 1U);
          EXPECT_EQ(transport_for(store).last_put_keys_.empty(),
                    same_key_replica);
        }
      }
    }
  }
}

TEST(KVCacheStoreDeathTest, RejectsTransportResultCountMismatch) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  initialize_store(&store, make_store_config(), std::move(index));
  auto objects = std::make_shared<StoreObjects>();
  auto transport = std::make_unique<MemoryStoreTransport>(objects);
  MemoryStoreTransport* transport_ptr = transport.get();
  use_transport(&store, std::move(transport));
  std::vector<BlockTransferInfo> infos = {make_block_info(102)};
  Slice<BlockTransferInfo> slice(infos);
  for (size_t result_count : {0U, 2U}) {
    transport_ptr->result_count_ = result_count;
    EXPECT_DEATH(store.batch_put(slice), "results.size\\(\\)");
    EXPECT_DEATH(store.batch_get_with_status(slice), "results.size\\(\\)");
  }
}

TEST(KVCacheStoreDeathTest, RejectsInvalidSplitTopologyBeforeStoreSetup) {
  for (const auto& [size, rank] : std::vector<std::pair<int32_t, int32_t>>{
           {0, 0}, {-1, 0}, {4, -1}, {4, 4}}) {
    KVCacheStoreInitConfig config = make_store_config();
    config.kv_split_size = size;
    config.kv_split_rank = rank;
    EXPECT_DEATH(
        {
          KVCacheStore store;
          store.init(config, {});
        },
        "kv_split");
  }
  for (const auto& [size, rank] : std::vector<std::pair<int32_t, int32_t>>{
           {0, 0}, {-1, 0}, {2, -1}, {2, 2}}) {
    KVCacheStoreInitConfig config = make_store_config();
    config.cp_size = size;
    config.cp_rank = rank;
    EXPECT_DEATH(
        {
          KVCacheStore store;
          store.init(config, {});
        },
        "cp_");
  }
  KVCacheStoreInitConfig nonfactor = make_store_config();
  nonfactor.cp_size = 4;
  nonfactor.kv_split_full_domain_size = 8;
  nonfactor.kv_split_size = 3;
  EXPECT_DEATH(
      {
        KVCacheStore store;
        store.init(nonfactor, {});
      },
      "kv_split_size must divide cp_size");

  KVCacheStoreInitConfig overlapping_tp = make_store_config(
      "target-model", /*tp_rank=*/7, /*tp_size=*/8, /*enable_mla=*/true);
  overlapping_tp.cp_size = 4;
  overlapping_tp.cp_rank = 3;
  overlapping_tp.kv_split_full_domain_size = 8;
  overlapping_tp.kv_split_size = 8;
  overlapping_tp.kv_split_rank = 7;
  EXPECT_DEATH(
      {
        KVCacheStore store;
        store.init(overlapping_tp, {});
      },
      "requires Host caches");

  KVCacheStoreInitConfig invalid_tp_domain = make_store_config(
      "target-model", /*tp_rank=*/0, /*tp_size=*/4, /*enable_mla=*/true);
  invalid_tp_domain.cp_size = 4;
  invalid_tp_domain.kv_split_full_domain_size = 8;
  EXPECT_DEATH(
      {
        KVCacheStore store;
        store.init(invalid_tp_domain, {});
      },
      "Store TP ranks");
}

TEST(KVCacheStoreTest, TargetKeyDoesNotDependOnDraftRegistration) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);

  KVCacheStore target_only_store;
  HostCacheStoreIndex target_only_index;
  target_only_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &target_cache,
                          std::nullopt,
                          /*is_target=*/true});
  initialize_store(
      &target_only_store, make_store_config(), std::move(target_only_index));

  KVCacheStore speculative_store;
  HostCacheStoreIndex speculative_index;
  speculative_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &target_cache,
                          std::nullopt,
                          /*is_target=*/true});
  speculative_index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          std::nullopt,
                          /*is_target=*/false});
  initialize_store(
      &speculative_store, make_store_config(), std::move(speculative_index));

  const std::vector<BlockTransferInfo> block_info = {make_block_info(3)};
  const auto target_only_keys = observe_keys(target_only_store, block_info);
  const auto speculative_keys = observe_keys(speculative_store, block_info);

  ASSERT_EQ(target_only_keys.size(), 1U);
  ASSERT_EQ(speculative_keys.size(), 2U);
  EXPECT_EQ(target_only_keys.front(),
            key_for_component(speculative_keys, "main"));
  EXPECT_NE(
      key_for_component(speculative_keys, "main"),
      key_for_component(speculative_keys, "spec_draft::mtp::draft-model"));
}

TEST(KVCacheStoreTest, UsesPerCacheMlaLayoutForTargetAndDraft) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        /*enable_mla=*/true,
                                                        /*is_target=*/true});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::dflash2::draft-model",
                          &draft_cache,
                          /*enable_mla=*/false,
                          /*is_target=*/false});
  initialize_store(&store,
                   make_store_config("target-model",
                                     /*tp_rank=*/1,
                                     /*tp_size=*/8,
                                     /*enable_mla=*/true),
                   std::move(index));

  const auto keys = observe_keys(store, {make_block_info(7, BlockType::KV)});
  const std::string target_key = key_for_component(keys, "main");
  const std::string draft_key =
      key_for_component(keys, "spec_draft::dflash2::draft-model");
  EXPECT_NE(target_key.find("mla:"), std::string::npos);
  EXPECT_NE(draft_key.find("8:1:"), std::string::npos);
  EXPECT_EQ(draft_key.find("mla:"), std::string::npos);
}

TEST(KVCacheStoreTest, DraftKeyDependsOnTargetDraftAndAlgorithm) {
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(5)};

  const auto build_draft_key = [&draft_cache, &block_info](
                                   const std::string& target_model_id,
                                   const std::string& component) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/1,
                                                          component,
                                                          &draft_cache,
                                                          std::nullopt,
                                                          /*is_target=*/false});
    initialize_store(
        &store, make_store_config(target_model_id), std::move(index));
    return observe_keys(store, block_info).front();
  };

  const std::string baseline =
      build_draft_key("target-a", "spec_draft::mtp::draft-a");
  EXPECT_NE(baseline, build_draft_key("target-b", "spec_draft::mtp::draft-a"));
  EXPECT_NE(baseline, build_draft_key("target-a", "spec_draft::mtp::draft-b"));
  EXPECT_NE(baseline,
            build_draft_key("target-a", "spec_draft::dspark::draft-a"));
  EXPECT_EQ(baseline, build_draft_key("target-a", "spec_draft::mtp::draft-a"));
}

// The CP index write mode changes which physical pages of an INDEX resource
// hold valid data while leaving every tensor shape identical, so the object
// keys must separate the modes: a replicated reader scoring a sharded
// writer's stale peer pages as valid is exactly the silent cross-mode hit
// the schema field exists to prevent.
TEST(KVCacheStoreTest, SchemaSeparatesCpIndexWriteModes) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(9)};

  const auto build_key = [&cache, &block_info](int32_t cp_index_write_mode) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                          "main",
                                                          &cache,
                                                          std::nullopt,
                                                          /*is_target=*/true});
    initialize_store(&store,
                     make_store_config(/*model_id=*/"target-model",
                                       /*tp_rank=*/1,
                                       /*tp_size=*/2,
                                       /*enable_mla=*/false,
                                       cp_index_write_mode),
                     std::move(index));
    return observe_keys(store, block_info).front();
  };

  const std::string replicated = build_key(kCpIndexWriteModeReplicated);
  EXPECT_NE(replicated, build_key(kCpIndexWriteModeSharded));
  EXPECT_EQ(replicated, build_key(kCpIndexWriteModeReplicated));
}

TEST(KVCacheStoreTest, SchemaExcludesHostCapacity) {
  KVCache small_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache large_cache = make_attention_cache(/*host_blocks=*/5, /*width=*/8);
  KVCache different_cache =
      make_attention_cache(/*host_blocks=*/2, /*width=*/16);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(7)};

  const auto build_target_key = [&block_info](KVCache* cache) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{
        /*cache_handle=*/0, "main", cache, std::nullopt, /*is_target=*/true});
    initialize_store(&store, make_store_config(), std::move(index));
    return observe_keys(store, block_info).front();
  };

  EXPECT_EQ(build_target_key(&small_cache), build_target_key(&large_cache));
  EXPECT_NE(build_target_key(&small_cache), build_target_key(&different_cache));
}

TEST(KVCacheStoreTest, CompressedKPoolRegistersIndexAndTailLayouts) {
  KVCache cache = make_compressed_kpool_cache(
      /*host_blocks=*/2, /*index_width=*/8, /*tail_blocks=*/3);
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  index[BlockType::LINEAR].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &cache,
                          std::nullopt,
                          /*is_target=*/true});
  KVCacheStore store;
  initialize_store(&store,
                   make_store_config("compressed-kpool",
                                     /*tp_rank=*/0,
                                     /*tp_size=*/2,
                                     /*enable_mla=*/true),
                   std::move(index));

  const std::vector<BlockTransferInfo> kv_info = {
      make_block_info(81, BlockType::KV)};
  const std::vector<BlockTransferInfo> tail_info = {
      make_block_info(81, BlockType::LINEAR)};
  EXPECT_EQ(store.batch_put(kv_info), 1U);
  EXPECT_EQ(store.batch_put(tail_info), 1U);
  EXPECT_NE(observe_keys(store, kv_info).front(),
            observe_keys(store, tail_info).front());
  for (BlockType type : {BlockType::KV, BlockType::LINEAR}) {
    for (const auto& [role, tensor] : cache.get_block_type_tensors(type)) {
      tensor[0].fill_(1 + static_cast<int32_t>(role));
    }
    const std::vector<BlockTransferInfo> source = {make_block_info(82, type)};
    ASSERT_EQ(store.batch_put(source), 1U);
    ASSERT_EQ(store.batch_get({make_block_info(82, type, 1)}), 1U);
    for (const auto& [role, tensor] : cache.get_block_type_tensors(type)) {
      EXPECT_TRUE(torch::equal(tensor[0], tensor[1]));
    }
  }

  KVCache different_index = make_compressed_kpool_cache(
      /*host_blocks=*/2, /*index_width=*/16, /*tail_blocks=*/3);
  HostCacheStoreIndex different_index_layout;
  different_index_layout[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &different_index,
                          std::nullopt,
                          /*is_target=*/true});
  KVCacheStore different_store;
  initialize_store(&different_store,
                   make_store_config("compressed-kpool",
                                     /*tp_rank=*/0,
                                     /*tp_size=*/2,
                                     /*enable_mla=*/true),
                   std::move(different_index_layout));
  EXPECT_NE(observe_keys(store, kv_info).front(),
            observe_keys(different_store, kv_info).front());
}

TEST(KVCacheStoreTest, MlaKeyExcludesTensorParallelTopology) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  const std::vector<BlockTransferInfo> block_info = {make_block_info(8)};

  const auto build_key = [&cache, &block_info](uint32_t tp_rank,
                                               uint32_t tp_size,
                                               bool enable_mla,
                                               int32_t split_size = 1,
                                               int32_t split_rank = 0) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                          "main",
                                                          &cache,
                                                          std::nullopt,
                                                          /*is_target=*/true});
    KVCacheStoreInitConfig config =
        make_store_config("target-model", tp_rank, tp_size, enable_mla);
    config.cp_size = split_size;
    config.cp_rank = split_rank;
    config.kv_split_full_domain_size =
        split_size * static_cast<int32_t>(tp_size);
    config.kv_split_size = split_size;
    config.kv_split_rank = split_rank;
    initialize_store(&store, config, std::move(index));
    return observe_keys(store, block_info).front();
  };

  const std::string mla_key = build_key(/*tp_rank=*/0,
                                        /*tp_size=*/1,
                                        /*enable_mla=*/true);
  for (bool enable_mla : {false, true}) {
    const std::string split_key = build_key(/*tp_rank=*/0,
                                            /*tp_size=*/8,
                                            enable_mla,
                                            /*split_size=*/4,
                                            /*split_rank=*/0);
    EXPECT_NE(split_key,
              build_key(/*tp_rank=*/0,
                        /*tp_size=*/8,
                        enable_mla,
                        /*split_size=*/4,
                        /*split_rank=*/1));
    EXPECT_NE(split_key,
              build_key(/*tp_rank=*/0,
                        /*tp_size=*/8,
                        enable_mla,
                        /*split_size=*/8,
                        /*split_rank=*/0));
  }
  EXPECT_EQ(build_key(/*tp_rank=*/0,
                      /*tp_size=*/8,
                      /*enable_mla=*/true,
                      /*split_size=*/4,
                      /*split_rank=*/2),
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/true,
                      /*split_size=*/4,
                      /*split_rank=*/2));
  EXPECT_EQ(mla_key,
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/true));
  EXPECT_NE(mla_key,
            build_key(/*tp_rank=*/0,
                      /*tp_size=*/1,
                      /*enable_mla=*/false));
  EXPECT_NE(build_key(/*tp_rank=*/0,
                      /*tp_size=*/8,
                      /*enable_mla=*/false),
            build_key(/*tp_rank=*/7,
                      /*tp_size=*/8,
                      /*enable_mla=*/false));
}

TEST(KVCacheStoreTest, NonSplitCachesUseCanonicalSplitZeroKeys) {
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCache linear_cache = make_linear_cache(/*host_blocks=*/2, /*width=*/4);
  const std::vector<BlockTransferInfo> draft_info = {
      make_block_info(11, BlockType::KV)};
  const std::vector<BlockTransferInfo> linear_info = {
      make_block_info(11, BlockType::LINEAR)};

  const auto build_key = [&](BlockType block_type,
                             bool is_target,
                             uint32_t tp_rank,
                             int32_t split_rank) {
    KVCacheStore store;
    HostCacheStoreIndex index;
    KVCache* cache = block_type == BlockType::KV ? &draft_cache : &linear_cache;
    index[block_type].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/0,
                            block_type == BlockType::KV ? "draft" : "main",
                            cache,
                            /*enable_mla=*/false,
                            is_target});
    KVCacheStoreInitConfig config = make_store_config("non-split",
                                                      tp_rank,
                                                      /*tp_size=*/2,
                                                      /*enable_mla=*/false);
    config.cp_size = 2;
    config.cp_rank = split_rank;
    config.kv_split_full_domain_size = 4;
    config.kv_split_size = 2;
    config.kv_split_rank = split_rank;
    initialize_store(&store, config, std::move(index));
    return observe_keys(store,
                        block_type == BlockType::KV ? draft_info : linear_info)
        .front();
  };

  EXPECT_EQ(build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/0, 0),
            build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/0, 1));
  EXPECT_NE(build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/0, 0),
            build_key(BlockType::KV, /*is_target=*/false, /*tp_rank=*/1, 0));
  EXPECT_EQ(build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/0, 0),
            build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/0, 1));
  EXPECT_NE(build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/0, 0),
            build_key(BlockType::LINEAR, /*is_target=*/true, /*tp_rank=*/1, 0));
}

TEST(KVCacheStoreTest, SelectsOnlyExplicitRdmaDevices) {
  KVCacheStoreInitConfig config = make_store_config();
  config.protocol = "rdma";
  config.rdma_devices = "mlx5_0,mlx5_1";
  KVCache cache = make_attention_cache(2, 8);
  KVCacheStore explicit_store;
  initialize_store(
      &explicit_store,
      config,
      {{BlockType::KV, {{0, "main", &cache, std::nullopt, true}}}});
  EXPECT_EQ(transport_for(explicit_store).config_.rdma_devices,
            "mlx5_0,mlx5_1");

  config.rdma_devices.clear();
  xllm::tests::ScopedEnvironmentVariable device_names("DEVICE_NAMES");
  ASSERT_TRUE(device_names.set("legacy_hca"));
  KVCacheStore automatic_store;
  initialize_store(
      &automatic_store,
      config,
      {{BlockType::KV, {{0, "main", &cache, std::nullopt, true}}}});
  EXPECT_TRUE(transport_for(automatic_store).config_.rdma_devices.empty());
}

TEST(KVCacheStoreTest, TcpIgnoresRdmaDevices) {
  KVCacheStoreInitConfig config = make_store_config();
  config.protocol = "tcp";
  config.rdma_devices = "mlx5_0";

  KVCache cache = make_attention_cache(2, 8);
  KVCacheStore store;
  initialize_store(
      &store,
      config,
      {{BlockType::KV, {{0, "main", &cache, std::nullopt, true}}}});
  EXPECT_TRUE(transport_for(store).config_.rdma_devices.empty());
}

TEST(KVCacheStoreDeathTest, RejectsDuplicateKeyComponentsPerBlockType) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);

  EXPECT_DEATH(
      {
        KVCacheStore store;
        HostCacheStoreIndex index;
        index[BlockType::KV].emplace_back(
            HostCacheStoreEntry{/*cache_handle=*/0,
                                "main",
                                &target_cache,
                                std::nullopt,
                                /*is_target=*/true});
        index[BlockType::KV].emplace_back(
            HostCacheStoreEntry{/*cache_handle=*/1,
                                "main",
                                &draft_cache,
                                std::nullopt,
                                /*is_target=*/false});
        initialize_store(&store, make_store_config(), std::move(index));
      },
      "Duplicate KVCacheStore key component");
}

TEST(KVCacheStoreTest, ObjectPresenceRequiresEveryComponentObject) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          std::nullopt,
                          /*is_target=*/false});
  initialize_store(&store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> block_info = {
      make_block_info(11, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(12, BlockType::KV, /*destination_block_id=*/1),
      make_block_info(13, BlockType::KV, /*destination_block_id=*/0)};
  const auto keys = observe_keys(store, block_info);
  ASSERT_EQ(keys.size(), 6U);
  const std::vector<MooncakeReplicaTier> tiers = {MooncakeReplicaTier::MEMORY,
                                                  MooncakeReplicaTier::DISK,
                                                  MooncakeReplicaTier::MEMORY,
                                                  MooncakeReplicaTier::MISSING,
                                                  MooncakeReplicaTier::MISSING,
                                                  MooncakeReplicaTier::MISSING};
  for (size_t index = 0; index < keys.size(); ++index) {
    transport_for(store).tiers_[keys[index]] = tiers[index];
  }
  std::vector<BlockTransferInfo> infos = block_info;
  Slice<BlockTransferInfo> slice(infos);
  EXPECT_EQ(store.batch_exist(slice), std::vector<uint8_t>({1, 0, 0}));
}

TEST(KVCacheStoreTest, RecordsGetVolumePerReplicaTier) {
  KVCache cache = make_attention_cache(3, 8);
  KVCacheStore store;
  initialize_store(
      &store,
      make_store_config(),
      {{BlockType::KV, {{0, "main", &cache, std::nullopt, true}}}});
  std::vector<BlockTransferInfo> infos = {
      make_block_info(20),
      make_block_info(21, BlockType::KV, 1),
      make_block_info(22, BlockType::KV, 2)};
  ASSERT_EQ(store.batch_put(infos), 3U);
  const auto keys = observe_keys(store, infos);
  ASSERT_EQ(keys.size(), 3U);
  transport_for(store).tiers_[keys[1]] = MooncakeReplicaTier::DISK;
  transport_for(store).tiers_[keys[2]] = MooncakeReplicaTier::MISSING;
  StoreGetStats stats;
  Slice<BlockTransferInfo> slice(infos);
  ASSERT_EQ(store.batch_get_with_status(slice, &stats),
            std::vector<uint8_t>({1, 1, 1}));
  const uint64_t object_bytes =
      cache.get_k_cache()[0].nbytes() + cache.get_v_cache()[0].nbytes();
  EXPECT_EQ(stats.read_bytes, 3 * object_bytes);
  EXPECT_EQ(stats.memory_objects, 1U);
  EXPECT_EQ(stats.memory_bytes, object_bytes);
  EXPECT_EQ(stats.disk_objects, 1U);
  EXPECT_EQ(stats.disk_bytes, object_bytes);
}

TEST(KVCacheStoreTest, RestoresBatchBlocksWithIndependentMisses) {
  KVCache cache = make_attention_cache(/*host_blocks=*/6, /*width=*/8);
  cache.get_k_cache()[0].fill_(7);
  cache.get_v_cache()[0].fill_(70);
  cache.get_k_cache()[1].fill_(9);
  cache.get_v_cache()[1].fill_(90);

  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  initialize_store(&store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> sources = {
      make_block_info(/*hash_value=*/94),
      make_block_info(
          /*hash_value=*/95, BlockType::KV, /*destination_block_id=*/1)};
  ASSERT_EQ(store.batch_put(sources), 2U);

  std::vector<BlockTransferInfo> destinations = {
      make_block_info(
          /*hash_value=*/97, BlockType::KV, /*destination_block_id=*/2),
      make_block_info(
          /*hash_value=*/94, BlockType::KV, /*destination_block_id=*/3),
      make_block_info(
          /*hash_value=*/96, BlockType::KV, /*destination_block_id=*/4),
      make_block_info(
          /*hash_value=*/95, BlockType::KV, /*destination_block_id=*/5)};
  Slice<BlockTransferInfo> destination_slice(destinations);
  EXPECT_EQ(store.batch_get_with_status(destination_slice),
            std::vector<uint8_t>({0, 1, 0, 1}));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[3], cache.get_k_cache()[0]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[3], cache.get_v_cache()[0]));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[5], cache.get_k_cache()[1]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[5], cache.get_v_cache()[1]));
  EXPECT_EQ(cache.get_k_cache()[2].count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.get_v_cache()[2].count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.get_k_cache()[4].count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.get_v_cache()[4].count_nonzero().item<int64_t>(), 0);
}

TEST(KVCacheStoreTest, DeduplicatesPhysicalKeysAndRespectsBlockTypeEntries) {
  KVCache target_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCache draft_cache = make_attention_cache(/*host_blocks=*/2, /*width=*/4);
  KVCache linear_cache = make_linear_cache(/*host_blocks=*/2, /*width=*/6);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &target_cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/1,
                          "spec_draft::mtp::draft-model",
                          &draft_cache,
                          std::nullopt,
                          /*is_target=*/false});
  index[BlockType::LINEAR].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          &linear_cache,
                          std::nullopt,
                          /*is_target=*/true});
  initialize_store(&store, make_store_config(), std::move(index));

  const std::vector<BlockTransferInfo> duplicate_kv = {
      make_block_info(11, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(11, BlockType::KV, /*destination_block_id=*/1)};
  ASSERT_EQ(store.batch_put(duplicate_kv), 2U);
  EXPECT_EQ(transport_for(store).last_put_keys_.size(), 2U);
  const std::vector<BlockTransferInfo> linear = {
      make_block_info(12, BlockType::LINEAR)};
  ASSERT_EQ(store.batch_put(linear), 1U);
  EXPECT_EQ(transport_for(store).last_put_keys_.size(), 1U);
}

TEST(KVCacheStoreTest, FailsDuplicateGetKeysWithoutStoreAccess) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  KVCacheStore store;
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/0,
                                                        "main",
                                                        &cache,
                                                        std::nullopt,
                                                        /*is_target=*/true});
  initialize_store(&store, make_store_config(), std::move(index));

  std::vector<BlockTransferInfo> block_info = {
      make_block_info(13, BlockType::KV, /*destination_block_id=*/0),
      make_block_info(13, BlockType::KV, /*destination_block_id=*/1)};
  Slice<BlockTransferInfo> slice(block_info);

  EXPECT_EQ(store.batch_get_with_status(slice), std::vector<uint8_t>({0, 0}));
  EXPECT_EQ(transport_for(store).batch_get_calls_, 0);
}

TEST(KVCacheStoreTest, RestoresIndexScaleWithoutAliasing) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  DeepSeekV4KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({3, 1, 128, 1, 16}, options);
  tensors.index_cache =
      torch::zeros({3, 1, 128, 1, 8}, options.dtype(torch::kInt8));
  tensors.indexer_cache_scale =
      torch::zeros({3, 1, 128, 1}, options.dtype(torch::kFloat16));
  tensors.compressed_block_type = BlockType::C1;
  KVCache c1(tensors);
  tensors.key_cache = torch::zeros_like(tensors.key_cache);
  tensors.index_cache = torch::zeros_like(tensors.index_cache);
  tensors.indexer_cache_scale = torch::zeros_like(tensors.indexer_cache_scale);
  tensors.compressed_block_type = BlockType::C2;
  KVCache c2(tensors);
  HostCacheStoreIndex index;
  index[BlockType::C1].emplace_back(HostCacheStoreEntry{
      /*cache_handle=*/0, "main", &c1, std::nullopt, /*is_target=*/true});
  index[BlockType::C2].emplace_back(HostCacheStoreEntry{
      /*cache_handle=*/0, "main", &c2, std::nullopt, /*is_target=*/true});
  KVCacheStore store;
  auto objects = std::make_shared<StoreObjects>();
  KVCacheStoreInitConfig config = make_store_config("deepseek-v41");
  initialize_store(&store, config, std::move(index));
  use_transport(&store, std::make_unique<MemoryStoreTransport>(objects));
  for (const auto& [type, cache] : std::map<BlockType, KVCache*>{
           {BlockType::C1, &c1}, {BlockType::C2, &c2}}) {
    const auto roles = cache->get_block_type_tensors(type);
    ASSERT_EQ(roles.size(), 3u);
    for (const auto& [role, tensor] : roles) {
      tensor[0].fill_(static_cast<int32_t>(type) + static_cast<int32_t>(role));
    }
    ASSERT_EQ(store.batch_put({make_block_info(44, type)}), 1u);
    std::vector<BlockTransferInfo> infos = {make_block_info(44, type, 2)};
    Slice<BlockTransferInfo> slice(infos);
    ASSERT_EQ(store.batch_get_with_status(slice), (std::vector<uint8_t>{1}));
    for (const auto& [role, tensor] : roles) {
      EXPECT_TRUE(torch::equal(tensor[0], tensor[2]))
          << static_cast<int32_t>(role);
    }
  }
  // Identical hashes and physical spans still name separate TOKEN objects.
  EXPECT_EQ(objects->size(), 2u);
}

TEST(KVCacheStoreTest, RestoresRoleBytesAndRequiresEveryRegisteredShard) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  KVCache target(IndexedKVCacheTensors{
      KVCacheTensors{torch::zeros({3, 2, 8}, options), torch::Tensor()},
      torch::zeros({3, 2, 4}, options.dtype(torch::kInt8)),
      torch::zeros({3, 2}, options)});
  KVCache draft = make_attention_cache(/*host_blocks=*/3, /*width=*/4);
  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{
      /*cache_handle=*/0, "main", &target, std::nullopt, /*is_target=*/true});
  index[BlockType::KV].emplace_back(HostCacheStoreEntry{/*cache_handle=*/1,
                                                        "draft",
                                                        &draft,
                                                        std::nullopt,
                                                        /*is_target=*/false});
  KVCacheStore store;
  initialize_store(&store, make_store_config(), std::move(index));
  auto objects = std::make_shared<StoreObjects>();
  auto transport = std::make_unique<MemoryStoreTransport>(objects);
  MemoryStoreTransport* transport_ptr = transport.get();
  use_transport(&store, std::move(transport));

  const std::vector<KVCache*> caches = {&target, &draft};
  for (size_t cache_index = 0; cache_index < caches.size(); ++cache_index) {
    for (const auto& [role, tensor] :
         caches[cache_index]->get_block_type_tensors(BlockType::KV)) {
      tensor[0].fill_(1 + cache_index * 10 + static_cast<size_t>(role));
    }
  }
  const std::vector<BlockTransferInfo> put_info = {
      make_block_info(/*hash_value=*/44)};
  EXPECT_EQ(store.batch_put(put_info), 1U);
  EXPECT_EQ(objects->size(), 2U);
  std::vector<BlockTransferInfo> get_info = {make_block_info(
      /*hash_value=*/44, BlockType::KV, /*destination_block_id=*/2)};
  Slice<BlockTransferInfo> get_slice(get_info);
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({1}));
  EXPECT_EQ(store.batch_exist(get_slice), std::vector<uint8_t>({1}));
  StoreGetStats stats;
  EXPECT_EQ(store.batch_get_with_status(get_slice, &stats),
            std::vector<uint8_t>({1}));
  EXPECT_EQ(stats.memory_objects, 2U);
  EXPECT_GT(stats.read_bytes, 0U);
  for (KVCache* cache : caches) {
    for (const auto& [role, tensor] :
         cache->get_block_type_tensors(BlockType::KV)) {
      EXPECT_TRUE(torch::equal(tensor[0], tensor[2]));
      tensor[2].zero_();
    }
  }

  const auto keys = observe_keys(store, put_info);
  const std::string draft_key = key_for_component(keys, "draft");
  transport_ptr->failed_gets_.emplace(draft_key);
  // Both objects exist, but a failed actual GET must remain a logical miss.
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({0}));
  transport_ptr->failed_gets_.clear();
  objects->erase(draft_key);
  target.get_k_cache()[2].zero_();
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({0}));
  // A partial backend GET may write one shard, but the logical block remains
  // a miss and cannot be published until all registered shards succeed.
  EXPECT_TRUE(torch::equal(target.get_k_cache()[0], target.get_k_cache()[2]));
  EXPECT_EQ(draft.get_k_cache()[2].count_nonzero().item<int64_t>(), 0);
  transport_ptr->failed_puts_.emplace(draft_key);
  EXPECT_EQ(store.batch_put(put_info), 0U);
  EXPECT_EQ(store.batch_exist(get_slice), std::vector<uint8_t>({0}));
  transport_ptr->failed_puts_.clear();
  EXPECT_EQ(store.batch_put(put_info), 1U);
  EXPECT_EQ(store.batch_exist(get_slice), std::vector<uint8_t>({1}));
  EXPECT_EQ(store.batch_get_with_status(get_slice), std::vector<uint8_t>({1}));
}

}  // namespace
}  // namespace xllm
