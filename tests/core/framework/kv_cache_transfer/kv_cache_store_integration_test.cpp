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
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/kv_cache_store.h"
#include "framework/kv_cache_transfer/mooncake_store_backend.h"

namespace xllm {

namespace {

KVCache make_attention_cache(int64_t host_blocks, int64_t width) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  return KVCache(
      KVCacheTensors{torch::zeros({host_blocks, 2, width}, options),
                     torch::zeros({host_blocks, 2, width}, options)});
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

class KVCacheStoreIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* master = std::getenv("XLLM_TEST_STORE_MASTER");
    ASSERT_NE(master, nullptr)
        << "Set XLLM_TEST_STORE_MASTER to a Mooncake master with an active "
           "TCP storage node when enabling the integration target.";
    config_ = make_store_config("split-store-test",
                                /*tp_rank=*/0,
                                /*tp_size=*/8,
                                /*enable_mla=*/true);
    config_.model_id +=
        ":" + std::to_string(getpid()) + ":" +
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    config_.master_server_address = master;
    config_.metadata_server = "P2PHANDSHAKE";
  }

  bool init_store(KVCacheStore& store,
                  KVCache& cache,
                  const KVCacheStoreInitConfig& config) {
    HostCacheStoreIndex index;
    index[BlockType::KV].emplace_back(
        HostCacheStoreEntry{/*cache_handle=*/0,
                            "main",
                            std::nullopt,
                            /*is_target=*/true,
                            {host_cache_store_group(cache, BlockType::KV)}});
    return store.init(config, std::move(index));
  }

  MooncakeStoreBackendConfig backend_config() const {
    MooncakeStoreBackendConfig config;
    config.localhost_name = config_.localhost_name;
    config.protocol = config_.protocol;
    config.metadata_server = config_.metadata_server;
    config.master_server_address = config_.master_server_address;
    return config;
  }

  KVCacheStoreInitConfig config_;
};

TEST_F(KVCacheStoreIntegrationTest,
       RestoresEachSplitLayoutWithoutKeyCollisions) {
  constexpr int32_t split_size = 2;
  const std::vector<BlockTransferInfo> infos = {make_block_info(91)};
  for (int32_t rank = 0; rank < split_size; ++rank) {
    KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
    cache.get_k_cache().fill_(split_size * 10 + rank);
    cache.get_v_cache().fill_(split_size * 10 + rank + 100);
    config_.cp_size = split_size;
    config_.cp_rank = rank;
    config_.kv_split_full_domain_size =
        split_size * static_cast<int32_t>(config_.tp_size);
    config_.kv_split_size = split_size;
    config_.kv_split_rank = rank;
    KVCacheStore store;
    ASSERT_TRUE(init_store(store, cache, config_));
    ASSERT_EQ(store.batch_put(infos), 1U);
  }
  for (int32_t rank = 0; rank < split_size; ++rank) {
    SCOPED_TRACE(::testing::Message() << "rank=" << rank);
    KVCache cache = make_attention_cache(/*host_blocks=*/3, /*width=*/8);
    config_.cp_size = split_size;
    config_.cp_rank = rank;
    config_.kv_split_full_domain_size =
        split_size * static_cast<int32_t>(config_.tp_size);
    config_.kv_split_size = split_size;
    config_.kv_split_rank = rank;
    KVCacheStore store;
    ASSERT_TRUE(init_store(store, cache, config_));
    ASSERT_EQ(store.batch_get(infos), 1U);
    EXPECT_TRUE(torch::equal(
        cache.get_k_cache()[0],
        torch::full_like(cache.get_k_cache()[0], split_size * 10 + rank)));
    EXPECT_TRUE(torch::equal(cache.get_v_cache()[0],
                             torch::full_like(cache.get_v_cache()[0],
                                              split_size * 10 + rank + 100)));
  }
}

TEST_F(KVCacheStoreIntegrationTest, RestoresBatchBlocksWithIndependentMisses) {
  KVCache cache = make_attention_cache(/*host_blocks=*/5, /*width=*/8);
  cache.get_k_cache()[0].fill_(7);
  cache.get_v_cache()[0].fill_(70);
  cache.get_k_cache()[1].fill_(9);
  cache.get_v_cache()[1].fill_(90);
  KVCacheStore store;
  ASSERT_TRUE(init_store(store, cache, config_));
  const std::vector<BlockTransferInfo> sources = {
      make_block_info(/*hash_value=*/94),
      make_block_info(
          /*hash_value=*/95, BlockType::KV, /*destination_block_id=*/1)};
  ASSERT_EQ(store.batch_put(sources), 2U);

  std::vector<BlockTransferInfo> destinations = {
      make_block_info(
          /*hash_value=*/97, BlockType::KV, /*destination_block_id=*/99),
      make_block_info(
          /*hash_value=*/94, BlockType::KV, /*destination_block_id=*/2),
      make_block_info(
          /*hash_value=*/96, BlockType::KV, /*destination_block_id=*/3),
      make_block_info(
          /*hash_value=*/95, BlockType::KV, /*destination_block_id=*/4)};
  Slice<BlockTransferInfo> destination_slice(destinations);
  EXPECT_EQ(store.batch_get_with_status(destination_slice),
            std::vector<uint8_t>({0, 1, 0, 1}));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[2], cache.get_k_cache()[0]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[2], cache.get_v_cache()[0]));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[4], cache.get_k_cache()[1]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[4], cache.get_v_cache()[1]));
  EXPECT_EQ(cache.get_k_cache()[3].count_nonzero().item<int64_t>(), 0);
  EXPECT_EQ(cache.get_v_cache()[3].count_nonzero().item<int64_t>(), 0);
}

TEST_F(KVCacheStoreIntegrationTest, MlaReplicaReadsTpZeroPublishedSplit) {
  KVCache cache = make_attention_cache(/*host_blocks=*/2, /*width=*/8);
  cache.get_k_cache().fill_(23);
  cache.get_v_cache().fill_(47);
  config_.cp_size = 4;
  config_.cp_rank = 2;
  config_.kv_split_full_domain_size = 4 * static_cast<int32_t>(config_.tp_size);
  config_.kv_split_size = 4;
  config_.kv_split_rank = 2;
  config_.tp_rank = 0;
  const std::vector<BlockTransferInfo> infos = {make_block_info(92)};
  {
    KVCacheStore writer;
    ASSERT_TRUE(init_store(writer, cache, config_));
    ASSERT_EQ(writer.batch_put(infos), 1U);
    EXPECT_EQ(writer.batch_put(infos), 1U);
  }
  cache.get_k_cache().zero_();
  cache.get_v_cache().zero_();
  config_.tp_rank = 4;
  KVCacheStore replica;
  ASSERT_TRUE(init_store(replica, cache, config_));
  ASSERT_EQ(replica.batch_get(infos), 1U);
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[0],
                           torch::full_like(cache.get_k_cache()[0], 23)));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[0],
                           torch::full_like(cache.get_v_cache()[0], 47)));
  EXPECT_EQ(replica.batch_put(infos), 1U);
}

TEST_F(KVCacheStoreIntegrationTest, RestoresIndexScaleWithoutAliasing) {
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  DeepSeekV4KVCacheTensors tensors;
  tensors.key_cache = torch::zeros({3, 1, 128, 1, 16}, options);
  tensors.index_cache =
      torch::zeros({3, 1, 128, 1, 8}, options.dtype(torch::kInt8));
  tensors.indexer_cache_scale =
      torch::zeros({3, 1, 128, 1}, options.dtype(torch::kFloat16));
  KVCache c1(tensors);
  tensors.key_cache = torch::zeros_like(tensors.key_cache);
  tensors.index_cache = torch::zeros_like(tensors.index_cache);
  tensors.indexer_cache_scale = torch::zeros_like(tensors.indexer_cache_scale);
  KVCache c2(tensors);

  HostCacheStoreIndex index;
  index[BlockType::KV].emplace_back(
      HostCacheStoreEntry{/*cache_handle=*/0,
                          "main",
                          std::nullopt,
                          /*is_target=*/true,
                          {host_cache_store_group(c1, BlockType::KV, {0}),
                           host_cache_store_group(c2, BlockType::KV, {1})},
                          128});
  KVCacheStore store;
  KVCacheStoreInitConfig config = make_store_config("deepseek-v41");
  config.model_id +=
      ":" + std::to_string(getpid()) + ":" +
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  config.master_server_address = config_.master_server_address;
  config.metadata_server = config_.metadata_server;
  ASSERT_TRUE(store.init(config, std::move(index)));

  const std::vector<KVCache*> caches = {&c1, &c2};
  for (size_t index = 0; index < caches.size(); ++index) {
    const auto roles = caches[index]->get_block_type_tensors(BlockType::KV);
    ASSERT_EQ(roles.size(), 3u);
    for (const auto& [role, tensor] : roles) {
      tensor[0].fill_(static_cast<int32_t>(index + 1) * 10 +
                      static_cast<int32_t>(role));
    }
  }
  ASSERT_EQ(store.batch_put({make_block_info(44, BlockType::KV)}), 1U);
  for (KVCache* cache : caches) {
    const BlockType type = BlockType::KV;
    const auto roles = cache->get_block_type_tensors(type);
    const std::vector<BlockTransferInfo> infos = {
        make_block_info(44, type, /*destination_block_id=*/2)};
    Slice<BlockTransferInfo> slice(infos);
    ASSERT_EQ(store.batch_get_with_status(slice), (std::vector<uint8_t>{1}));
    for (const auto& [role, tensor] : roles) {
      EXPECT_TRUE(torch::equal(tensor[0], tensor[2]))
          << static_cast<int32_t>(role);
    }
  }
}

TEST_F(KVCacheStoreIntegrationTest, ReadsValidBuffersDespiteInvalidNeighbors) {
  KVCache cache = make_attention_cache(/*host_blocks=*/3, /*width=*/8);
  cache.get_k_cache()[0].fill_(7);
  cache.get_v_cache()[0].fill_(70);
  MooncakeStoreBackend backend;
  const MooncakeStoreBackendConfig config = backend_config();
  const StoreRegisteredRange key_range{cache.get_k_cache().data_ptr(),
                                       cache.get_k_cache().nbytes()};
  const StoreRegisteredRange value_range{cache.get_v_cache().data_ptr(),
                                         cache.get_v_cache().nbytes()};
  ASSERT_TRUE(backend.init(config, {key_range, key_range, value_range}));
  const std::string key = config_.model_id + ":buffers";
  MultiBuffer source{
      {cache.get_k_cache()[0].data_ptr(), cache.get_v_cache()[0].data_ptr()},
      {cache.get_k_cache()[0].nbytes(), cache.get_v_cache()[0].nbytes()}};
  ASSERT_EQ(backend.batch_put({key}, {source}), std::vector<uint8_t>({1}));
  MultiBuffer buffer{
      {cache.get_k_cache()[2].data_ptr(), cache.get_v_cache()[2].data_ptr()},
      source.sizes};
  MultiBuffer mismatched = buffer;
  mismatched.sizes.clear();
  MultiBuffer oversized = buffer;
  oversized.sizes.front() =
      static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1;
  EXPECT_EQ(backend.batch_get({key, key, key}, {mismatched, buffer, oversized}),
            std::vector<uint8_t>({0, 1, 0}));
  EXPECT_TRUE(torch::equal(cache.get_k_cache()[0], cache.get_k_cache()[2]));
  EXPECT_TRUE(torch::equal(cache.get_v_cache()[0], cache.get_v_cache()[2]));
  EXPECT_EQ(backend.batch_get({key}, {mismatched}), std::vector<uint8_t>({0}));
  MultiBuffer short_buffer = buffer;
  short_buffer.sizes.back() -= sizeof(float);
  EXPECT_EQ(backend.batch_get({key}, {short_buffer}),
            std::vector<uint8_t>({0}));
  EXPECT_TRUE(backend.batch_get({}, {}).empty());
}

TEST_F(KVCacheStoreIntegrationTest, ValidatesRealClientGetByteCounts) {
  const torch::Tensor source = torch::arange(32, torch::kFloat32);
  torch::Tensor destination = torch::zeros_like(source);
  torch::Tensor missing_destination = torch::zeros_like(source);
  torch::Tensor larger_destination = torch::full({33}, -1.0, torch::kFloat32);
  MooncakeStoreBackend backend;
  ASSERT_TRUE(backend.init(
      backend_config(),
      {{source.data_ptr(), source.nbytes()},
       {destination.data_ptr(), destination.nbytes()},
       {missing_destination.data_ptr(), missing_destination.nbytes()},
       {larger_destination.data_ptr(), larger_destination.nbytes()}}));
  const std::string key = config_.model_id + ":read-bytes";
  const MultiBuffer source_buffer{{source.data_ptr()}, {source.nbytes()}};
  ASSERT_EQ(backend.batch_put({key}, {source_buffer}),
            std::vector<uint8_t>({1}));
  const MultiBuffer destination_buffer{{destination.data_ptr()},
                                       {destination.nbytes()}};
  const MultiBuffer missing_buffer{{missing_destination.data_ptr()},
                                   {missing_destination.nbytes()}};
  EXPECT_EQ(backend.batch_get({key, key + ":missing"},
                              {destination_buffer, missing_buffer}),
            std::vector<uint8_t>({1, 0}));
  EXPECT_TRUE(torch::equal(destination, source));

  // RealClient accepts sufficient capacity and returns the object's actual
  // bytes, not destination capacity. That positive short result must be a miss.
  const MultiBuffer larger_buffer{{larger_destination.data_ptr()},
                                  {larger_destination.nbytes()}};
  EXPECT_EQ(backend.batch_get({key}, {larger_buffer}),
            std::vector<uint8_t>({0}));
  EXPECT_TRUE(
      torch::equal(larger_destination.narrow(0, 0, source.numel()), source));
  EXPECT_EQ(larger_destination[32].item<float>(), -1.0F);
}

}  // namespace
}  // namespace xllm
