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

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/kv_cache_store.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"

namespace xllm {
namespace {

std::string make_store_local_hostname(const std::string& configured,
                                      uint32_t worker_id) {
  constexpr uint32_t kDefaultPort = 12345;
  if (configured.empty()) {
    return "127.0.0.1:" + std::to_string(kDefaultPort + worker_id);
  }

  std::string host = configured;
  uint32_t port = kDefaultPort;
  size_t host_end = std::string::npos;
  size_t port_begin = std::string::npos;
  const size_t bracket_end = configured.find("]:");
  if (!configured.empty() && configured.front() == '[' &&
      bracket_end != std::string::npos) {
    host_end = bracket_end + 1;
    port_begin = bracket_end + 2;
  } else {
    const size_t last_colon = configured.rfind(':');
    const bool has_single_colon = last_colon != std::string::npos &&
                                  configured.find(':') == last_colon &&
                                  last_colon + 1 < configured.size();
    if (has_single_colon) {
      host_end = last_colon;
      port_begin = last_colon + 1;
    }
  }
  if (port_begin != std::string::npos) {
    const std::string port_text = configured.substr(port_begin);
    const bool numeric =
        std::all_of(port_text.begin(), port_text.end(), [](char character) {
          return character >= '0' && character <= '9';
        });
    if (numeric) {
      port = static_cast<uint32_t>(std::stoul(port_text));
      host = configured.substr(0, host_end);
    }
  }
  CHECK_LE(worker_id, 65535U);
  CHECK_LE(port, 65535U - worker_id)
      << "Mooncake local endpoint port exceeds 65535.";
  return host + ":" + std::to_string(port + worker_id);
}

bool same_copy_layout(const BlockTypeTensorMap& lhs,
                      const BlockTypeTensorMap& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  auto left = lhs.begin();
  auto right = rhs.begin();
  for (; left != lhs.end(); ++left, ++right) {
    if (left->first != right->first ||
        left->second.scalar_type() != right->second.scalar_type() ||
        left->second.sizes() != right->second.sizes()) {
      return false;
    }
  }
  return true;
}

}  // namespace

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(const Options& options,
                                                   const torch::Device& device)
    : options_(options), device_(device) {
  device_.set_device();
  device_.init_device_context();
  load_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/2,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.load");
}

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(
    const Options& options,
    const torch::Device& device,
    const Stream* compute_stream,
    std::vector<xllm::KVCache>* kv_caches_ptr,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : HierarchyKVCacheTransfer(options, device) {
  CacheRegistration registration;
  registration.role = CacheRole::TARGET;
  registration.enable_mla = options.enable_mla();
  registration.device_kv_caches = kv_caches_ptr;
  registration.kv_cache_shape = kv_cache_shape;
  registration.create_options = create_options;
  registration.producer_stream = compute_stream;
  registration.store_key_component = "main";
  register_cache(std::move(registration));
  CHECK(finalize_registration());
}

HierarchyKVCacheTransfer::~HierarchyKVCacheTransfer() { shutdown(); }

HierarchyKVCacheTransfer::CacheHandle HierarchyKVCacheTransfer::register_cache(
    CacheRegistration registration) {
  CHECK(!registration_finalized_)
      << "Hierarchy KV cache registration is already finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  CHECK(registration.device_kv_caches != nullptr)
      << "Device KV caches must not be null.";
  CHECK(!registration.device_kv_caches->empty())
      << "Device KV caches must not be empty.";
  CHECK(registration.producer_stream != nullptr)
      << "Producer stream must not be null.";
  const auto duplicate_role =
      std::find_if(cache_domains_.begin(),
                   cache_domains_.end(),
                   [&registration](const CacheDomain& domain) {
                     return domain.role == registration.role;
                   });
  CHECK(duplicate_role == cache_domains_.end())
      << "Cache role is already registered.";
  CHECK_LT(cache_domains_.size(),
           static_cast<size_t>(std::numeric_limits<CacheHandle>::max()));

  CacheDomain domain;
  domain.handle = static_cast<CacheHandle>(cache_domains_.size());
  domain.role = registration.role;
  domain.enable_mla = registration.enable_mla;
  domain.device_kv_caches = registration.device_kv_caches;
  domain.kv_cache_shape = std::move(registration.kv_cache_shape);
  domain.create_options = std::move(registration.create_options);
  domain.producer_stream = registration.producer_stream;
  domain.store_key_component = std::move(registration.store_key_component);
  const CacheHandle handle = domain.handle;
  cache_domains_.emplace_back(std::move(domain));
  return handle;
}

bool HierarchyKVCacheTransfer::finalize_registration() {
  CHECK(!registration_finalized_)
      << "Hierarchy KV cache registration is already finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  CHECK(!cache_domains_.empty()) << "No KV cache domain is registered.";

  const Stream* producer_stream = cache_domains_.front().producer_stream;
  for (const CacheDomain& domain : cache_domains_) {
    CHECK_EQ(domain.producer_stream, producer_stream)
        << "All KV cache domains must share one producer stream in phase two.";
  }

  if (options_.host_blocks_factor() > 1.0) {
    int64_t max_layer_count = 0;
    std::vector<HostKVGroupLayout> combined_groups;
    const HostCachePlan* target_plan = nullptr;
    for (CacheRole role : {CacheRole::TARGET, CacheRole::DRAFT}) {
      for (CacheDomain& domain : cache_domains_) {
        if (domain.role != role) {
          continue;
        }
        domain.host_plan = build_host_cache_plan(*domain.device_kv_caches,
                                                 domain.kv_cache_shape,
                                                 domain.create_options,
                                                 options_.host_blocks_factor(),
                                                 options_.instance_role(),
                                                 domain.enable_mla,
                                                 target_plan);
        if (role == CacheRole::TARGET) {
          target_plan = &domain.host_plan;
        }
      }
    }
    int32_t next_group_id = 0;
    for (CacheDomain& domain : cache_domains_) {
      domain.host_layout = std::make_unique<HostKVLayout>(
          create_host_kv_layout(&domain, &next_group_id));
      max_layer_count = std::max<int64_t>(
          max_layer_count,
          static_cast<int64_t>(domain.device_kv_caches->size()));
      const std::vector<HostKVGroupLayout>& domain_groups =
          domain.host_layout->groups();
      combined_groups.insert(
          combined_groups.end(), domain_groups.begin(), domain_groups.end());
    }

    HostKVTransferConfig config;
    config.layer_copy_batches = options_.layers_wise_copy_batchs();
    config.mode = HostKVTransferMode::AUTO;
    host_kv_transfer_ = create_host_kv_transfer(
        HostKVLayout(max_layer_count, std::move(combined_groups), device_),
        device_,
        *producer_stream,
        config);
  }

  if (options_.enable_kvcache_store()) {
    CHECK(options_.host_blocks_factor() > 1.0)
        << "Mooncake Store requires Host cache capacity.";
    HostCacheStoreIndex store_index;
    for (CacheDomain& domain : cache_domains_) {
      CHECK(!domain.store_key_component.empty())
          << "Mooncake Store requires a cache key component.";
      std::map<BlockType, std::vector<HostCacheStoreGroup>> resource_groups;
      for (const HostKVGroupLayout& group : domain.host_layout->groups()) {
        std::vector<int64_t> layer_ids;
        layer_ids.reserve(group.layers.size());
        for (const HostKVLayerLayout& layer : group.layers) {
          CHECK_EQ(layer.group_layer_slot,
                   static_cast<int64_t>(layer_ids.size()));
          layer_ids.emplace_back(layer.absolute_layer_id);
        }
        const BlockType resource = domain.group_resources.at(group.group_id);
        resource_groups[resource].emplace_back(
            HostCacheStoreGroup{std::move(layer_ids), group.host_roles});
      }
      for (auto& [resource, groups] : resource_groups) {
        const bool replicated =
            resource == BlockType::KV && domain.host_plan.kv_tp_replicated;
        store_index[resource].emplace_back(HostCacheStoreEntry{
            domain.handle,
            domain.store_key_component,
            replicated,
            domain.role == CacheRole::TARGET,
            std::move(groups),
            resource == BlockType::KV    ? domain.host_plan.block_size
            : resource == BlockType::SWA ? domain.host_plan.window_block_size
                                         : 0});
      }
    }

    KVCacheStoreInitConfig store_config;
    const std::string store_local_hostname = make_store_local_hostname(
        options_.store_local_hostname(), options_.store_worker_id());
    store_config.localhost_name = store_local_hostname;
    store_config.protocol = options_.store_protocol();
    store_config.rdma_devices = options_.store_rdma_devices();
    store_config.metadata_server = options_.store_metadata_server();
    store_config.master_server_address = options_.store_master_server_address();
    store_config.model_id = options_.store_namespace();
    store_config.tp_rank = options_.tp_rank();
    store_config.tp_size = options_.tp_size();
    store_config.cp_rank = options_.cp_rank();
    store_config.cp_size = options_.cp_size();
    store_config.kv_split_full_domain_size =
        options_.kv_split_full_domain_size();
    store_config.kv_split_size = options_.kv_split_size();
    store_config.kv_split_rank = options_.kv_split_rank();
    // The store's object keys must separate INDEX objects written under
    // different CP index write modes (the shapes are identical; the valid
    // pages are not), so the schema hash carries the process's declared
    // mode.
    store_config.cp_index_write_mode = declared_cp_index_write_mode();
    store_config.enable_mla = options_.enable_mla();
    LOG(INFO) << "[Mooncake][StoreEngine] initialize, endpoint="
              << store_local_hostname << ", protocol=" << store_config.protocol
              << ", worker_rank=" << options_.store_worker_id()
              << ", tp_rank=" << store_config.tp_rank
              << ", tp_size=" << store_config.tp_size
              << ", cp_rank=" << store_config.cp_rank
              << ", cp_size=" << store_config.cp_size
              << ", kv_split_full_domain_size="
              << store_config.kv_split_full_domain_size
              << ", kv_split_size=" << store_config.kv_split_size
              << ", kv_split_rank=" << store_config.kv_split_rank
              << ", enable_mla=" << store_config.enable_mla;
    kv_cache_store_ = std::make_unique<KVCacheStore>();
    CHECK(kv_cache_store_->init(store_config, std::move(store_index)))
        << "Failed to initialize Mooncake Store.";
    LOG(INFO) << "[Mooncake][StoreEngine] ready, endpoint="
              << store_local_hostname << ", protocol=" << store_config.protocol
              << ", worker_rank=" << options_.store_worker_id()
              << ", tp_rank=" << store_config.tp_rank
              << ", tp_size=" << store_config.tp_size
              << ", cp_rank=" << store_config.cp_rank
              << ", cp_size=" << store_config.cp_size
              << ", kv_split_full_domain_size="
              << store_config.kv_split_full_domain_size
              << ", kv_split_size=" << store_config.kv_split_size
              << ", kv_split_rank=" << store_config.kv_split_rank
              << ", enable_mla=" << store_config.enable_mla;
  }
  registration_finalized_ = true;
  return true;
}

void HierarchyKVCacheTransfer::shutdown() {
  if (shutdown_) {
    return;
  }
  shutdown_ = true;
  // No load task may outlive transfer resources or Host cache storage.
  load_threadpool_.reset();
  device_.set_device();
  if (host_kv_transfer_ != nullptr) {
    host_kv_transfer_->drain();
    host_kv_transfer_.reset();
  }
  kv_cache_store_.reset();
  for (CacheDomain& domain : cache_domains_) {
    domain.host_layout.reset();
    domain.host_regions.clear();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  load_handles_.clear();
}

HostKVLayout HierarchyKVCacheTransfer::create_host_kv_layout(
    CacheDomain* domain,
    int32_t* next_group_id) const {
  CHECK(domain != nullptr);
  CHECK(next_group_id != nullptr);
  std::vector<HostKVGroupLayout> groups;
  std::map<BlockType, std::vector<size_t>> resource_groups;
  const int64_t num_layers =
      static_cast<int64_t>(domain->device_kv_caches->size());
  groups.reserve(static_cast<size_t>(num_layers));
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    KVCache& cache = domain->device_kv_caches->at(layer_id);
    for (const BlockType resource :
         {BlockType::KV, BlockType::SWA, BlockType::LINEAR}) {
      BlockTypeTensorMap tensors =
          cache.get_block_type_tensors(resource, /*checkpoint_row=*/0);
      if (tensors.empty()) {
        continue;
      }
      if (!domain->kv_cache_shape.has_grouped_cache_layout()) {
        for (auto& [role, tensor] : tensors) {
          tensor = fold_tensor_rows_for_transfer(
              tensor, role, domain->kv_cache_shape.replicated_block_pages());
        }
      }
      std::vector<size_t>& candidates = resource_groups[resource];
      const auto compatible =
          std::find_if(candidates.begin(), candidates.end(), [&](size_t index) {
            return same_copy_layout(groups[index].layers.front().device_roles,
                                    tensors);
          });
      size_t index = 0;
      if (compatible == candidates.end()) {
        CHECK_LT(*next_group_id, std::numeric_limits<int32_t>::max());
        index = groups.size();
        groups.emplace_back(HostKVGroupLayout{(*next_group_id)++});
        candidates.emplace_back(index);
        domain->group_resources.emplace(groups.back().group_id, resource);
      } else {
        index = *compatible;
      }
      HostKVGroupLayout& group = groups[index];
      group.layers.emplace_back(
          HostKVLayerLayout{layer_id,
                            static_cast<int64_t>(group.layers.size()),
                            std::move(tensors),
                            &cache,
                            resource});
    }
  }
  size_t allocation_count = 0;
  for (const HostKVGroupLayout& group : groups) {
    allocation_count += group.layers.front().device_roles.size();
  }
  domain->host_regions.reserve(allocation_count);
  for (HostKVGroupLayout& group : groups) {
    const BlockType resource = domain->group_resources.at(group.group_id);
    for (const auto& [role, tensor] : group.layers.front().device_roles) {
      std::vector<int64_t> shape = tensor.sizes().vec();
      shape[0] = domain->host_plan.block_counts.at(resource);
      shape.insert(shape.begin() + 1,
                   static_cast<int64_t>(group.layers.size()));
      torch::Tensor host_tensor;
      domain->host_regions.emplace_back();
      create_host_page_aligned_tensor(shape,
                                      tensor.scalar_type(),
                                      &host_tensor,
                                      &domain->host_regions.back());
      group.host_roles.emplace(role, std::move(host_tensor));
    }
  }
  return HostKVLayout(num_layers, std::move(groups), device_);
}

HostKVRequest HierarchyKVCacheTransfer::make_request(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    TransferType transfer_type) const {
  HostKVRequest request;
  request.target_mappings.reserve(block_transfer_info.size());
  request.draft_mappings.reserve(block_transfer_info.size());
  for (const BlockTransferInfo& info : block_transfer_info) {
    CHECK(info.transfer_type == transfer_type)
        << "Host KV batch contains mixed transfer types.";
    const bool is_load = transfer_type == TransferType::H2D;
    bool has_participating_domain = false;
    for (const CacheDomain& domain : cache_domains_) {
      for (const auto& [group_id, resource] : domain.group_resources) {
        if (info.block_type != resource) {
          continue;
        }
        has_participating_domain = true;
        std::vector<HostKVMapping>& mappings = domain.role == CacheRole::DRAFT
                                                   ? request.draft_mappings
                                                   : request.target_mappings;
        mappings.emplace_back(
            HostKVMapping{group_id,
                          is_load ? info.src_block_id : info.dst_block_id,
                          is_load ? info.dst_block_id : info.src_block_id,
                          info.checkpoint_row});
      }
    }
    CHECK(has_participating_domain)
        << "No KV cache domain supports block type "
        << static_cast<int32_t>(info.block_type) << ".";
  }
  return request;
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  CHECK(!block_transfer_info.empty());
  device_.set_device();
  switch (block_transfer_info.front().transfer_type) {
    case TransferType::D2H2G:
      return offload(block_transfer_info);
    case TransferType::H2D: {
      if (host_kv_transfer_ == nullptr) {
        LOG(ERROR) << "Host KV load requested without Host cache.";
        return 0;
      }
      HostKVRequest request =
          make_request(block_transfer_info, TransferType::H2D);
      HostKVLoadHandle handle =
          host_kv_transfer_->prepare_load(!request.draft_mappings.empty());
      CHECK(handle.synchronizer != nullptr)
          << "Failed to create Host KV load synchronizer.";
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto existing = load_handles_.find(batch_id);
        if (existing != load_handles_.end()) {
          LOG(ERROR) << "Host KV load handle collision at batch_id=" << batch_id
                     << "; replacing the unconsumed handle.";
          existing->second.synchronizer->abort();
        }
        load_handles_[batch_id] = handle;
      }
      load_threadpool_->schedule(
          [this, request = std::move(request), handle]() mutable {
            load_from_host(request, handle);
          });
      return static_cast<uint32_t>(block_transfer_info.size());
    }
    default:
      LOG(ERROR) << "Unsupported transfer type: "
                 << static_cast<uint32_t>(
                        block_transfer_info.front().transfer_type);
      return 0;
  }
}

std::vector<uint8_t> HierarchyKVCacheTransfer::prefetch_kv_blocks(
    Slice<BlockTransferInfo>& block_transfer_info,
    StoreGetStats* stats) {
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  CHECK(!block_transfer_info.empty());
  if (!options_.enable_kvcache_store() || kv_cache_store_ == nullptr ||
      block_transfer_info[0].transfer_type != TransferType::G2H) {
    LOG(ERROR) << "Unsupported prefetch transfer type: "
               << static_cast<uint32_t>(block_transfer_info[0].transfer_type);
    return std::vector<uint8_t>(block_transfer_info.size(), /*value=*/0);
  }
  std::vector<uint8_t> hits =
      kv_cache_store_->batch_get_with_status(block_transfer_info, stats);
  const size_t hit_count =
      std::count(hits.begin(), hits.end(), static_cast<uint8_t>(1));
  VLOG(1) << "[Mooncake][PrefetchGet] worker_rank="
          << options_.store_worker_id()
          << ", kv_split_size=" << options_.kv_split_size()
          << ", kv_split_rank=" << options_.kv_split_rank() << ", type="
          << static_cast<int32_t>(block_transfer_info[0].block_type)
          << ", blocks=" << hits.size() << ", hits=" << hit_count;
  return hits;
}

std::vector<uint8_t> HierarchyKVCacheTransfer::probe_kv_blocks(
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  if (kv_cache_store_ == nullptr) {
    return std::vector<uint8_t>(block_transfer_info.size(), /*value=*/0);
  }
  return kv_cache_store_->batch_exist(block_transfer_info);
}

bool HierarchyKVCacheTransfer::supports_block_type(BlockType block_type) const {
  return std::any_of(cache_domains_.begin(),
                     cache_domains_.end(),
                     [block_type](const CacheDomain& domain) {
                       return std::any_of(domain.group_resources.begin(),
                                          domain.group_resources.end(),
                                          [block_type](const auto& entry) {
                                            return block_type == entry.second;
                                          });
                     });
}

bool HierarchyKVCacheTransfer::supports_block_type(CacheRole role,
                                                   BlockType block_type) const {
  const auto domain = std::find_if(
      cache_domains_.begin(),
      cache_domains_.end(),
      [role](const CacheDomain& candidate) { return candidate.role == role; });
  return domain != cache_domains_.end() &&
         std::any_of(domain->group_resources.begin(),
                     domain->group_resources.end(),
                     [block_type](const auto& entry) {
                       return block_type == entry.second;
                     });
}

uint32_t HierarchyKVCacheTransfer::offload(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (host_kv_transfer_ == nullptr) {
    return static_cast<uint32_t>(block_transfer_info.size());
  }
  HostKVRequest request =
      make_request(block_transfer_info, TransferType::D2H2G);
  if (!offload_to_host(request)) {
    LOG(ERROR) << "Offload to Host cache failed.";
    return 0;
  }
  if (kv_cache_store_ != nullptr) {
    const uint32_t put_count = kv_cache_store_->batch_put(block_transfer_info);
    if (put_count != block_transfer_info.size()) {
      LOG(WARNING) << "Mooncake BatchPut partially failed: " << put_count << "/"
                   << block_transfer_info.size();
    }
    VLOG(1) << "[Mooncake][OffloadPut] worker_rank="
            << options_.store_worker_id() << ", tp_rank=" << options_.tp_rank()
            << ", tp_size=" << options_.tp_size()
            << ", cp_rank=" << options_.cp_rank()
            << ", cp_size=" << options_.cp_size()
            << ", kv_split_full_domain_size="
            << options_.kv_split_full_domain_size()
            << ", kv_split_size=" << options_.kv_split_size()
            << ", kv_split_rank=" << options_.kv_split_rank()
            << ", enable_mla=" << options_.enable_mla()
            << ", blocks=" << block_transfer_info.size()
            << ", success=" << put_count;
  }
  return static_cast<uint32_t>(block_transfer_info.size());
}

bool HierarchyKVCacheTransfer::offload_to_host(const HostKVRequest& request) {
  return host_kv_transfer_->offload(request);
}

bool HierarchyKVCacheTransfer::load_from_host(const HostKVRequest& request,
                                              const HostKVLoadHandle& handle) {
  return host_kv_transfer_->load(request, handle);
}

void HierarchyKVCacheTransfer::set_layer_synchronizer(
    ModelInputParams& params) {
  std::optional<HostKVLoadHandle> handle =
      take_load_handle(params.meta.batch_id);
  if (params.meta.requires_host_restore && !handle.has_value()) {
    if (params.parallel.layer_wise_load_synchronizer != nullptr) {
      if (params.parallel.layer_wise_load_batch_id == params.meta.batch_id) {
        // A derived target/draft input re-entered with a stale
        // requires_host_restore flag: the first consumer already took the
        // handle and installed the synchronizer, and this copy retained it,
        // so no second take is needed.
        LOG(WARNING) << "Stale Host restore flag at batch_id="
                     << params.meta.batch_id
                     << "; retaining installed synchronizer";
        params.meta.requires_host_restore = false;
        return;
      }
      // The installed synchronizer belongs to a different batch: reusing it
      // would synchronize this batch's layers against another batch's
      // events. Fail this batch instead of silently trusting foreign state.
      LOG(ERROR) << "Installed Host restore synchronizer at batch_id="
                 << params.meta.batch_id << " belongs to batch_id="
                 << params.parallel.layer_wise_load_batch_id
                 << "; failing the batch";
      throw std::runtime_error(
          "Host restore synchronizer belongs to another batch at batch_id=" +
          std::to_string(params.meta.batch_id));
    }
    // The restore bookkeeping lost this batch handle before any synchronizer
    // was installed (e.g. a lost per-rank registration). Aborting here
    // amplifies one degraded batch into a full-group restart (worker loss ->
    // master exit); fail this batch instead.
    LOG(ERROR) << "Missing Host restore handle at batch_id="
               << params.meta.batch_id << "; failing the batch";
    throw std::runtime_error("Host restore handle missing at batch_id=" +
                             std::to_string(params.meta.batch_id));
  }
  if (!handle.has_value()) {
    return;
  }
  CHECK(handle->synchronizer != nullptr)
      << "Missing Host restore synchronizer at batch_id="
      << params.meta.batch_id;
  CHECK_GT(handle->layers_per_event, 0U)
      << "Invalid Host restore granularity at batch_id="
      << params.meta.batch_id;
  CHECK_GT(handle->synchronizer->size(), 0U)
      << "Empty Host restore synchronizer at batch_id=" << params.meta.batch_id;
  if (handle->draft_event_index.has_value()) {
    CHECK_LT(*handle->draft_event_index, handle->synchronizer->size())
        << "Invalid Host draft restore event at batch_id="
        << params.meta.batch_id;
  }
  params.parallel.layer_wise_load_synchronizer = handle->synchronizer;
  params.parallel.layer_wise_load_batch_id = params.meta.batch_id;
  params.parallel.layers_per_event = handle->layers_per_event;
  params.parallel.draft_load_event_index = handle->draft_event_index;
  // Derived target/draft inputs retain the synchronizer, not a second request
  // to consume the same batch's handle.
  params.meta.requires_host_restore = false;
}

std::optional<HostKVLoadHandle> HierarchyKVCacheTransfer::take_load_handle(
    uint64_t batch_id) {
  CHECK(registration_finalized_)
      << "Hierarchy KV cache registration is not finalized.";
  CHECK(!shutdown_) << "Hierarchy KV cache transfer is shut down.";
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = load_handles_.find(batch_id);
  if (it == load_handles_.end()) {
    return std::nullopt;
  }
  HostKVLoadHandle handle = std::move(it->second);
  load_handles_.erase(it);
  return handle;
}

}  // namespace xllm
