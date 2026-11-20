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

#include "framework/kv_cache/prefill_layerwise_kv_cache.h"

#include <folly/ScopeGuard.h>

#include <algorithm>
#include <chrono>
#include <utility>

#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/process_group.h"
#include "platform/device.h"
#if defined(USE_NPU)
#include "platform/npu/npu_batch_memcpy.h"
#endif

namespace xllm {
namespace {

constexpr int32_t kCopyStreamTimeoutMs = 60000;

torch::Tensor allocate_device_tensor(const std::vector<int64_t>& shape,
                                     torch::ScalarType dtype,
                                     const torch::Device& device,
                                     const KVCacheCreateOptions& options,
                                     bool indexer_scale = false) {
  if (device.is_cpu()) {
    return torch::zeros(shape, torch::dtype(dtype).device(device));
  }
#if defined(USE_NPU)
  const aclFormat format = indexer_scale
                               ? ACL_FORMAT_ND
                               : get_npu_kv_cache_format(options.model_type());
  // Keep the same allocator selection as the regular KV cache path; the
  // layerwise device slots do not require huge pages for the batched D2H/H2D
  // copies (the Host side is page-aligned separately).
  if (options.enable_kv_cache_huge_page_allocator()) {
    return alloc_npu_huge_page_tensor(shape, dtype, format);
  }
  return at_npu::native::npu_format_cast(
      torch::empty(shape, torch::dtype(dtype).device(device)), format);
#else
  LOG(FATAL) << "Prefill layerwise KV offload requires NPU.";
  return {};
#endif
}

torch::Tensor allocate_host_tensor(const torch::Tensor& device_tensor,
                                   int64_t* host_bytes) {
  if (!device_tensor.defined()) {
    return {};
  }
  if (device_tensor.device().is_cpu()) {
    *host_bytes += static_cast<int64_t>(device_tensor.nbytes());
    return torch::zeros_like(device_tensor);
  }
  auto region = std::make_shared<HostPageAlignedRegion>();
  torch::Tensor tensor;
  create_host_page_aligned_tensor(device_tensor.sizes().vec(),
                                  device_tensor.scalar_type(),
                                  &tensor,
                                  region.get());
  *host_bytes += static_cast<int64_t>(region->total_bytes);
  // Transfer registration is destroyed before the cache owner. Tensor views
  // retain this allocation too, including tensors held by the model executor.
  return torch::from_blob(
      region->base_ptr,
      tensor.sizes(),
      [region](void* /*unused*/) {},
      tensor.options());
}

KVCache make_cache(const KVCacheTensors& kv,
                   const torch::Tensor& index,
                   const std::optional<torch::Tensor>& scale) {
  if (!index.defined()) {
    return KVCache(kv);
  }
  IndexedKVCacheTensors tensors;
  tensors.kv_cache_tensors = kv;
  tensors.index_cache = index;
  tensors.index_cache_scale = scale;
  return KVCache(tensors);
}

void normalize_blocks(std::vector<int64_t>* blocks) {
  std::sort(blocks->begin(), blocks->end());
  blocks->erase(std::unique(blocks->begin(), blocks->end()), blocks->end());
}

}  // namespace

PrefillLayerwiseKVCache::PrefillLayerwiseKVCache(
    const KVCacheShape& shape,
    const KVCacheCreateOptions& options,
    int32_t num_buffers,
    int64_t host_memory_bytes,
    std::vector<KVCache>& device_caches)
    : device_caches_(&device_caches),
      num_buffers_(num_buffers),
      block_count_(shape.key_cache_shape().at(0)) {
  // These are startup allocation invariants, not request validation.
  CHECK(device_caches.empty());
  CHECK_GT(options.num_layers(), 0);
  CHECK_GT(num_buffers_, 0);
  CHECK_LE(num_buffers_, options.num_layers());
  CHECK_GT(host_memory_bytes, 0);
  CHECK_GT(block_count_, 0);
  CHECK(!shape.has_grouped_cache_layout());
  CHECK(!shape.has_kpool_tail_shape());
  CHECK(!options.enable_linear_attention());
  CHECK(!options.enable_kv_cache_quant());
  CHECK(options.layer_cache_owned().empty());

  const int64_t layers = options.num_layers();
  const std::vector<bool>& index_mask = options.indexer_cache_enabled_layers();
  CHECK(index_mask.empty() || index_mask.size() == static_cast<size_t>(layers));
  const std::vector<char> slot_has_index = prefill_layerwise_index_slots(
      layers, num_buffers_, shape.has_index_cache_shape(), index_mask);
  num_buffers_ =
      static_cast<int32_t>(std::min<int64_t>(num_buffers_, layers - 1));
  const auto has_index = [&shape, &index_mask](int64_t layer) {
    return shape.has_index_cache_shape() &&
           (index_mask.empty() || index_mask[static_cast<size_t>(layer)]);
  };

  std::vector<KVCache> slots;
  slots.reserve(slot_has_index.size());
  const torch::Device& device = options.device();
  for (size_t slot = 0; slot < slot_has_index.size(); ++slot) {
    KVCacheTensors kv;
    kv.key_cache = allocate_device_tensor(
        shape.key_cache_shape(), options.dtype(), device, options);
    if (shape.has_value_cache_shape()) {
      kv.value_cache = allocate_device_tensor(
          shape.value_cache_shape(), options.dtype(), device, options);
    }
    torch::Tensor index;
    std::optional<torch::Tensor> scale;
    if (slot_has_index[static_cast<size_t>(slot)]) {
      index = allocate_device_tensor(
          shape.index_cache_shape(),
          options.enable_indexer_cache_quant() ? torch::kChar : options.dtype(),
          device,
          options);
      if (options.enable_indexer_cache_quant()) {
        CHECK(shape.has_index_cache_scale_shape());
        scale = allocate_device_tensor(shape.index_cache_scale_shape(),
                                       torch::kFloat16,
                                       device,
                                       options,
                                       /*indexer_scale=*/true);
      }
    }
    slots.emplace_back(make_cache(kv, index, scale));
    for (const KVCacheTensor& tensor : slots.back().get_cache_tensors()) {
      device_bytes_ += static_cast<int64_t>(tensor.tensor.nbytes());
    }
  }

  device_caches.reserve(static_cast<size_t>(layers));
  host_caches_.reserve(static_cast<size_t>(layers));
  for (int64_t layer = 0; layer < layers; ++layer) {
    const KVCache& slot = slots[static_cast<size_t>(
        prefill_layerwise_buffer_index(layer, num_buffers_))];
    const KVCacheTensors device_kv{slot.get_k_cache(), slot.get_v_cache()};
    const torch::Tensor index =
        has_index(layer) ? slot.get_index_cache() : torch::Tensor{};
    const std::optional<torch::Tensor> scale =
        has_index(layer) ? slot.get_indexer_cache_scale() : std::nullopt;
    device_caches.emplace_back(make_cache(device_kv, index, scale));
    KVCacheTensors host_kv{
        allocate_host_tensor(device_kv.key_cache, &host_bytes_),
        allocate_host_tensor(device_kv.value_cache, &host_bytes_)};
    std::optional<torch::Tensor> host_scale;
    if (scale.has_value()) {
      host_scale = allocate_host_tensor(*scale, &host_bytes_);
    }
    host_caches_.emplace_back(make_cache(
        host_kv, allocate_host_tensor(index, &host_bytes_), host_scale));
    CHECK_LE(host_bytes_, host_memory_bytes)
        << "Prefill layerwise Host cache exceeds its configured budget.";
  }
  load_ready_.resize(static_cast<size_t>(layers));
  save_ready_.resize(static_cast<size_t>(layers));
  push_completed_layers_.resize(static_cast<size_t>(layers), 0);
  if (!device.is_cpu()) {
    Device platform_device(device);
#if defined(USE_NPU)
    batch_memcpy_ =
        std::make_unique<npu::NPUBatchMemcpy>(/*submit_asynchronously=*/true);
    batch_memcpy_->init(device.index());
#endif
    CHECK(batch_memcpy_ != nullptr);
    load_stream_ = platform_device.get_stream_from_pool(kCopyStreamTimeoutMs);
    save_stream_ = platform_device.get_stream_from_pool(kCopyStreamTimeoutMs);
  }
  LOG(INFO) << "Prefill layerwise KV cache: layers=" << layers
            << ", resident_layer=0, shared_buffers=" << num_buffers_
            << ", device_bytes=" << device_bytes_
            << ", host_bytes=" << host_bytes_ << ", PD PUSH source=Device";
}

PrefillLayerwiseKVCache::~PrefillLayerwiseKVCache() { drain(); }

uint32_t PrefillLayerwiseKVCache::size() const {
  return static_cast<uint32_t>(host_caches_.size());
}

void PrefillLayerwiseKVCache::require_usable() const {
  if (failed_) {
    LOG(FATAL) << "Prefill layerwise KV cache failed and cannot be reused; "
                  "restart the worker before serving more prefills.";
  }
}

bool PrefillLayerwiseKVCache::begin(const ModelInputParams& params,
                                    int64_t block_size,
                                    const Stream* compute_stream,
                                    bool expect_push) {
  require_usable();
  if (!params.block_copy.swap_blocks.empty()) {
    LOG(ERROR) << "Prefill layerwise KV offload does not support beam-search "
                  "block copy, batch_id="
               << params.meta.batch_id;
    return false;
  }
  std::vector<int64_t> read_blocks;
  std::vector<int64_t> write_blocks;
  const torch::Tensor& tables = params.attention.host.block_tables;
  const std::vector<int32_t>& kv_lengths = params.attention.host.kv_seq_lens;
  const std::vector<int32_t>& q_lengths = params.attention.host.q_seq_lens;
  // Untrusted batch metadata is rejected before the first KV DMA submission.
  bool valid = block_size > 0 && params.meta.batch_forward_type.no_decode() &&
               tables.defined() && tables.device().is_cpu() &&
               tables.dim() == 2 && tables.scalar_type() == torch::kInt32 &&
               tables.is_contiguous() &&
               tables.size(0) == static_cast<int64_t>(kv_lengths.size()) &&
               q_lengths.size() == kv_lengths.size();
  if (valid) {
    const auto block_ids = tables.accessor<int32_t, 2>();
    read_blocks.reserve(static_cast<size_t>(tables.numel()));
    write_blocks.reserve(static_cast<size_t>(tables.numel()));
    for (size_t row = 0; row < kv_lengths.size(); ++row) {
      const int64_t kv_length = kv_lengths[row];
      const int64_t q_length = q_lengths[row];
      const int64_t count = (kv_length + block_size - 1) / block_size;
      if (q_length < 0 || kv_length < q_length || count > tables.size(1)) {
        valid = false;
        break;
      }
      const int64_t prefix_length = kv_length - q_length;
      const int64_t first_write_block = prefix_length / block_size;
      const bool first_write_block_has_prefix = prefix_length % block_size != 0;
      for (int64_t col = 0; col < count; ++col) {
        // Blocks after the first write block contain only this batch's tokens,
        // so attention writes them before reading and an H2D load would move
        // data that is never observed.
        if (col < first_write_block ||
            (col == first_write_block && first_write_block_has_prefix)) {
          read_blocks.emplace_back(block_ids[row][col]);
        }
        if (q_length > 0 && col >= first_write_block) {
          write_blocks.emplace_back(block_ids[row][col]);
        }
      }
    }
  }
  if (!valid) {
    LOG(ERROR) << "Invalid prefill layerwise KV metadata, batch_id="
               << params.meta.batch_id;
    return false;
  }
  return begin(std::move(read_blocks),
               std::move(write_blocks),
               compute_stream,
               expect_push);
}

bool PrefillLayerwiseKVCache::begin(const ModelInputParams& params,
                                    int64_t block_size,
                                    const Stream* compute_stream,
                                    bool expect_push,
                                    ProcessGroup* preparation_group) {
  const bool prepared = begin(params, block_size, compute_stream, expect_push);
  if (preparation_group == nullptr || preparation_group->world_size() <= 1) {
    return prepared;
  }
  auto cancel = folly::makeGuard([this, prepared]() {
    if (prepared) {
      cancel_preparation();
    }
  });
  const torch::TensorOptions options =
      torch::dtype(torch::kInt64)
          .device((*device_caches_)[0].get_k_cache().device());
  const torch::Tensor local_status =
      torch::full({1}, static_cast<int64_t>(prepared), options);
  const torch::Tensor rank_status =
      preparation_group->allgather_base_sync(local_status).to(torch::kCPU);
  // A malformed collective result cannot give a rank-uniform execution
  // decision; continuing would let some ranks enter model collectives alone.
  CHECK_EQ(rank_status.numel(), preparation_group->world_size())
      << "Invalid prefill KV preparation collective result.";
  const bool all_prepared = rank_status.eq(1).all().item<bool>();
  if (all_prepared) {
    cancel.dismiss();
  } else {
    LOG(ERROR) << "Prefill KV preparation rejected across TP ranks, batch_id="
               << params.meta.batch_id << ", rank=" << preparation_group->rank()
               << ", local_prepared=" << prepared;
  }
  return all_prepared;
}

bool PrefillLayerwiseKVCache::begin(std::vector<int64_t> read_blocks,
                                    std::vector<int64_t> write_blocks,
                                    const Stream* compute_stream,
                                    bool expect_push) {
  require_usable();
  normalize_blocks(&read_blocks);
  normalize_blocks(&write_blocks);
  const auto valid_blocks = [this](const std::vector<int64_t>& blocks) {
    return blocks.empty() ||
           (blocks.front() >= 0 && blocks.back() < block_count_);
  };
  if (active_ || !valid_blocks(read_blocks) || !valid_blocks(write_blocks) ||
      (batch_memcpy_ != nullptr && compute_stream == nullptr)) {
    LOG(ERROR) << "Invalid prefill layerwise KV batch: active=" << active_
               << ", failed=" << failed_ << ", block_count=" << block_count_;
    return false;
  }
  compute_stream_ = compute_stream;
  read_blocks_ = std::move(read_blocks);
  write_blocks_ = std::move(write_blocks);
  current_layer_ = -1;
  next_load_layer_ = 1 + num_buffers_;
  std::fill(load_ready_.begin(), load_ready_.end(), nullptr);
  std::fill(save_ready_.begin(), save_ready_.end(), nullptr);
  // Layer 0 can compute immediately from its resident prefix while these
  // initial shared-buffer loads run on the independent H2D stream.
  for (int64_t layer = 1; layer < next_load_layer_; ++layer) {
    if (!load_layer(layer)) {
      // Only private Device scratch has been touched so far. Drain it before
      // returning a request error; no persistent Host data needs repair.
      cancel_preparation();
      return false;
    }
  }
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    expect_push_ = expect_push;
    push_finished_ = !expect_push;
    std::fill(push_completed_layers_.begin(), push_completed_layers_.end(), 0);
  }
  active_ = true;
  return true;
}

bool PrefillLayerwiseKVCache::synchronize_layer(int64_t layer_index) {
  if (!active_ || failed_ || layer_index != current_layer_ + 1 ||
      layer_index >= static_cast<int64_t>(size())) {
    LOG(ERROR) << "Out-of-order prefill KV layer: requested=" << layer_index
               << ", current=" << current_layer_;
    return false;
  }
  if (current_layer_ >= 0 && !save_layer(current_layer_)) {
    abort();
    return false;
  }
  // Prefetch available slots, but do not stall an already-loaded layer behind
  // a future layer's network write. Wait only when the current layer needs
  // that slot. Local save events fence its D2H consumer independently.
  const int64_t last_prefetch = std::min<int64_t>(
      layer_index + num_buffers_ - 1, static_cast<int64_t>(size()) - 1);
  while (next_load_layer_ <= last_prefetch) {
    if (next_load_layer_ > layer_index &&
        !push_layer_complete(next_load_layer_ - num_buffers_)) {
      break;
    }
    if (!load_layer(next_load_layer_)) {
      abort();
      return false;
    }
    ++next_load_layer_;
  }
  if (batch_memcpy_ != nullptr && layer_index > 0 &&
      !compute_stream_->wait_event(
          load_ready_[static_cast<size_t>(layer_index)])) {
    abort();
    return false;
  }
  current_layer_ = layer_index;
  return true;
}

bool PrefillLayerwiseKVCache::copy_layer(int64_t layer_index, bool load) {
  const std::vector<KVCacheTensor> device =
      (*device_caches_)[static_cast<size_t>(layer_index)].get_cache_tensors();
  const std::vector<KVCacheTensor> host =
      host_caches_[static_cast<size_t>(layer_index)].get_cache_tensors();
  const std::vector<int64_t>& blocks = load ? read_blocks_ : write_blocks_;
  if (blocks.empty()) {
    return true;
  }
  std::vector<torch::Tensor> sources;
  std::vector<torch::Tensor> destinations;
  sources.reserve(device.size() * blocks.size());
  destinations.reserve(device.size() * blocks.size());
  for (size_t role = 0; role < device.size(); ++role) {
    for (int64_t block : blocks) {
      sources.emplace_back((load ? host[role] : device[role]).tensor[block]);
      destinations.emplace_back(
          (load ? device[role] : host[role]).tensor[block]);
    }
  }
  if (batch_memcpy_ == nullptr) {
    for (size_t index = 0; index < sources.size(); ++index) {
      destinations[index].copy_(sources[index]);
    }
    return true;
  }
  const bool success = load ? batch_memcpy_->submit_h2d(
                                  sources, destinations, load_stream_.get())
                            : batch_memcpy_->submit_d2h(
                                  sources, destinations, save_stream_.get());
  if (!success) {
    LOG(ERROR) << "Prefill KV copy failed: layer=" << layer_index
               << ", direction=" << (load ? "H2D" : "D2H")
               << ", blocks=" << blocks.size();
  }
  return success;
}

bool PrefillLayerwiseKVCache::load_layer(int64_t layer_index) {
  if (layer_index > num_buffers_ &&
      !wait_push_completion(layer_index - num_buffers_)) {
    return false;
  }
  if (batch_memcpy_ != nullptr && layer_index > num_buffers_) {
    const StreamEventPtr& saved =
        save_ready_[static_cast<size_t>(layer_index - num_buffers_)];
    if (saved == nullptr || !load_stream_->wait_event(saved)) {
      return false;
    }
  }
  if (!copy_layer(layer_index, /*load=*/true)) {
    return false;
  }
  if (batch_memcpy_ != nullptr) {
    load_ready_[static_cast<size_t>(layer_index)] =
        load_stream_->record_event_or_sync();
    return load_ready_[static_cast<size_t>(layer_index)] != nullptr;
  }
  return true;
}

bool PrefillLayerwiseKVCache::save_layer(int64_t layer_index) {
  if (batch_memcpy_ != nullptr) {
    save_stream_->wait_stream(*compute_stream_);
  }
  if (!copy_layer(layer_index, /*load=*/false)) {
    return false;
  }
  if (batch_memcpy_ != nullptr) {
    save_ready_[static_cast<size_t>(layer_index)] =
        save_stream_->record_event_or_sync();
    return save_ready_[static_cast<size_t>(layer_index)] != nullptr;
  }
  return true;
}

void PrefillLayerwiseKVCache::cancel_preparation() {
  // No model or PUSH has started, so only scratch loads need draining. This
  // is distinct from execution abort: authoritative Host data is intact.
  drain();
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    expect_push_ = false;
    push_finished_ = true;
    std::fill(push_completed_layers_.begin(), push_completed_layers_.end(), 0);
  }
  compute_stream_ = nullptr;
  active_ = false;
}

void PrefillLayerwiseKVCache::drain() {
  // Failure to establish completion can corrupt shared cache or release live
  // DMA memory; it cannot safely be converted to a single-request error.
  if (load_stream_ != nullptr) {
    CHECK_EQ(load_stream_->synchronize(), 0) << "Cannot drain prefill KV H2D.";
  }
  if (save_stream_ != nullptr) {
    CHECK_EQ(save_stream_->synchronize(), 0) << "Cannot drain prefill KV D2H.";
  }
}

bool PrefillLayerwiseKVCache::finish() {
  if (!active_ || failed_ ||
      current_layer_ + 1 != static_cast<int64_t>(size())) {
    abort();
    return false;
  }
  const bool success = save_layer(current_layer_);
  drain();
  const bool pushed = success && wait_push_completion(std::nullopt);
  active_ = false;
  failed_ = !pushed;
  return pushed;
}

void PrefillLayerwiseKVCache::abort() {
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    failed_ = true;
  }
  push_cv_.notify_all();
  drain();
  active_ = false;
}

void PrefillLayerwiseKVCache::notify_push_completion(
    std::optional<int64_t> layer,
    bool safe_to_reuse) {
  {
    std::lock_guard<std::mutex> lock(push_mutex_);
    if (!safe_to_reuse ||
        (layer.has_value() &&
         (*layer < 0 || *layer >= static_cast<int64_t>(size())))) {
      failed_ = true;
    } else if (layer.has_value()) {
      push_completed_layers_[static_cast<size_t>(*layer)] = true;
    } else {
      push_finished_ = true;
    }
  }
  push_cv_.notify_all();
}

bool PrefillLayerwiseKVCache::push_layer_complete(int64_t layer_index) {
  std::lock_guard<std::mutex> lock(push_mutex_);
  return !failed_ && (!expect_push_ || push_finished_ ||
                      push_completed_layers_[static_cast<size_t>(layer_index)]);
}

bool PrefillLayerwiseKVCache::wait_push_completion(
    std::optional<int64_t> layer) {
  std::unique_lock<std::mutex> lock(push_mutex_);
  if (!expect_push_) {
    return !failed_;
  }
  const bool completed = push_cv_.wait_for(
      lock, std::chrono::milliseconds(kCopyStreamTimeoutMs), [this, layer]() {
        return failed_ || push_finished_ ||
               (layer.has_value() &&
                push_completed_layers_[static_cast<size_t>(*layer)]);
      });
  if (!completed) {
    LOG(ERROR) << "Timed out waiting for prefill KV PUSH completion, layer="
               << layer.value_or(-1);
    failed_ = true;
  }
  return !failed_;
}

bool PrefillLayerwiseKVCache::record_stream(int64_t /*layer_index*/,
                                            Stream* /*stream*/) {
  LOG(ERROR) << "Prefill layerwise KV cache is a load synchronizer, not an "
                "event recorder.";
  return false;
}

}  // namespace xllm
