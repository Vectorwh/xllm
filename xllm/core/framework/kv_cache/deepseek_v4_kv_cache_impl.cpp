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

#include "framework/kv_cache/deepseek_v4_kv_cache_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <sstream>
#include <utility>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "framework/kv_cache/deepseek_v4_cache_layout.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "platform/platform.h"
#include "util/utils.h"

namespace xllm {
namespace {

torch::Tensor swap_tensor_blocks(const torch::Tensor& tensor,
                                 const torch::Tensor& src_tensor,
                                 const torch::Tensor& dst_tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  torch::Tensor selected = torch::index_select(tensor, 0, src_tensor);
  torch::Tensor result = tensor.clone();
  result.index_copy_(0, dst_tensor, selected);
  return result;
}

torch::Tensor cast_to_nd_format(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
#if defined(USE_NPU)
  if (!tensor.device().is_privateuseone()) {
    return tensor;
  }
  return at_npu::native::npu_format_cast(tensor, ACL_FORMAT_ND);
#else
  return tensor;
#endif
}

Dsv4StateCache make_dsv4_state_cache(torch::Tensor packed,
                                     torch::Tensor kv,
                                     torch::Tensor score) {
  if (Platform::is_mlu()) {
    return Dsv4StateCache::from_packed(
        std::move(packed), std::move(kv), std::move(score));
  }
  return Dsv4StateCache::from_split(std::move(kv), std::move(score));
}

std::string tensor_shape_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << tensor.sizes();
  return oss.str();
}

}  // namespace

Dsv4StateCache Dsv4StateCache::from_split(torch::Tensor kv,
                                          torch::Tensor score) {
  Dsv4StateCache state;
  state.kv_ = std::move(kv);
  state.score_ = std::move(score);
  return state;
}

Dsv4StateCache Dsv4StateCache::from_packed(torch::Tensor packed,
                                           torch::Tensor fallback_kv,
                                           torch::Tensor fallback_score) {
  Dsv4StateCache state;
  state.packed_layout_ = true;
  if (packed.defined()) {
    state.packed_ = std::move(packed);
  } else {
    state.kv_ = std::move(fallback_kv);
    state.score_ = std::move(fallback_score);
  }
  return state;
}

torch::Tensor Dsv4StateCache::kv() const {
  if (!packed_.defined()) {
    return kv_;
  }
  const int64_t state_width = packed_.size(2) / 2;
  return packed_.narrow(/*dim=*/2, /*start=*/0, /*length=*/state_width);
}

torch::Tensor Dsv4StateCache::score() const {
  if (!packed_.defined()) {
    return score_;
  }
  const int64_t state_width = packed_.size(2) / 2;
  return packed_.narrow(
      /*dim=*/2, /*start=*/state_width, /*length=*/state_width);
}

torch::Tensor Dsv4StateCache::packed() const { return packed_; }

void Dsv4StateCache::swap_blocks(const torch::Tensor& src,
                                 const torch::Tensor& dst) {
  if (packed_layout_) {
    packed_ = swap_tensor_blocks(packed_, src, dst);
    return;
  }
  kv_ = swap_tensor_blocks(kv_, src, dst);
  score_ = swap_tensor_blocks(score_, src, dst);
}

DeepSeekV4KVCacheImpl::DeepSeekV4KVCacheImpl(DeepSeekV4KVCacheTensors tensors)
    : key_cache_(std::move(tensors.key_cache)),
      index_cache_(std::move(tensors.index_cache)),
      indexer_cache_scale_(std::move(tensors.indexer_cache_scale)),
      swa_cache_(std::move(tensors.swa_cache)),
      compress_state_(
          make_dsv4_state_cache(std::move(tensors.compress_state),
                                std::move(tensors.compress_kv_state),
                                std::move(tensors.compress_score_state))),
      index_state_(make_dsv4_state_cache(
          std::move(tensors.compress_index_state),
          std::move(tensors.compress_index_kv_state),
          std::move(tensors.compress_index_score_state))) {}

torch::Tensor DeepSeekV4KVCacheImpl::get_k_cache() const { return key_cache_; }

torch::Tensor DeepSeekV4KVCacheImpl::get_index_cache() const {
  return index_cache_;
}

std::optional<torch::Tensor> DeepSeekV4KVCacheImpl::get_indexer_cache_scale()
    const {
  if (indexer_cache_scale_.defined() && indexer_cache_scale_.numel() > 0) {
    return indexer_cache_scale_;
  }
  return std::nullopt;
}

torch::Tensor DeepSeekV4KVCacheImpl::get_swa_cache() const {
  return swa_cache_;
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_kv_state() const {
  return compress_state_.kv();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_score_state() const {
  return compress_state_.score();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_kv_state() const {
  return index_state_.kv();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_score_state() const {
  return index_state_.score();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_state() const {
  return compress_state_.packed();
}

torch::Tensor DeepSeekV4KVCacheImpl::get_compress_index_state() const {
  return index_state_.packed();
}

std::vector<KVCacheTensor> DeepSeekV4KVCacheImpl::get_cache_tensors() const {
  std::vector<KVCacheTensor> tensors;
  tensors.reserve(8);
  auto add_tensor = [&tensors](KVCacheTensorRole role,
                               const torch::Tensor& tensor,
                               BlockType block_type) {
    if (tensor.defined() && tensor.numel() > 0) {
      tensors.emplace_back(
          KVCacheTensor{role, tensor, cache_group_id(block_type)});
    }
  };

  add_tensor(KVCacheTensorRole::SWA, swa_cache_, BlockType::SWA);
  add_tensor(KVCacheTensorRole::KEY, key_cache_, BlockType::KV);
  add_tensor(KVCacheTensorRole::INDEX, index_cache_, BlockType::KV);
  add_tensor(
      KVCacheTensorRole::INDEX_SCALE, indexer_cache_scale_, BlockType::KV);
  if (Platform::is_mlu()) {
    add_tensor(KVCacheTensorRole::COMPRESS_STATE,
               compress_state_.packed(),
               BlockType::SWA);
    add_tensor(KVCacheTensorRole::COMPRESS_INDEX_STATE,
               index_state_.packed(),
               BlockType::SWA);
  } else {
    add_tensor(
        KVCacheTensorRole::KV_STATE, compress_state_.kv(), BlockType::SWA);
    add_tensor(KVCacheTensorRole::SCORE_STATE,
               compress_state_.score(),
               BlockType::SWA);
    add_tensor(
        KVCacheTensorRole::INDEX_KV_STATE, index_state_.kv(), BlockType::SWA);
    add_tensor(KVCacheTensorRole::INDEX_SCORE_STATE,
               index_state_.score(),
               BlockType::SWA);
  }
  return tensors;
}

BlockTypeTensorMap DeepSeekV4KVCacheImpl::get_block_type_tensors(
    BlockType type) const {
  BlockTypeTensorMap tensors;
  auto add_tensor = [&tensors](KVCacheTensorRole role,
                               const torch::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() > 0) {
      tensors.emplace(role, tensor);
    }
  };
  if (type == BlockType::SWA) {
    add_tensor(KVCacheTensorRole::SWA, swa_cache_);
  } else if (type == BlockType::KV) {
    add_tensor(KVCacheTensorRole::KEY, key_cache_);
    add_tensor(KVCacheTensorRole::INDEX, index_cache_);
    add_tensor(KVCacheTensorRole::INDEX_SCALE, indexer_cache_scale_);
  }
  return tensors;
}

bool DeepSeekV4KVCacheImpl::empty() const { return !swa_cache_.defined(); }

std::vector<std::vector<int64_t>> DeepSeekV4KVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(3);
  if (key_cache_.defined()) {
    shapes.emplace_back(key_cache_.sizes().vec());
  }
  if (index_cache_.defined()) {
    shapes.emplace_back(index_cache_.sizes().vec());
  }
  if (swa_cache_.defined()) {
    shapes.emplace_back(swa_cache_.sizes().vec());
  }
  return shapes;
}

void DeepSeekV4KVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                                        torch::Tensor& dst_tensor) {
  key_cache_ = swap_tensor_blocks(key_cache_, src_tensor, dst_tensor);
  index_cache_ = swap_tensor_blocks(index_cache_, src_tensor, dst_tensor);
  indexer_cache_scale_ =
      swap_tensor_blocks(indexer_cache_scale_, src_tensor, dst_tensor);
  swa_cache_ = swap_tensor_blocks(swa_cache_, src_tensor, dst_tensor);
  compress_state_.swap_blocks(src_tensor, dst_tensor);
  index_state_.swap_blocks(src_tensor, dst_tensor);
}

DeepSeekV4KVCacheTensors create_dsv4_cache_tensors(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    int64_t layer_idx) {
  CHECK(kv_cache_shape.has_key_cache_shape())
      << "DeepSeek V4 cache shape must contain cache pool counts.";
  const std::vector<int64_t>& pool_counts = kv_cache_shape.key_cache_shape();
  CHECK_EQ(pool_counts.size(), 2u)
      << "DeepSeek V4/V4.1 cache shape must contain window and KV counts.";
  CHECK_GE(layer_idx, 0);
  CHECK_GT(create_options.window_size(), 0);
  const std::vector<int32_t>& ratios = create_options.compress_ratios();
  const int32_t ratio = layer_idx < static_cast<int64_t>(ratios.size())
                            ? ratios[static_cast<size_t>(layer_idx)]
                            : 1;
  const std::vector<int32_t>& sources = create_options.kv_source_layer_ids();
  Dsv4CacheLayoutOptions layout_options;
  layout_options.head_dim = create_options.head_dim();
  layout_options.index_head_dim = create_options.index_head_dim();
  layout_options.compress_ratio = ratio;
  layout_options.dtype = create_options.dtype();
  layout_options.is_v41 =
      util::is_deepseek_v41_model_type(create_options.model_type());
  layout_options.is_kv_source_layer =
      std::find(sources.begin(), sources.end(), layer_idx) != sources.end();
  const Dsv4LayerCacheLayout layout =
      build_dsv4_layer_cache_layout(layout_options);
  const int64_t kv_block_count = pool_counts[1];
  const std::shared_ptr<KVCacheTensorAllocator> allocator =
      create_options.tensor_allocator();
  auto allocate_raw = [&](const CacheTensorAllocationSpec& spec,
                          const std::vector<int64_t>& shape) {
#if defined(USE_NPU)
    if (create_options.enable_kv_cache_huge_page_allocator()) {
      return alloc_npu_huge_page_tensor(shape, spec.dtype, ACL_FORMAT_ND);
    }
#endif
    if (allocator != nullptr) {
      return cast_to_nd_format(allocator->allocate(
          spec.role, shape, spec.dtype, create_options.device()));
    }
    return cast_to_nd_format(torch::empty(
        shape, torch::dtype(spec.dtype).device(create_options.device())));
  };
  auto allocate_tensor = [&](const CacheTensorAllocationSpec& spec) {
    std::vector<int64_t> shape = spec.block_shape;
    shape.insert(
        shape.begin(),
        spec.block_type == BlockType::SWA ? pool_counts[0] : kv_block_count);
    torch::Tensor tensor = allocate_raw(spec, shape);
    // Empty/padded DP shards can read never-written slots. Preserve the
    // existing zero initialization for every owned plane, including states.
    tensor.zero_();
    return tensor;
  };

  DeepSeekV4KVCacheTensors tensors;
  for (const CacheTensorAllocationSpec& spec : layout.tensors) {
    torch::Tensor tensor = allocate_tensor(spec);
    switch (spec.role) {
      case KVCacheTensorRole::KEY:
        tensors.key_cache = std::move(tensor);
        break;
      case KVCacheTensorRole::INDEX:
        tensors.index_cache = std::move(tensor);
        break;
      case KVCacheTensorRole::INDEX_SCALE:
        tensors.indexer_cache_scale = std::move(tensor);
        break;
      case KVCacheTensorRole::SWA:
        tensors.swa_cache = std::move(tensor);
        break;
      case KVCacheTensorRole::KV_STATE:
        tensors.compress_kv_state = std::move(tensor);
        break;
      case KVCacheTensorRole::SCORE_STATE:
        tensors.compress_score_state = std::move(tensor);
        break;
      case KVCacheTensorRole::INDEX_KV_STATE:
        tensors.compress_index_kv_state = std::move(tensor);
        break;
      case KVCacheTensorRole::INDEX_SCORE_STATE:
        tensors.compress_index_score_state = std::move(tensor);
        break;
      case KVCacheTensorRole::COMPRESS_STATE: {
        tensors.compress_state = std::move(tensor);
        const int64_t width = tensors.compress_state.size(2) / 2;
        tensors.compress_kv_state =
            tensors.compress_state.narrow(/*dim=*/2, /*start=*/0, width);
        tensors.compress_score_state =
            tensors.compress_state.narrow(/*dim=*/2, width, width);
        break;
      }
      case KVCacheTensorRole::COMPRESS_INDEX_STATE: {
        tensors.compress_index_state = std::move(tensor);
        const int64_t width = tensors.compress_index_state.size(2) / 2;
        tensors.compress_index_kv_state =
            tensors.compress_index_state.narrow(/*dim=*/2, /*start=*/0, width);
        tensors.compress_index_score_state =
            tensors.compress_index_state.narrow(/*dim=*/2, width, width);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported DSV4 allocation role: "
                   << spec.role.to_string();
    }
  }
  return tensors;
}

std::string dsv4_shape_summary(const DeepSeekV4KVCacheTensors& tensors) {
  std::ostringstream summary;
  bool first = true;
  auto append = [&](const char* name, const torch::Tensor& tensor) {
    if (!tensor.defined()) {
      return;
    }
    if (!first) {
      summary << ", ";
    }
    summary << name << "=" << tensor_shape_string(tensor);
    first = false;
  };
  append("key_cache", tensors.key_cache);
  append("index_cache", tensors.index_cache);
  append("indexer_cache_scale", tensors.indexer_cache_scale);
  append("swa_cache", tensors.swa_cache);
  append("compress_kv_state", tensors.compress_kv_state);
  append("compress_score_state", tensors.compress_score_state);
  append("compress_index_kv_state", tensors.compress_index_kv_state);
  append("compress_index_score_state", tensors.compress_index_score_state);
  return summary.str();
}

}  // namespace xllm
