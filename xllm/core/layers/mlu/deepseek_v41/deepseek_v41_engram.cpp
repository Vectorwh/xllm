/* Copyright 2026 The xLLM Authors.

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

#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"

#include <ATen/Parallel.h>
#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <framework/core/stream_guard.h>
#include <framework/graphs/MLUGraphUtils.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "framework/config/model_config.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/mlu/bangc/include/engram_dequantize.h"
#include "kernels/mlu/bangc/include/engram_gate.h"
#include "kernels/mlu/bangc/include/engram_gather_dequantize.h"
#include "kernels/mlu/bangc/include/engram_hash.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"
#include "layers/mlu/deepseek_v41/engram_shared_memory.h"
#include "util/json_reader.h"

#ifndef _OPENMP
#error "Engram CPU gather requires OpenMP to enable ATen parallel_for"
#endif

namespace xllm::layer {
namespace {
// Match ENGRAM_IMAGE_SENTINEL_BASE_ID / ENGRAM_IMAGE_PAD_ID in the Python
// Engram hash contract. These IDs remain fixed independently of image_token_id.
constexpr int64_t kEngramImageSentinelBaseId = 129264;
constexpr int64_t kEngramImagePadId = 129265;

void configure_engram_cpu_threads(int32_t threads) {
  static thread_local int32_t configured_threads = 0;
  if (configured_threads != threads) {
    torch::set_num_threads(threads);
    configured_threads = threads;
  }
}

struct EngramOptions {
  bool cpu_offload = true;
  bool node_shared_memory = false;
  bool cp_reduce_scatter = true;
  bool prefill_device_hash = true;
};

std::string engram_option(const nlohmann::json& config,
                          const std::string& key,
                          const std::string& default_value,
                          const std::vector<std::string>& choices) {
  const std::string value = config.value(key, default_value);
  if (std::find(choices.begin(), choices.end(), value) == choices.end()) {
    std::string expected;
    for (const auto& choice : choices) {
      expected += expected.empty() ? choice : " or " + choice;
    }
    LOG(FATAL) << "Engram " << key << " must be " << expected;
  }
  return value;
}

// Parses and validates engram_config once per layer before any state is
// allocated, so a misconfiguration fails at construction.
EngramOptions parse_engram_options(const std::string& text) {
  JsonReader reader;
  CHECK(reader.parse_text(text)) << "Invalid Engram JSON configuration";
  const auto config = reader.data();
  CHECK(config.is_object()) << "engram_config must be a JSON object";
  for (const auto& [key, value] : config.items()) {
    CHECK(key == "placement" || key == "replication" ||
          key == "communication" || key == "prefill_hash")
        << "Unknown Engram configuration key: " << key;
    CHECK(value.is_string())
        << "Engram configuration " << key << " must be a string";
  }
  EngramOptions options;
  options.cpu_offload =
      engram_option(config, "placement", "cpu", {"cpu", "device"}) == "cpu";
  options.node_shared_memory =
      engram_option(config, "replication", "none", {"none", "per_node"}) ==
      "per_node";
  options.cp_reduce_scatter =
      engram_option(config,
                    "communication",
                    "reduce_scatter_allgather",
                    {"allreduce", "reduce_scatter_allgather"}) ==
      "reduce_scatter_allgather";
  options.prefill_device_hash =
      engram_option(config, "prefill_hash", "device", {"cpu", "device"}) ==
      "device";
  CHECK(!options.node_shared_memory || options.cpu_offload)
      << "Engram replication=per_node requires placement=cpu";
  return options;
}

// Collects the query lengths of the first `sequences` requests and returns
// their total row count.
int64_t collect_query_lengths(const ModelInputParams& input,
                              int32_t sequences,
                              std::vector<int32_t>* lengths) {
  CHECK_GE(sequences, 0);
  if (lengths != nullptr) {
    lengths->clear();
    lengths->reserve(sequences + 1);
  }
  int64_t rows = 0;
  for (int32_t seq = 0; seq < sequences; ++seq) {
    const int32_t length = input.get_q_seq_len(seq);
    CHECK_GE(length, 0);
    if (lengths != nullptr) {
      lengths->emplace_back(length);
    }
    rows += length;
  }
  return rows;
}

// Writes the per-request lookback tokens, marking unused slots as blocked.
void copy_lookback_history(const std::vector<int32_t>& history,
                           torch::Tensor& output) {
  CHECK_LE(static_cast<int64_t>(history.size()), output.numel());
  output.fill_(-1);
  std::copy(history.begin(), history.end(), output.data_ptr<int64_t>());
}
}  // namespace

torch::Tensor valid_engram_host_tokens(const torch::Tensor& tokens,
                                       const ModelInputParams& input) {
  const auto& host = input.embedding.input_token_ids_host;
  if (!host.defined() || has_device_rewritten_tokens(input)) {
    return {};
  }
  if (!host.device().is_cpu() || !host.is_contiguous() ||
      host.scalar_type() != torch::kInt32) {
    // Upstream builders provide contiguous CPU int32 ids. Any other layout
    // costs a synchronous device readback on every step, so make it visible.
    LOG_FIRST_N(WARNING, 1)
        << "Engram ignores the host token snapshot: expected contiguous CPU "
        << "int32, got " << host.scalar_type() << " on " << host.device();
    return {};
  }
  const int64_t rows = input.enable_graph ? collect_query_lengths(
                                                input,
                                                input.meta.actual_num_sequences,
                                                /*lengths=*/nullptr)
                                          : tokens.numel();
  if (host.numel() != rows) {
    return {};
  }
  // Negative overlap placeholders are replaced on the device. Graph padding
  // is separate from the live host rows validated above.
  const int32_t* ids = host.data_ptr<int32_t>();
  return std::all_of(ids, ids + rows, [](int32_t id) { return id >= 0; })
             ? host
             : torch::Tensor();
}

EngramCpLayout build_engram_cp_layout(
    const mlu_v4_cp::DeepseekV4CpContext& context,
    int32_t tp_size) {
  CHECK_GT(tp_size, 0);
  const auto& geometry = context.geometry;
  CHECK_GT(geometry.total_tokens, 0);
  CHECK_GT(geometry.rows_by_rank.size(), 1);
  CHECK_GE(context.cp_rank, 0);
  CHECK_LT(context.cp_rank, geometry.rows_by_rank.size());
  int64_t max_rows = 0;
  for (const auto& rows : geometry.rows_by_rank) {
    max_rows = std::max(max_rows, static_cast<int64_t>(rows.size()));
  }
  EngramCpLayout layout;
  layout.rows_per_rank = (max_rows + tp_size - 1) / tp_size;
  layout.local_rows = geometry.rows_by_rank[context.cp_rank].size();
  layout.cp_rank = context.cp_rank;
  layout.cp_size = static_cast<int32_t>(geometry.rows_by_rank.size());
  layout.tp_size = tp_size;
  const int64_t padded_cp_rows = layout.rows_per_rank * tp_size;
  // Padding uses an existing row; these extra results are removed after the
  // TP gather and never reach WKV. No extra zero tensor or mask is required.
  std::vector<int64_t> indices;
  indices.reserve(layout.cp_size * padded_cp_rows);
  for (const auto& rows : geometry.rows_by_rank) {
    for (int64_t row : rows) {
      CHECK_GE(row, 0);
      CHECK_LT(row, geometry.total_tokens)
          << "Engram CP communication does not accept dummy token rows";
      indices.emplace_back(row);
    }
    indices.insert(indices.end(), padded_cp_rows - rows.size(), 0);
  }
  layout.pack_indices = torch::tensor(
      indices, context.local_row_indices.options().dtype(torch::kInt64));
  return layout;
}

EngramDeviceHasher::Input::Input(int64_t rows,
                                 int64_t depth,
                                 const torch::TensorOptions& options)
    : rows_(rows), depth_(depth), options_(options) {
  CHECK_GE(rows, 0);
  CHECK_LE(rows, std::numeric_limits<int32_t>::max());
  CHECK_GT(depth, 0);
}

EngramDeviceHasher::Input::~Input() {
  if (uploaded_.isCreated()) {
    uploaded_.synchronize();
  }
}

void EngramDeviceHasher::Input::prepare(const ModelInputParams& input,
                                        bool graph) {
  if (uploaded_.isCreated()) {
    uploaded_.synchronize();
  }
  const int32_t sequences = graph || input.enable_graph
                                ? input.meta.actual_num_sequences
                                : input.meta.num_sequences;
  CHECK_GE(sequences, 0);
  const int64_t capacity =
      static_cast<int64_t>(std::max(input.meta.num_sequences, sequences)) + 1;
  if (!prepared() || sequences > host_history_.size(0)) {
    CHECK(!prepared() || (!graph && !input.enable_graph))
        << "Engram graph request count exceeds its metadata capacity";
    const auto host = torch::TensorOptions().pinned_memory(true);
    host_starts_ = torch::empty({capacity + 1}, host.dtype(torch::kInt32));
    host_history_ = torch::empty({capacity, depth_}, host.dtype(torch::kInt64));
    starts_ = torch::empty(host_starts_.sizes(), options_.dtype(torch::kInt32));
    history_ =
        torch::empty(host_history_.sizes(), options_.dtype(torch::kInt64));
  }
  const auto& history = input.embedding.engram_lookback_token_ids;
  CHECK_EQ(history.size(), sequences * depth_);
  std::vector<int32_t> lengths;
  CHECK_LE(collect_query_lengths(input, sequences, &lengths), rows_)
      << "Engram sequence lengths exceed the hash input capacity";
  host_starts_.fill_(rows_);
  int32_t* starts = host_starts_.data_ptr<int32_t>();
  starts[0] = 0;
  for (int32_t seq = 0; seq < sequences; ++seq) {
    starts[seq + 1] = starts[seq] + lengths[seq];
  }
  copy_lookback_history(history, host_history_);
  starts_.copy_(host_starts_, /*non_blocking=*/true);
  history_.copy_(host_history_, /*non_blocking=*/true);
  uploaded_.place(torch_mlu::getCurrentMLUStream(history_.device().index()));
}

EngramDeviceHasher::EngramDeviceHasher(const StateDict& state,
                                       int64_t pad_token,
                                       const torch::TensorOptions& options)
    : pad_token_(pad_token), options_(options) {
  auto load = [&](const std::string& name) {
    const auto tensor = state.get_tensor(name);
    CHECK(tensor.defined()) << "Missing Engram " << name;
    return tensor.to(options.device(), torch::kInt64).contiguous();
  };
  token_map_ = load("token_map");
  multipliers_ = load("multipliers");
  CHECK_EQ(multipliers_.dim(), 2);
  const int64_t layers = multipliers_.size(0);
  primes_ = load("primes").view({layers, -1});
  offsets_ = load("offsets").view({layers, -1});
}

std::unique_ptr<EngramDeviceHasher::Input> EngramDeviceHasher::create_input(
    int64_t rows) const {
  return std::make_unique<Input>(rows, multipliers_.size(1) - 1, options_);
}

torch::Tensor EngramDeviceHasher::forward(const torch::Tensor& tokens,
                                          const Input& input) const {
  return launch(tokens, input, multipliers_, primes_, offsets_);
}

torch::Tensor EngramDeviceHasher::forward_layer(const torch::Tensor& tokens,
                                                const Input& input,
                                                int32_t layer) const {
  CHECK_GE(layer, 0);
  CHECK_LT(layer, multipliers_.size(0));
  // A single leading row of a contiguous matrix is itself contiguous.
  return launch(tokens,
                input,
                multipliers_.narrow(/*dim=*/0, layer, /*length=*/1),
                primes_.narrow(/*dim=*/0, layer, /*length=*/1),
                offsets_.narrow(/*dim=*/0, layer, /*length=*/1))
      .select(/*dim=*/0, /*index=*/0);
}

torch::Tensor EngramDeviceHasher::launch(const torch::Tensor& tokens,
                                         const Input& input,
                                         const torch::Tensor& multipliers,
                                         const torch::Tensor& primes,
                                         const torch::Tensor& offsets) const {
  CHECK(input.prepared()) << "Engram hash input must be prepared before use";
  CHECK_EQ(tokens.numel(), input.capacity());
  return kernel::mlu::bangc::engram_hash(tokens,
                                         input.starts(),
                                         input.history(),
                                         token_map_,
                                         multipliers,
                                         primes,
                                         offsets,
                                         pad_token_);
}

DeepseekV41EngramImpl::DeepseekV41EngramImpl(const ModelContext& context,
                                             int32_t engram_id)
    : args_(context.get_model_args()),
      engram_id_(engram_id),
      lookup_group_(context.get_parallel_args().tp_group_),
      world_group_(context.get_parallel_args().process_group_),
      tp_group_(context.get_parallel_args().tp_group_),
      options_(context.get_tensor_options()) {
  CHECK_GT(args_.vocab_size(), 0) << "Engram vocab_size must be positive";
  CHECK_GE(args_.engram_pad_token_id(), 0)
      << "engram_pad_token_id must be in [0, vocab_size)";
  CHECK_LT(args_.engram_pad_token_id(), args_.vocab_size())
      << "engram_pad_token_id must be in [0, vocab_size)";
  CHECK_EQ(args_.engram_layer_ids().size(),
           args_.engram_num_embeddings().size())
      << "engram_layer_ids and engram_num_embeddings must have the same length";
  CHECK_GE(engram_id_, 0) << "Invalid Engram layer index";
  CHECK_LT(engram_id_, args_.engram_num_embeddings().size())
      << "Invalid Engram layer index";
  CHECK_GE(args_.engram_max_ngram_size(), 2);
  CHECK_GT(args_.engram_n_heads(), 0);
  CHECK_GT(args_.engram_compressed_vocab_size(), 0);
  CHECK_GT(args_.engram_num_embeddings()[engram_id_], 0);
  wkv_module_ = "layers." +
                std::to_string(args_.engram_layer_ids().at(engram_id_)) +
                ".engram.wkv";
  wkv_quant_ = context.get_quant_args().for_module(wkv_module_);
  const EngramOptions engram_options =
      parse_engram_options(ModelConfig::get_instance().engram_config());
  cpu_offload_ = engram_options.cpu_offload;
  node_shared_memory_ = engram_options.node_shared_memory;
  cp_reduce_scatter_ = engram_options.cp_reduce_scatter;
  prefill_device_hash_ = engram_options.prefill_device_hash;
  // Shared tables exchange hash heads only within a TP query group. The
  // row-sharded CP path reduces the global token pack across the EP group.
  if (!node_shared_memory_ && context.get_parallel_args().cp_size() > 1) {
    lookup_group_ = context.get_parallel_args().moe_ep_group_;
  }
  CHECK(!node_shared_memory_ || world_group_ != nullptr)
      << "Engram per_node sharing requires the model world process group";
  const int64_t rows = args_.engram_num_embeddings().at(engram_id_);
  table_start_ = node_shared_memory_ ? 0
                                     : rows * lookup_group_->rank() /
                                           lookup_group_->world_size();
  table_end_ = node_shared_memory_ ? rows
                                   : rows * (lookup_group_->rank() + 1) /
                                         lookup_group_->world_size();
  LOG(INFO) << "Engram " << engram_id_ << " rank " << lookup_group_->rank()
            << ": table rows [" << table_start_ << ", " << table_end_
            << ") stored on " << (cpu_offload_ ? "host" : "MLU")
            << (node_shared_memory_ ? " (shared across same-host model ranks)"
                                    : "");
  // Decode raw E4M3 bytes by lookup; no FP8 arithmetic is needed on MLU.
  torch::Tensor bytes = torch::arange(256, torch::kInt16).to(torch::kUInt8);
  fp8_values_ = bytes.view(torch::kFloat8_e4m3fn)
                    .to(torch::kFloat32)
                    .to(options_.device());
}

void DeepseekV41EngramImpl::load_wkv(const StateDict& state) {
  if (state.size() == 0) {
    return;
  }
  const int64_t in = (args_.engram_max_ngram_size() - 1) *
                     args_.engram_n_heads() * args_.engram_head_dim();
  const int64_t out = (args_.hc_mult() + 1) * args_.hidden_size();
  for (const auto& [name, tensor] : state) {
    if (name != "weight" && name != "weight_scale" && name != "smooth") {
      continue;
    }
    CHECK(!wkv_ && pending_wkv_.emplace(name, tensor).second)
        << "Unexpected additional tensor " << wkv_module_ << "." << name;
    if (name == "weight") {
      CHECK_EQ(tensor.sizes(), torch::IntArrayRef({out, in}))
          << wkv_module_ << ".weight";
      CHECK(tensor.scalar_type() == torch::kBFloat16 ||
            tensor.scalar_type() == torch::kInt8)
          << wkv_module_ << ".weight";
    } else {
      CHECK_EQ(tensor.scalar_type(), torch::kBFloat16)
          << wkv_module_ << "." << name;
      if (name == "smooth") {
        CHECK_EQ(tensor.sizes(), torch::IntArrayRef({in}))
            << wkv_module_ << ".smooth";
      } else {
        CHECK_EQ(tensor.sizes(), torch::IntArrayRef({out, 1}))
            << wkv_module_ << ".weight_scale";
      }
    }
  }
  const auto weight = pending_wkv_.find("weight");
  if (weight == pending_wkv_.end()) {
    return;
  }
  const bool integer = weight->second.scalar_type() == torch::kInt8;
  if (integer) {
    CHECK(wkv_quant_.is_compressed_tensors_w8a8_dynamic() &&
          wkv_quant_.bits() == 8 && wkv_quant_.group_size() == 0 &&
          wkv_quant_.preserve_smooth())
        << wkv_module_;
    if (!pending_wkv_.contains("weight_scale") ||
        !pending_wkv_.contains("smooth")) {
      return;
    }
  } else {
    CHECK_EQ(pending_wkv_.size(), 1)
        << "Unexpected quantization tensors for BF16 " << wkv_module_;
  }
  wkv_ = register_module(
      "wkv",
      ReplicatedLinear(
          in, out, false, integer ? wkv_quant_ : QuantArgs(), options_));
  wkv_->load_state_dict(StateDict(std::move(pending_wkv_), wkv_module_ + "."));
}

void DeepseekV41EngramImpl::load_state_dict(const StateDict& state) {
  load_wkv(state.get_dict_with_prefix("wkv."));
  for (const std::string name :
       {"q_weight", "k_weight", "embed.weight", "embed.scale"}) {
    torch::Tensor value = state.get_tensor(name);
    if (!value.defined()) {
      continue;
    }
    if (name == "embed.weight" || name == "embed.scale") {
      CHECK_EQ(value.dim(), 2) << "Invalid Engram " << name << " shape";
      CHECK_EQ(value.size(0), args_.engram_num_embeddings()[engram_id_])
          << "Invalid Engram " << name << " rows";
      CHECK_EQ(value.element_size(), 1) << "Engram tables must use FP8 bytes";
      CHECK_GT(value.size(1), 0) << "Empty Engram " << name << " row";
      if (name == "embed.weight") {
        CHECK_EQ(value.size(1), args_.engram_head_dim());
      } else {
        CHECK_EQ(args_.engram_head_dim() % value.size(1), 0);
      }
      if (node_shared_memory_) {
        weights_[name] = load_shared_engram_tensor(
            value,
            "layer_" + std::to_string(args_.engram_layer_ids()[engram_id_]) +
                "." + name,
            *world_group_);
        continue;
      }
      value = value.slice(0, table_start_, table_end_)
                  .contiguous()
                  .view(torch::kUInt8);
      if (cpu_offload_) {
        // Own the host shard independently of the checkpoint mmap lifetime.
        weights_[name] = value.cpu().contiguous().clone();
        continue;
      }
    }
    weights_[name] = value.to(options_.device()).contiguous();
  }
}

void DeepseekV41EngramImpl::lookup_host_rows(const torch::Tensor& hashes,
                                             torch::Tensor& weight_rows,
                                             torch::Tensor& scale_rows) const {
  const auto& weight = weights_.at("embed.weight");
  const auto& scale = weights_.at("embed.scale");
  CHECK(hashes.device().is_cpu() && hashes.is_contiguous());
  CHECK_EQ(hashes.scalar_type(), torch::kInt64);
  for (const auto& tensor : {weight_rows, scale_rows}) {
    CHECK(tensor.device().is_cpu());
    CHECK_EQ(tensor.stride(1), 1);
    CHECK_EQ(tensor.scalar_type(), torch::kUInt8);
    CHECK_EQ(tensor.dim(), 2);
  }
  const int64_t local_cols = local_hash_columns(hashes.size(0));
  CHECK_EQ(weight_rows.size(0), hashes.size(0) * local_cols);
  CHECK_EQ(scale_rows.size(0), weight_rows.size(0));
  CHECK_EQ(weight_rows.size(1), weight.size(1));
  CHECK_EQ(scale_rows.size(1), scale.size(1));
  const int64_t dim = weight.size(1);
  const int64_t scale_dim = scale.size(1);
  const int64_t* ids = hashes.data_ptr<int64_t>();
  const uint8_t* table = weight.data_ptr<uint8_t>();
  const uint8_t* scales = scale.data_ptr<uint8_t>();
  uint8_t* rows = weight_rows.data_ptr<uint8_t>();
  uint8_t* row_scales = scale_rows.data_ptr<uint8_t>();
  const int64_t cols = hashes.size(1);
  const int64_t head_start =
      shard_hash_heads(hashes.size(0)) ? lookup_group_->rank() * local_cols : 0;
  const int64_t weight_stride = weight_rows.stride(0);
  const int64_t scale_stride = scale_rows.stride(0);
  auto gather = [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t col = head_start + index % local_cols;
      const int64_t row =
          col < cols ? ids[index / local_cols * cols + col] - table_start_ : -1;
      if (row < 0 || row >= table_end_ - table_start_) {
        std::memset(rows + index * weight_stride, 0, dim);
        std::memset(row_scales + index * scale_stride, 0, scale_dim);
        continue;
      }
      std::memcpy(rows + index * weight_stride, table + row * dim, dim);
      std::memcpy(row_scales + index * scale_stride,
                  scales + row * scale_dim,
                  scale_dim);
    }
  };
  // Keep small decode lookups on their callback thread. Large prefill gathers
  // use the caller's bounded intra-op pool instead of serial random reads.
  constexpr int64_t kGatherGrain = 1024;
  torch::parallel_for(0, weight_rows.size(0), kGatherGrain, gather);
}

void DeepseekV41EngramImpl::verify_loaded_weights() const {
  if (!wkv_) {
    for (const std::string key : {"weight", "weight_scale", "smooth"}) {
      CHECK(pending_wkv_.contains(key))
          << "Missing " << wkv_module_ << "." << key;
    }
  }
  CHECK(wkv_ && wkv_->is_weight_loaded()) << wkv_module_;
  CHECK(token_map_.defined()) << "Missing Engram hash state";
  for (const std::string name :
       {"q_weight", "k_weight", "embed.weight", "embed.scale"}) {
    CHECK(weights_.count(name)) << "Missing Engram weight " << name;
  }
}

void DeepseekV41EngramImpl::load_hash_state(
    const StateDict& state,
    std::shared_ptr<const EngramDeviceHasher>* shared_hasher) {
  auto map = state.get_tensor("token_map");
  CHECK(map.defined()) << "Missing Engram token_map";
  CHECK(map.scalar_type() == torch::kInt32 ||
        map.scalar_type() == torch::kInt64);
  CHECK_EQ(map.dim(), 1);
  CHECK_EQ(map.numel(), args_.vocab_size())
      << "Engram token_map must cover vocab_size";
  token_map_ = map.to(torch::kCPU, torch::kInt64).contiguous().clone();
  CHECK_GE(token_map_.min().item<int64_t>(), 0)
      << "Engram token_map contains a negative compressed id";
  CHECK_LT(token_map_.max().item<int64_t>(),
           args_.engram_compressed_vocab_size())
      << "Engram token_map compressed id exceeds compressed_vocab_size";
  const int64_t cols = hash_columns();
  auto load_layer = [&](const std::string& name, int64_t elements) {
    const auto tensor = state.get_tensor(name);
    CHECK(tensor.defined()) << "Missing Engram " << name;
    CHECK_EQ(tensor.scalar_type(), torch::kInt64);
    CHECK_GE(tensor.dim(), 2);
    CHECK_EQ(tensor.size(0), args_.engram_layer_ids().size());
    CHECK_EQ(tensor[engram_id_].numel(), elements)
        << "Invalid Engram " << name << " shape";
    return tensor[engram_id_].to(torch::kCPU).flatten().contiguous().clone();
  };
  multipliers_ = load_layer("multipliers", args_.engram_max_ngram_size());
  primes_ = load_layer("primes", cols);
  offsets_ = load_layer("offsets", cols);
  CHECK_GT(multipliers_.min().item<int64_t>(), 0);
  // Training bounds each positive multiplier so every mapped-id product fits
  // int64. XOR of those nonnegative products also leaves the sign bit clear.
  CHECK_LE(multipliers_.max().item<int64_t>(),
           std::numeric_limits<int64_t>::max() /
               std::max<int64_t>(1, args_.engram_compressed_vocab_size() - 1))
      << "Engram multiplier can overflow a compressed-id product";
  CHECK_GT(primes_.min().item<int64_t>(), 0);
  CHECK_GE(offsets_.min().item<int64_t>(), 0);
  const int64_t rows = args_.engram_num_embeddings()[engram_id_];
  CHECK_LE(primes_.max().item<int64_t>(), rows);
  CHECK((offsets_ <= rows - primes_).all().item<bool>())
      << "Engram hash buckets exceed embedding rows";
  if (!uses_device_hasher()) {
    return;
  }
  // Created only after this layer validated the hash parameters above.
  if (shared_hasher != nullptr && *shared_hasher != nullptr) {
    device_hasher_ = *shared_hasher;
    return;
  }
  device_hasher_ = std::make_shared<const EngramDeviceHasher>(
      state, args_.engram_pad_token_id(), options_);
  if (shared_hasher != nullptr) {
    *shared_hasher = device_hasher_;
  }
}

torch::Tensor DeepseekV41EngramImpl::hash_on_device(
    const torch::Tensor& tokens,
    const EngramDeviceHasher::Input& input) const {
  return device_hasher_->forward_layer(tokens, input, engram_id_);
}

bool DeepseekV41EngramImpl::prefill_hashes_on_device(
    const ModelInputParams& input) const {
  return cpu_offload_ && prefill_device_hash_ &&
         input.meta.batch_forward_type.no_decode() && !input.enable_graph &&
         !has_device_rewritten_tokens(input);
}

void DeepseekV41EngramImpl::compute_hashes(const int64_t* tokens,
                                           const int64_t* history,
                                           const std::vector<int32_t>& lengths,
                                           int64_t* output) const {
  const int64_t heads = args_.engram_n_heads();
  const int64_t depth = args_.engram_max_ngram_size() - 1;
  const int64_t cols = hash_columns();
  const int64_t* map = token_map_.data_ptr<int64_t>();
  const int64_t* mul = multipliers_.data_ptr<int64_t>();
  const int64_t* prime = primes_.data_ptr<int64_t>();
  const int64_t* offset = offsets_.data_ptr<int64_t>();
  int64_t start = 0;
  for (size_t seq = 0; seq < lengths.size(); ++seq) {
    for (int64_t row = 0; row < lengths[seq]; ++row) {
      int64_t rolling = 0;
      bool blocked = false;
      for (int64_t shift = 0; shift <= depth; ++shift) {
        const int64_t relative = row - shift;
        const int64_t token = relative >= 0
                                  ? tokens[start + relative]
                                  : history[seq * depth - relative - 1];
        blocked = blocked || token < 0 || token == kEngramImageSentinelBaseId ||
                  token == kEngramImagePadId;
        const int64_t value =
            blocked ? map[args_.engram_pad_token_id()]
                    : map[std::min<int64_t>(token, token_map_.numel() - 1)];
        rolling ^= value * mul[shift];
        if (shift == 0) {
          continue;
        }
        for (int64_t head = 0; head < heads; ++head) {
          const int64_t col = (shift - 1) * heads + head;
          output[(start + row) * cols + col] =
              rolling % prime[col] + offset[col];
        }
      }
    }
    start += lengths[seq];
  }
}

int64_t DeepseekV41EngramImpl::hash_columns() const {
  return (args_.engram_max_ngram_size() - 1) * args_.engram_n_heads();
}

int64_t DeepseekV41EngramImpl::shard_hash_columns() const {
  const int64_t shards = lookup_group_->world_size();
  return (hash_columns() + shards - 1) / shards;
}

int64_t DeepseekV41EngramImpl::scale_dim() const {
  const auto it = weights_.find("embed.scale");
  CHECK(it != weights_.end())
      << "Engram embed.scale must be loaded before host-table staging";
  return it->second.size(1);
}

bool DeepseekV41EngramImpl::shard_hash_heads(int64_t tokens) const {
  if (!node_shared_memory_ || lookup_group_->world_size() == 1) {
    return false;
  }
  const int64_t cols = hash_columns();
  const int64_t shard_cols = shard_hash_columns();
  const int64_t row_bytes = args_.engram_head_dim() + scale_dim();
  // Tiny transfers do not amortize a packed collective and layout conversion.
  // The threshold uses bytes saved by this TP group, including head padding,
  // rather than a model-specific token count or speculative-decode width.
  constexpr int64_t kMinSavedHostBytes = 512 * 1024;
  return tokens * (cols - shard_cols) * row_bytes >= kMinSavedHostBytes;
}

int64_t DeepseekV41EngramImpl::local_hash_columns(int64_t tokens) const {
  return shard_hash_heads(tokens) ? shard_hash_columns() : hash_columns();
}

DeepseekV41EngramImpl::HashState::HashState(const DeepseekV41EngramImpl& owner,
                                            int64_t rows,
                                            int64_t query_width)
    : owner_(owner),
      stream_(
          torch_mlu::getStreamFromPool(false, owner.options_.device().index())),
      rows_ready_(CNRT_NOTIFIER_DISABLE_TIMING_ALL | CNRT_NOTIFIER_EXTERNAL),
      consumed_(CNRT_NOTIFIER_DISABLE_TIMING_ALL | CNRT_NOTIFIER_EXTERNAL) {
  CHECK_GT(query_width, 0);
  CHECK_EQ(rows % query_width, 0);
  lengths_.assign(rows / query_width, static_cast<int32_t>(query_width));
  CHECK(!owner.uses_device_hasher() || owner.device_hasher_ != nullptr)
      << "Engram hash state must be loaded before creating staging";
  if (!owner.cpu_offload_) {
    device_input_ = owner.device_hasher_->create_input(rows);
  }
  const int64_t depth = owner.args_.engram_max_ngram_size() - 1;
  auto host = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
  host_tokens_ = torch::empty({rows}, host);
  // One extra sequence holds graph padding even when all capacity slots are
  // real sequences and one or more of those sequences have zero query rows.
  host_history_ = torch::empty({rows + 1, depth}, host);
  prepared_history_ = torch::empty(host_history_.sizes(), host);
  host_hashes_ = torch::empty({rows, owner.hash_columns()}, host);
  history_ =
      torch::empty(host_history_.sizes(), owner.options_.dtype(torch::kInt64));
  hashes_ =
      torch::empty(host_hashes_.sizes(), owner.options_.dtype(torch::kInt64));
  if (owner.cpu_offload_) {
    const int64_t dim = owner.args_.engram_head_dim();
    const int64_t scale_dim = owner.scale_dim();
    host_packed_rows_ =
        torch::empty({rows * owner.local_hash_columns(rows), dim + scale_dim},
                     host.dtype(torch::kUInt8));
    host_weight_rows_ = host_packed_rows_.narrow(1, 0, dim);
    host_scale_rows_ = host_packed_rows_.narrow(1, dim, scale_dim);
    packed_rows_ = torch::empty(host_packed_rows_.sizes(),
                                owner.options_.dtype(torch::kUInt8));
    // The allocator may reuse buffers whose preceding writes are still queued
    // on the allocation stream. The side stream must not upload into those
    // buffers until that work completes, even when inputs come from a host
    // snapshot and no device-token dependency is needed.
    inputs_ready_.place(
        torch_mlu::getCurrentMLUStream(owner_.options_.device().index()));
    inputs_ready_.wait(stream_);
  }
}

DeepseekV41EngramImpl::HashState::~HashState() {
  stream_.synchronize();
  if (history_ready_.isCreated()) {
    history_ready_.synchronize();
  }
  if (rows_ready_.isCreated()) {
    rows_ready_.synchronize();
  }
  if (consumed_.isCreated()) {
    consumed_.synchronize();
  }
}

void DeepseekV41EngramImpl::HashState::prepare(const ModelInputParams& input,
                                               bool device_history) {
  if (device_input_) {
    device_input_->prepare(input, /*graph=*/true);
    return;
  }
  // This buffer is a DMA source. Wait only for its last upload, not the model.
  if (rows_ready_.isCreated()) {
    rows_ready_.synchronize();
  }
  if (history_ready_.isCreated()) {
    history_ready_.synchronize();
  }
  if (!prefetched_ && consumed_.isCreated()) {
    // Inline captured callbacks read lengths_ directly. Their previous graph
    // invocation must finish before the host changes this sequence layout.
    consumed_.synchronize();
  }
  const auto& history = input.embedding.engram_lookback_token_ids;
  CHECK_EQ(history.size(), input.meta.actual_num_sequences * history_.size(1));
  CHECK_LE(input.meta.actual_num_sequences, history_.size(0));
  const int64_t active_rows =
      collect_query_lengths(input, input.meta.actual_num_sequences, &lengths_);
  CHECK_LE(active_rows, host_tokens_.numel())
      << "Engram sequence lengths exceed graph token capacity";
  if (active_rows < host_tokens_.numel()) {
    CHECK_LT(lengths_.size(), history_.size(0))
        << "Engram graph requires a history slot for padded token rows";
    lengths_.emplace_back(host_tokens_.numel() - active_rows);
  }
  copy_lookback_history(history, prepared_history_);
  if (device_history) {
    history_.copy_(prepared_history_, /*non_blocking=*/true);
    history_ready_.place(
        torch_mlu::getCurrentMLUStream(owner_.options_.device().index()));
  }
  cpu_threads_ = torch::get_num_threads();
  // A prior eager prepare is not the input for this graph invocation. Inline
  // capture must record its CPU callback unless prefetch supplies fresh rows.
  prefetched_ = false;
  prefetch_required_ = !device_history;
}

void DeepseekV41EngramImpl::HashState::compute(void* state) noexcept {
  auto& hash = *static_cast<HashState*>(state);
  configure_engram_cpu_threads(hash.cpu_threads_);
  torch::NoGradGuard guard;
  hash.owner_.compute_hashes(hash.host_tokens_.data_ptr<int64_t>(),
                             hash.host_history_.data_ptr<int64_t>(),
                             hash.lengths_,
                             hash.host_hashes_.data_ptr<int64_t>());
  if (hash.owner_.cpu_offload_) {
    hash.owner_.lookup_host_rows(
        hash.host_hashes_, hash.host_weight_rows_, hash.host_scale_rows_);
  }
}

void DeepseekV41EngramImpl::HashState::lookup(void* state) noexcept {
  auto& hash = *static_cast<HashState*>(state);
  configure_engram_cpu_threads(hash.cpu_threads_);
  torch::NoGradGuard guard;
  hash.owner_.lookup_host_rows(
      hash.host_hashes_, hash.host_weight_rows_, hash.host_scale_rows_);
}

void DeepseekV41EngramImpl::HashState::upload_host_outputs(
    bool include_hashes) {
  if (include_hashes) {
    hashes_.copy_(host_hashes_, /*non_blocking=*/true);
  }
  if (owner_.cpu_offload_) {
    packed_rows_.copy_(host_packed_rows_, /*non_blocking=*/true);
  }
}

void DeepseekV41EngramImpl::HashState::publish_prefetch() {
  rows_ready_.place(stream_);
  prefetched_ = true;
}

void DeepseekV41EngramImpl::HashState::prefetch(
    const torch::Tensor& tokens,
    const torch::Tensor& host_tokens) {
  if (device_input_) {
    return;
  }
  // This method runs outside capture. Reuse the host history directly instead
  // of sending it to the device and back through the callback.
  CHECK_EQ(tokens.numel(), host_tokens_.numel());
  host_history_.copy_(prepared_history_);
  if (!host_tokens.defined()) {
    inputs_ready_.place(
        torch_mlu::getCurrentMLUStream(owner_.options_.device().index()));
    inputs_ready_.wait(stream_);
  }
  torch_mlu::mlu::MLUStreamGuard guard(stream_);
  if (host_tokens.defined()) {
    CHECK(host_tokens.device().is_cpu());
    CHECK_LE(host_tokens.numel(), host_tokens_.numel());
    host_tokens_.fill_(-1);
    host_tokens_.narrow(0, 0, host_tokens.numel()).copy_(host_tokens);
  } else {
    host_tokens_.copy_(tokens, /*non_blocking=*/true);
  }
  CHECK_EQ(cnrtInvokeHostFunc(stream_.stream(), compute, this), cnrtSuccess);
  // CPU work needs only the owned host snapshot. Fence the previous device
  // consumers immediately before overwriting their staging buffers.
  if (consumed_.isCreated()) {
    consumed_.wait(stream_);
  }
  upload_host_outputs(/*include_hashes=*/true);
  publish_prefetch();
  prefetch_required_ = false;
}

void DeepseekV41EngramImpl::HashState::prepare_eager(
    const torch::Tensor& tokens,
    const ModelInputParams& input,
    torch::Tensor device_tokens) {
  if (device_input_) {
    device_input_->prepare(input);
    return;
  }
  stream_.synchronize();
  if (consumed_.isCreated()) {
    consumed_.wait(stream_);
  }
  const bool device_prefill =
      device_tokens.defined() && owner_.prefill_hashes_on_device(input);
  if (device_prefill) {
    CHECK_EQ(device_tokens.numel(), host_tokens_.numel());
    CHECK(device_tokens.device() == owner_.options_.device());
    // The preceding side-stream synchronization makes replacing the prior
    // snapshot safe, including when the caller supplied a temporary tensor.
    prefill_tokens_ = std::move(device_tokens);
    CHECK_EQ(collect_query_lengths(
                 input, input.meta.num_sequences, /*lengths=*/nullptr),
             host_tokens_.numel());
    cpu_threads_ = torch::get_num_threads();
    inputs_ready_.place(
        torch_mlu::getCurrentMLUStream(owner_.options_.device().index()));
    inputs_ready_.wait(stream_);
    torch_mlu::mlu::MLUStreamGuard guard(stream_);
    if (!prefill_device_input_) {
      prefill_device_input_ =
          owner_.device_hasher_->create_input(host_tokens_.numel());
    }
    prefill_device_input_->prepare(input);
    // Keep the device hashes in persistent staging. The temporary output is
    // produced and consumed on this side stream, while graph/decode consumers
    // retain the stable hashes_ address and the existing consumed_ fence.
    hashes_.copy_(
        owner_.hash_on_device(prefill_tokens_, *prefill_device_input_),
        /*non_blocking=*/true);
    host_hashes_.copy_(hashes_, /*non_blocking=*/true);
    CHECK_EQ(cnrtInvokeHostFunc(stream_.stream(), lookup, this), cnrtSuccess);
    upload_host_outputs(/*include_hashes=*/false);
    publish_prefetch();
    return;
  }
  CHECK(tokens.defined()) << "Engram CPU hashing requires host tokens";
  CHECK(tokens.device().is_cpu());
  CHECK_EQ(tokens.numel(), host_tokens_.numel());
  host_tokens_.copy_(tokens);
  if (host_history_.size(0) < input.meta.num_sequences) {
    host_history_ =
        torch::empty({input.meta.num_sequences, host_history_.size(1)},
                     host_history_.options().pinned_memory(true));
  }
  const auto& history = input.embedding.engram_lookback_token_ids;
  CHECK_EQ(history.size(), input.meta.num_sequences * host_history_.size(1));
  copy_lookback_history(history, host_history_);
  CHECK_EQ(collect_query_lengths(input, input.meta.num_sequences, &lengths_),
           tokens.numel());
  cpu_threads_ = torch::get_num_threads();
  torch_mlu::mlu::MLUStreamGuard guard(stream_);
  CHECK_EQ(cnrtInvokeHostFunc(stream_.stream(), compute, this), cnrtSuccess);
  upload_host_outputs(/*include_hashes=*/true);
  publish_prefetch();
}

void DeepseekV41EngramImpl::HashState::mark_consumed() {
  // Capture records consumers without executing them. A fence recorded there
  // could never execute when capture is discarded; replay owners mark this
  // boundary again outside capture after submitting the actual consumers.
  if (torch_mlu::currentStreamCaptureStatus() !=
      torch_mlu::CaptureStatus::None) {
    return;
  }
  CHECK(!prefetch_required_)
      << "Engram graph input was prepared without device history, but no "
      << "prefetch supplied its hashes before the consumers ran";
  consumed_.place(
      torch_mlu::getCurrentMLUStream(owner_.options_.device().index()));
}

torch::Tensor DeepseekV41EngramImpl::HashState::lookup_packed_rows() const {
  if (owner_.shard_hash_heads(hashes_.size(0))) {
    // The decoder reads this rank-major layout directly, including padded
    // head owners, without materializing a token-major permutation.
    return owner_.lookup_group_->allgather_base_sync(packed_rows_)
        .view({-1, packed_rows_.size(1)});
  }
  return packed_rows_;
}

torch::Tensor DeepseekV41EngramImpl::HashState::forward(
    const torch::Tensor& tokens) {
  if (device_input_) {
    return owner_.hash_on_device(tokens, *device_input_);
  }
  CHECK_EQ(tokens.numel(), host_tokens_.numel())
      << "Engram token count must match the staging capacity";
  if (prefetched_) {
    rows_ready_.wait(
        torch_mlu::getCurrentMLUStream(owner_.options_.device().index()));
    return hashes_;
  }
  // Standalone layer users may capture the inline path without the model's
  // graph-input hook. Device history snapshots preserve replay semantics.
  CHECK(!prefetch_required_)
      << "Engram graph input was prepared without device history; prefetch() "
      << "must run before forward";
  host_tokens_.copy_(tokens.to(torch::kInt64), /*non_blocking=*/true);
  host_history_.copy_(history_, /*non_blocking=*/true);
  CHECK_EQ(cnrtInvokeHostFunc(torch_mlu::getCurMLUStream(), compute, this),
           cnrtSuccess);
  upload_host_outputs(/*include_hashes=*/true);
  if (torch_mlu::currentStreamCaptureStatus() ==
      torch_mlu::CaptureStatus::None) {
    rows_ready_.place(
        torch_mlu::getCurrentMLUStream(owner_.options_.device().index()));
  }
  return hashes_;
}

void DeepseekV41EngramImpl::prepare_embeddings(
    const torch::Tensor& host_tokens,
    const ModelInputParams& input,
    const torch::Tensor& device_tokens) {
  const int64_t rows =
      device_tokens.defined() ? device_tokens.numel() : host_tokens.numel();
  if (!eager_state_ || eager_state_->forward_capacity() != rows) {
    eager_state_ = create_hash_state(rows);
  }
  eager_state_->prepare_eager(host_tokens, input, device_tokens);
  eager_prepared_ = true;
}

std::unique_ptr<DeepseekV41EngramImpl::HashState>
DeepseekV41EngramImpl::create_hash_state(int64_t rows,
                                         int64_t query_width) const {
  return std::make_unique<HashState>(*this, rows, query_width);
}

torch::Tensor DeepseekV41EngramImpl::forward(const torch::Tensor& hidden,
                                             const torch::Tensor& tokens,
                                             const ModelInputParams& input) {
  return forward(hidden, tokens, input, EngramLayerInput());
}

torch::Tensor DeepseekV41EngramImpl::forward(
    const torch::Tensor& hidden,
    const torch::Tensor& tokens,
    const ModelInputParams& input,
    const EngramLayerInput& engram_input,
    const torch::Tensor& row_indices) {
  HashState* hash_state = engram_input.hash_state;
  const EngramCpLayout* cp_layout = engram_input.cp_layout;
  const int64_t dim = args_.engram_head_dim();
  CHECK(hash_state == nullptr || !engram_input.hashes.defined())
      << "Engram accepts a single hash source";
  // Rows prepared by prepare_embeddings() belong to the next eager forward
  // only. Any other caller would leave them stale for a later step.
  CHECK(!eager_prepared_ ||
        (hash_state == nullptr && !engram_input.hashes.defined()))
      << "Engram eager rows were prepared but not consumed";
  torch::Tensor hashes = engram_input.hashes;
  if (!hashes.defined()) {
    if (hash_state == nullptr) {
      if (!eager_prepared_) {
        // Read tokens back only when this forward hashes them on the CPU.
        const torch::Tensor host_tokens =
            cpu_offload_ && !prefill_hashes_on_device(input)
                ? tokens.to(torch::kCPU, torch::kInt64).contiguous()
                : torch::Tensor();
        prepare_embeddings(host_tokens, input, tokens);
      }
      hash_state = eager_state_.get();
      eager_prepared_ = false;
    }
    hashes = hash_state->forward(tokens);
  }
  CHECK(!cpu_offload_ || hash_state != nullptr);
  const bool cp_reduce_scatter =
      uses_cp_reduce_scatter() && cp_layout != nullptr;
  torch::Tensor output_row_indices;
  if (cp_reduce_scatter) {
    CHECK(row_indices.defined());
    CHECK(tp_group_ != nullptr);
    CHECK_EQ(tp_group_->world_size(), cp_layout->tp_size);
    CHECK_EQ(lookup_group_->world_size(),
             cp_layout->cp_size * cp_layout->tp_size);
    CHECK_EQ(lookup_group_->rank(),
             cp_layout->cp_rank * cp_layout->tp_size + tp_group_->rank());
    CHECK_EQ(row_indices.numel(), cp_layout->local_rows);
    CHECK_EQ(hidden.size(0), cp_layout->local_rows);
    CHECK_EQ(cp_layout->pack_indices.numel(),
             cp_layout->rows_per_rank * lookup_group_->world_size());
    // Decode directly into the EP destination order, including padding, so
    // the full embedding never needs a separate permutation or copy.
    output_row_indices = cp_layout->pack_indices;
  }
  torch::Tensor rows;
  if (cpu_offload_) {
    const int64_t head_shards =
        shard_hash_heads(hashes.size(0)) ? lookup_group_->world_size() : 1;
    rows = kernel::mlu::bangc::engram_dequantize_packed(
        hash_state->lookup_packed_rows(),
        hashes,
        fp8_values_,
        dim,
        scale_dim(),
        table_start_,
        table_end_,
        head_shards,
        /*all_owned=*/node_shared_memory_,
        output_row_indices);
  } else {
    rows = kernel::mlu::bangc::engram_gather_dequantize(
        weights_.at("embed.weight"),
        weights_.at("embed.scale"),
        hashes,
        fp8_values_,
        table_start_,
        table_end_,
        output_row_indices);
  }
  const bool gather_after_gate = cp_reduce_scatter && cp_layout->tp_size > 1;
  torch::Tensor gate_hidden = hidden;
  if (cp_reduce_scatter) {
    rows = parallel_state::reduce_scatter(rows, lookup_group_);
    if (gather_after_gate) {
      // Keep token shards through the row-independent projection and gate.
      // Gathering the final hidden rows avoids repeating WKV on each TP rank.
      const int64_t shard_rows = cp_layout->rows_per_rank;
      const int64_t first_row = tp_group_->rank() * shard_rows;
      const int64_t remaining =
          std::max<int64_t>(cp_layout->local_rows - first_row, 0);
      const int64_t valid_rows = std::min<int64_t>(shard_rows, remaining);
      CHECK_EQ(rows.size(0), shard_rows);
      if (valid_rows == shard_rows) {
        gate_hidden = hidden.narrow(/*dim=*/0, first_row, shard_rows);
      } else {
        // A TP shard can be entirely padding even when its CP rank has real
        // tokens. Zero hidden padding keeps the gate input well-defined; its
        // extra output rows are discarded after the TP gather.
        auto shape = hidden.sizes().vec();
        shape[0] = shard_rows;
        gate_hidden = torch::zeros(shape, hidden.options());
        if (valid_rows > 0) {
          gate_hidden.narrow(/*dim=*/0, /*start=*/0, valid_rows)
              .copy_(hidden.narrow(/*dim=*/0, first_row, valid_rows));
        }
      }
    } else {
      rows = rows.narrow(/*dim=*/0, /*start=*/0, cp_layout->local_rows);
    }
  } else {
    // CP-inactive calls, including DP decode, and shared CPU tables retain
    // their existing group and token semantics. Allreduce is also an explicit
    // configuration for measuring the two communication strategies.
    if (!node_shared_memory_) {
      rows = parallel_state::reduce(rows, lookup_group_);
    }
    if (row_indices.defined()) {
      rows = rows.index_select(/*dim=*/0, row_indices);
    }
  }
  torch::Tensor kv = wkv_->forward(rows.flatten(1));
  torch::Tensor output =
      kernel::mlu::bangc::engram_gate(gate_hidden,
                                      kv,
                                      weights_.at("q_weight"),
                                      weights_.at("k_weight"),
                                      args_.rms_norm_eps());
  if (gather_after_gate) {
    // Rank-major output is contiguous, so merging rank and token dimensions
    // is a view rather than a second copy of the gathered hidden state.
    output = tp_group_->allgather_base_sync(output).flatten(/*start_dim=*/0,
                                                            /*end_dim=*/1);
    output = output.narrow(/*dim=*/0, /*start=*/0, cp_layout->local_rows);
  }
  torch::Tensor local_tokens =
      row_indices.defined() ? tokens.index_select(0, row_indices) : tokens;
  torch::Tensor text_mask = local_tokens.ne(kEngramImageSentinelBaseId) &
                            local_tokens.ne(kEngramImagePadId);
  output = torch::where(text_mask.view({-1, 1, 1}), output, hidden);
  if (hash_state != nullptr) {
    hash_state->mark_consumed();
  }
  return output;
}

}  // namespace xllm::layer
