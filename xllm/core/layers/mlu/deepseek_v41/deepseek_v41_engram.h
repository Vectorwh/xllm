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

#pragma once
#include <framework/core/MLUEvent.h>
#include <framework/core/MLUStream.h>

#include <memory>

#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm::layer {
namespace mlu_v4_cp {
struct DeepseekV4CpContext;
}  // namespace mlu_v4_cp

// All ranks share the same destination order. Padding makes every EP rank
// receive an equal slice while preserving each CP rank's local token order.
struct EngramCpLayout {
  torch::Tensor pack_indices;
  int64_t rows_per_rank = 0;
  int64_t local_rows = 0;
  int32_t cp_rank = 0;
  int32_t cp_size = 1;
  int32_t tp_size = 1;
};

EngramCpLayout build_engram_cp_layout(
    const mlu_v4_cp::DeepseekV4CpContext& context,
    int32_t tp_size);

// Returns input.embedding.input_token_ids_host when that upstream snapshot
// still describes `tokens` (live rows only in graph mode), otherwise an
// undefined tensor. HashState accepts only snapshots validated here.
torch::Tensor valid_engram_host_tokens(const torch::Tensor& tokens,
                                       const ModelInputParams& input);

StateDict build_deepseek_v41_hash_state(const ModelArgs& args,
                                        const std::string& tokenizer_path);

// Immutable device parameters shared by all Engram layers of a model.
class EngramDeviceHasher final {
 public:
  class Input final {
   public:
    Input(int64_t rows, int64_t depth, const torch::TensorOptions& options);
    ~Input();
    void prepare(const ModelInputParams& input, bool graph = false);
    int64_t capacity() const { return rows_; }
    bool prepared() const { return history_.defined(); }
    const torch::Tensor& starts() const { return starts_; }
    const torch::Tensor& history() const { return history_; }

   private:
    int64_t rows_;
    int64_t depth_;
    torch::TensorOptions options_;
    // Allocated by the first prepare(), which knows the request capacity.
    torch::Tensor host_starts_, host_history_, starts_, history_;
    torch_mlu::MLUEvent uploaded_;
  };

  EngramDeviceHasher(const StateDict& state,
                     int64_t pad_token,
                     const torch::TensorOptions& options);
  std::unique_ptr<Input> create_input(int64_t rows) const;
  // Hashes every Engram layer in one launch: [layers, tokens, hash columns].
  torch::Tensor forward(const torch::Tensor& tokens, const Input& input) const;
  // Hashes one Engram layer, indexed by its position in engram_layer_ids:
  // [tokens, hash columns].
  torch::Tensor forward_layer(const torch::Tensor& tokens,
                              const Input& input,
                              int32_t layer) const;

 private:
  torch::Tensor launch(const torch::Tensor& tokens,
                       const Input& input,
                       const torch::Tensor& multipliers,
                       const torch::Tensor& primes,
                       const torch::Tensor& offsets) const;
  int64_t pad_token_;
  torch::TensorOptions options_;
  torch::Tensor token_map_, multipliers_, primes_, offsets_;
};

struct EngramLayerInput;

class DeepseekV41EngramImpl final : public torch::nn::Module {
 public:
  class HashState final {
   public:
    HashState(const DeepseekV41EngramImpl& owner,
              int64_t rows,
              int64_t query_width = 1);
    ~HashState();
    int64_t forward_capacity() const { return host_tokens_.numel(); }
    void prepare(const ModelInputParams& input, bool device_history = true);
    void prefetch(const torch::Tensor& tokens,
                  const torch::Tensor& host_tokens = {});
    // Passing an existing device snapshot permits eager CPU-table prefill to
    // hash on MLU. Without it, this method always uses the CPU hash path.
    // host_tokens may be undefined only when the owner's
    // prefill_hashes_on_device(input) holds and device_tokens is defined.
    void prepare_eager(const torch::Tensor& host_tokens,
                       const ModelInputParams& input,
                       torch::Tensor device_tokens = {});
    // Standalone graph owners call this after each replay; the runtime calls
    // it through ModelGraphMetadataState for model-owned states.
    void mark_consumed();
    torch::Tensor forward(const torch::Tensor& tokens);
    torch::Tensor lookup_packed_rows() const;

   private:
    static void compute(void* state) noexcept;
    static void lookup(void* state) noexcept;
    // Enqueue H2D uploads of the CPU-produced staging on the current stream.
    void upload_host_outputs(bool include_hashes);
    // Publish side-stream uploads as the input for the next forward.
    void publish_prefetch();
    const DeepseekV41EngramImpl& owner_;
    torch::Tensor host_tokens_, host_history_, host_hashes_;
    torch::Tensor history_, hashes_;
    torch::Tensor host_weight_rows_, host_scale_rows_;
    torch::Tensor host_packed_rows_, packed_rows_;
    torch::Tensor prepared_history_;
    // Retain eager device input storage until its side-stream hash completes.
    torch::Tensor prefill_tokens_;
    torch_mlu::MLUStream stream_;
    torch_mlu::MLUEvent inputs_ready_, history_ready_, rows_ready_, consumed_;
    bool prefetched_ = false;
    // Set by prepare(device_history=false): the device history is not
    // uploaded, so only prefetch() can supply this invocation's hashes.
    bool prefetch_required_ = false;
    int32_t cpu_threads_ = 1;
    std::vector<int32_t> lengths_;
    std::unique_ptr<EngramDeviceHasher::Input> device_input_;
    // Separate from device_input_: CPU-table graph/decode still hashes on CPU.
    std::unique_ptr<EngramDeviceHasher::Input> prefill_device_input_;
  };
  std::unique_ptr<HashState> create_hash_state(int64_t rows,
                                               int64_t query_width = 1) const;
  DeepseekV41EngramImpl(const ModelContext& context, int32_t engram_id);
  void load_state_dict(const StateDict& state);
  // `shared_hasher` lets the layers of one model share a single set of device
  // hash constants: a layer that needs them reuses *shared_hasher when it is
  // set, and otherwise creates the hasher from `state` and publishes it there.
  void load_hash_state(
      const StateDict& state,
      std::shared_ptr<const EngramDeviceHasher>* shared_hasher = nullptr);
  void verify_loaded_weights() const;
  // Position of this layer in engram_layer_ids, which is also its row in
  // EngramDeviceHasher::forward.
  int32_t engram_id() const { return engram_id_; }
  bool cpu_offload() const { return cpu_offload_; }
  bool uses_device_hasher() const {
    return !cpu_offload_ || prefill_device_hash_;
  }
  bool uses_cp_reduce_scatter() const {
    return cp_reduce_scatter_ && !node_shared_memory_;
  }
  // Whether an eager forward of `input` hashes host-table rows on MLU and so
  // needs no host tokens.
  bool prefill_hashes_on_device(const ModelInputParams& input) const;
  // Starts the next eager forward's lookup. host_tokens may be undefined when
  // prefill_hashes_on_device(input) holds and device_tokens is defined.
  void prepare_embeddings(const torch::Tensor& host_tokens,
                          const ModelInputParams& input,
                          const torch::Tensor& device_tokens = {});
  // Hashes and looks up its own rows, or consumes prepare_embeddings().
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& tokens,
                        const ModelInputParams& input);
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& tokens,
                        const ModelInputParams& input,
                        const EngramLayerInput& engram_input,
                        const torch::Tensor& row_indices = {});

 private:
  void compute_hashes(const int64_t* tokens,
                      const int64_t* history,
                      const std::vector<int32_t>& lengths,
                      int64_t* output) const;
  void lookup_host_rows(const torch::Tensor& hashes,
                        torch::Tensor& weight_rows,
                        torch::Tensor& scale_rows) const;
  torch::Tensor hash_on_device(const torch::Tensor& tokens,
                               const EngramDeviceHasher::Input& input) const;
  int64_t hash_columns() const;
  int64_t shard_hash_columns() const;
  int64_t scale_dim() const;
  bool shard_hash_heads(int64_t tokens) const;
  int64_t local_hash_columns(int64_t tokens) const;
  ModelArgs args_;
  int32_t engram_id_;
  ProcessGroup* lookup_group_;
  ProcessGroup* world_group_;
  ProcessGroup* tp_group_;
  torch::TensorOptions options_;
  int64_t table_start_;
  int64_t table_end_;
  bool cpu_offload_;
  bool node_shared_memory_ = false;
  bool cp_reduce_scatter_ = true;
  bool prefill_device_hash_ = true;
  bool eager_prepared_ = false;
  std::unordered_map<std::string, torch::Tensor> weights_;
  torch::Tensor token_map_, multipliers_, primes_, offsets_, fp8_values_;
  std::shared_ptr<const EngramDeviceHasher> device_hasher_;
  // Drain callbacks before destroying the tables and hash constants they read.
  std::unique_ptr<HashState> eager_state_;
};
TORCH_MODULE(DeepseekV41Engram);

// What one Engram layer consumes in one forward. At most one hash source is
// set: `hash_state` (graph or standalone staging) or `hashes` (device hashes
// computed for the whole stack). With neither, the layer uses its own eager
// staging. `cp_layout` selects the row-sharded CP communication.
struct EngramLayerInput {
  DeepseekV41EngramImpl::HashState* hash_state = nullptr;
  torch::Tensor hashes;
  const EngramCpLayout* cp_layout = nullptr;
};
}  // namespace xllm::layer
