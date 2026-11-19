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

#include <memory>
#include <optional>
#include <vector>

#include "layers/mlu/deepseek_v41/deepseek_v41_engram.h"

namespace xllm::layer {
// Prepares the Engram inputs of a whole decoder stack. Device-resident tables
// hash every layer in one launch; host tables start their lookups before the
// stack runs so the CPU gathers overlap with the preceding layers. Table
// placement, hash placement and CP communication stay Engram decisions: the
// model only hands over its tokens and forwards each layer's EngramLayerInput.
class DeepseekV41EngramInputs final {
 public:
  // Staging owned by one captured graph bucket.
  class GraphState final {
   public:
    // Call outside capture after each capture or replay is enqueued.
    void finish_consumption();
    // Whether the bucket hashes all layers on device instead of per layer.
    bool hashes_on_device() const { return device_input_ != nullptr; }
    size_t num_layers() const { return hash_states_.size(); }
    // Per-layer staging, or nullptr for layers without Engram and for
    // device-hashed buckets.
    DeepseekV41EngramImpl::HashState* hash_state(size_t layer) const {
      return hash_states_.at(layer).get();
    }

   private:
    friend class DeepseekV41EngramInputs;
    std::unique_ptr<EngramDeviceHasher::Input> device_input_;
    std::vector<std::unique_ptr<DeepseekV41EngramImpl::HashState>> hash_states_;
  };

  // The Engram inputs of one forward. It must outlive the decoder stack.
  class Step final {
   public:
    // Selects the row-sharded CP communication for layers that use it.
    void use_cp_context(const mlu_v4_cp::DeepseekV4CpContext& context);
    EngramLayerInput layer(size_t layer) const;

   private:
    friend class DeepseekV41EngramInputs;
    Step(const DeepseekV41EngramInputs& owner, const GraphState* graph)
        : owner_(owner), graph_(graph) {}
    const DeepseekV41EngramInputs& owner_;
    const GraphState* graph_;
    // [engram layers, tokens, hash columns] when hashed in one device launch.
    torch::Tensor device_hashes_;
    std::optional<EngramCpLayout> cp_layout_;
  };

  // `layers` is indexed by decoder layer; layers without Engram are empty.
  DeepseekV41EngramInputs(std::vector<DeepseekV41Engram> layers,
                          int32_t tp_size);
  void load_hash_state(const StateDict& state);

  std::unique_ptr<GraphState> create_graph_state(int64_t rows,
                                                 int64_t query_width) const;
  // Per capture/replay: stage the request layout, then start the lookups for
  // the tokens. Both run outside capture, in this order.
  void prepare_graph(GraphState& graph, const ModelInputParams& input) const;
  void prefetch_graph(GraphState& graph,
                      const torch::Tensor& tokens,
                      const ModelInputParams& input) const;

  // Starts this forward's Engram input work. `graph` is the active bucket's
  // state in graph mode and nullptr for eager execution.
  Step begin(const torch::Tensor& tokens,
             const ModelInputParams& input,
             const GraphState* graph);

 private:
  void prepare_eager(const torch::Tensor& tokens,
                     const ModelInputParams& input);

  std::vector<DeepseekV41Engram> layers_;
  int32_t tp_size_;
  bool has_engram_ = false;
  // Device-resident tables need no host rows, so one launch hashes all layers.
  bool hash_all_on_device_ = false;
  bool uses_cp_reduce_scatter_ = false;
  std::shared_ptr<const EngramDeviceHasher> device_hasher_;
  std::unique_ptr<EngramDeviceHasher::Input> eager_device_input_;
};
}  // namespace xllm::layer
