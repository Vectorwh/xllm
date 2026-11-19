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

#include "layers/mlu/deepseek_v41/deepseek_v41_engram_inputs.h"

#include <algorithm>
#include <utility>

#include "layers/mlu/deepseek_v4/deepseek_v4_cp_context.h"
#include "layers/mlu/deepseek_v41/deepseek_v41_metadata.h"

namespace xllm::layer {
void DeepseekV41EngramInputs::GraphState::finish_consumption() {
  for (auto& hash_state : hash_states_) {
    if (hash_state) {
      hash_state->mark_consumed();
    }
  }
}

void DeepseekV41EngramInputs::Step::use_cp_context(
    const mlu_v4_cp::DeepseekV4CpContext& context) {
  if (owner_.uses_cp_reduce_scatter_) {
    cp_layout_ = build_engram_cp_layout(context, owner_.tp_size_);
  }
}

EngramLayerInput DeepseekV41EngramInputs::Step::layer(size_t layer) const {
  EngramLayerInput input;
  const auto& engram = owner_.layers_.at(layer);
  if (!engram) {
    return input;
  }
  if (graph_ != nullptr) {
    input.hash_state = graph_->hash_state(layer);
  }
  if (device_hashes_.defined()) {
    input.hashes = device_hashes_.select(/*dim=*/0, engram->engram_id());
  }
  input.cp_layout = cp_layout_ ? &*cp_layout_ : nullptr;
  return input;
}

DeepseekV41EngramInputs::DeepseekV41EngramInputs(
    std::vector<DeepseekV41Engram> layers,
    int32_t tp_size)
    : layers_(std::move(layers)), tp_size_(tp_size) {
  for (const auto& layer : layers_) {
    if (!layer) {
      continue;
    }
    has_engram_ = true;
    hash_all_on_device_ = hash_all_on_device_ || !layer->cpu_offload();
    uses_cp_reduce_scatter_ =
        uses_cp_reduce_scatter_ || layer->uses_cp_reduce_scatter();
  }
}

void DeepseekV41EngramInputs::load_hash_state(const StateDict& state) {
  for (auto& layer : layers_) {
    if (layer) {
      layer->load_hash_state(state, &device_hasher_);
    }
  }
  CHECK(!hash_all_on_device_ || device_hasher_ != nullptr);
}

std::unique_ptr<DeepseekV41EngramInputs::GraphState>
DeepseekV41EngramInputs::create_graph_state(int64_t rows,
                                            int64_t query_width) const {
  auto graph = std::make_unique<GraphState>();
  graph->hash_states_.resize(layers_.size());
  if (hash_all_on_device_) {
    CHECK(device_hasher_ != nullptr)
        << "Engram hash state must be loaded before graph capture";
    graph->device_input_ = device_hasher_->create_input(rows);
    return graph;
  }
  for (size_t id = 0; id < layers_.size(); ++id) {
    if (layers_[id]) {
      graph->hash_states_[id] =
          layers_[id]->create_hash_state(rows, query_width);
    }
  }
  return graph;
}

void DeepseekV41EngramInputs::prepare_graph(
    GraphState& graph,
    const ModelInputParams& input) const {
  if (graph.device_input_) {
    graph.device_input_->prepare(input);
  }
  for (auto& hash_state : graph.hash_states_) {
    if (hash_state) {
      // prefetch_graph() supplies the hashes; no device history is needed.
      hash_state->prepare(input, /*device_history=*/false);
    }
  }
}

void DeepseekV41EngramInputs::prefetch_graph(
    GraphState& graph,
    const torch::Tensor& tokens,
    const ModelInputParams& input) const {
  if (graph.hashes_on_device() || !has_engram_) {
    return;
  }
  const torch::Tensor host_tokens = valid_engram_host_tokens(tokens, input);
  for (auto& hash_state : graph.hash_states_) {
    if (hash_state) {
      hash_state->prefetch(tokens, host_tokens);
    }
  }
}

DeepseekV41EngramInputs::Step DeepseekV41EngramInputs::begin(
    const torch::Tensor& tokens,
    const ModelInputParams& input,
    const GraphState* graph) {
  Step step(*this, graph);
  if (!has_engram_ || !deepseek_v41::has_request_rows(input)) {
    return step;
  }
  if (!hash_all_on_device_) {
    // Graph lookups were already started by prefetch_graph().
    if (graph == nullptr) {
      prepare_eager(tokens, input);
    }
    return step;
  }
  if (graph != nullptr) {
    step.device_hashes_ =
        device_hasher_->forward(tokens, *graph->device_input_);
    return step;
  }
  if (!eager_device_input_ ||
      eager_device_input_->capacity() != tokens.numel()) {
    eager_device_input_ = device_hasher_->create_input(tokens.numel());
  }
  eager_device_input_->prepare(input);
  step.device_hashes_ = device_hasher_->forward(tokens, *eager_device_input_);
  return step;
}

void DeepseekV41EngramInputs::prepare_eager(const torch::Tensor& tokens,
                                            const ModelInputParams& input) {
  torch::Tensor host_tokens = valid_engram_host_tokens(tokens, input);
  // Read the tokens back once for all layers, and only when some layer hashes
  // them on the CPU for this input.
  const bool needs_host_tokens =
      std::any_of(layers_.begin(), layers_.end(), [&input](const auto& layer) {
        return layer && !layer->prefill_hashes_on_device(input);
      });
  if (!host_tokens.defined() && needs_host_tokens) {
    host_tokens = tokens.to(torch::kCPU, torch::kInt64).contiguous();
  }
  for (auto& layer : layers_) {
    if (layer) {
      layer->prepare_embeddings(host_tokens, input, tokens);
    }
  }
}
}  // namespace xllm::layer
