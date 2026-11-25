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

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model_context.h"
#include "core/runtime/dit_forward_params.h"
#include "models/model_registry.h"

namespace xllm {

// Form-B text encoder proxy (vlm_encoder_proxy.h); referenced by pointer.
class VlmTextEncoderProxy;

// Cross-language shell for the python DiT backend: implements the C++
// ``DiTModel`` interface and drives a python ``DiTPipelineExecutor`` built
// from ``xllm.python.model_executor.dit_executor``. Mirrors the LLM-side
// ``PyCausalLM``/``PyExecutorImpl`` split — every request crosses the
// boundary once for the pipeline call (plus once for a form-B text encoder),
// always under the GIL.
//
// Hidden visibility like ``PyCausalLM``: the class holds ``pybind11::object``
// members whose types are compiled with hidden visibility, which a
// default-visibility enclosing class would downgrade (-Wattributes).
//
// Selected from ``create_dit_model`` when ``--model_impl python`` is set.
// The native C++ pipelines remain registered and are untouched; the backend
// routing keeps both stacks available.
class __attribute__((visibility("hidden"))) PyDiTModel : public DiTModel {
 public:
  explicit PyDiTModel(const DiTModelContext& context);
  ~PyDiTModel() override;

  DiTForwardOutput forward(const DiTForwardInput& input) override;

  torch::Device device() const override { return device_; }
  const torch::TensorOptions& options() const override { return options_; }

  void load_model(std::unique_ptr<DiTModelLoader> loader) override;

  bool load_lora_model(const LoRAConfig& /*config*/,
                       const std::vector<LoRARequest>& requests) override {
    if (requests.empty()) {
      return true;
    }
    LOG(WARNING) << "LoRA loading is not supported by the Python DiT backend.";
    return false;
  }

 private:
  void register_python_process_groups();
  void build_text_encoder_proxy();

  DiTModelContext context_;
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  torch::Device device_ = torch::kCPU;
  std::string dtype_;
  std::string model_type_;

  // Form-B text encoder owned here, injected into the python pipeline as a
  // proxy object; null when the model has no text_encoder component.
  std::shared_ptr<VlmTextEncoderProxy> text_encoder_proxy_;
  std::unique_ptr<DiTFolderLoader> text_encoder_loader_;

  pybind11::object py_executor_;
};

}  // namespace xllm
