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
#pragma once

// Shared multimodal preprocessing for native MLU and Python GLM5-Next VLMs.

#include "models/model_registry.h"
#include "processors/glm4v_prompt_processor.h"
#include "processors/glm4v_video_processor.h"
#include "processors/multimodal_processor.h"
#include "processors/qwen2_vl_image_processor.h"

namespace xllm {

// Bind preprocessing policy at registration; all processing stays shared.
class Glm5NextVideoProcessor final : public Glm4VVideoProcessor {
 public:
  explicit Glm5NextVideoProcessor(const ModelArgs& args)
      : Glm4VVideoProcessor(args, SamplingMode::FIXED_FPS) {}
};

using Glm5NextVLMultimodalProcessor =
    MultimodalProcessor<GLM4VPromptProcessor,
                        Qwen2VLImageProcessor,
                        Glm5NextVideoProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(glm5_next, Glm5NextVLMultimodalProcessor);

}  // namespace xllm
