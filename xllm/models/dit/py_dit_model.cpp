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

#include "models/dit/py_dit_model.h"

#include <Python.h>
#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/dit_cache/dit_cache_config.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "models/dit/vlm_encoder_proxy.h"
#include "models/py_model_helper.h"
#include "models/vlm/mposition/mposition.h"
#include "processors/multimodal_processor.h"

namespace py = pybind11;

namespace xllm {
namespace {

constexpr const char* kPythonComponents[] = {"transformer",
                                             "vae",
                                             "audio_vae",
                                             "scheduler"};

// Ordered kinds the python backend registers as named subgroups; must match
// the python side's expectations and be identical on every rank. "tp" records
// the dit_tp layout so TP-sharded components (JoyOV2 MoE / attention) can
// mirror the C++ Column/RowParallelLinear shard+reduce semantics.
constexpr const char* kGroupKinds[] = {"tp",
                                       "sp",
                                       "cfg",
                                       "vae",
                                       "text_encoder_tp"};

py::object read_component_config(const std::string& component_dir) {
  // Component configs are not named uniformly: transformer/vae use config.json
  // while the scheduler ships scheduler_config.json. Mirror the native
  // DiTFolderLoader, which reads config.json for safetensors folders and
  // otherwise any *.json in the component directory.
  std::string config_path;
  for (const char* candidate : {"config.json", "scheduler_config.json"}) {
    const std::string path = component_dir + "/" + candidate;
    if (std::filesystem::exists(path)) {
      config_path = path;
      break;
    }
  }
  if (config_path.empty()) {
    for (const auto& entry :
         std::filesystem::directory_iterator(component_dir)) {
      if (entry.path().extension() == ".json") {
        config_path = entry.path().string();
        break;
      }
    }
  }
  CHECK(!config_path.empty()) << "No JSON config found in " << component_dir;
  std::ifstream file(config_path);
  CHECK(file.is_open()) << "Failed to open " << config_path;
  nlohmann::json json;
  file >> json;
  return py::module_::import("json").attr("loads")(py::str(json.dump()));
}

std::string component_dir(const DiTModelLoader& loader,
                          const std::string& component) {
  return loader.model_root_path() + "/" + component;
}

void clear_python_object(py::object& object) {
  if (!object) {
    return;
  }
  if (!Py_IsInitialized()) {
    // CPython has already torn down its GIL; leaking at process exit beats
    // decref'ing during static destruction.
    (void)object.release();
    return;
  }
  py::gil_scoped_acquire gil;
  object = py::object();
}

}  // namespace

// Exposes the form-B text encoder proxy to the embedded interpreter.
PYBIND11_EMBEDDED_MODULE(xllm_dit, m) {
  py::class_<VlmTextEncoderProxy, std::shared_ptr<VlmTextEncoderProxy>>(
      m, "VlmTextEncoderProxy")
      .def("encode_prompts",
           &VlmTextEncoderProxy::encode_prompts,
           py::arg("prompts"),
           py::arg("images"));
}

PyDiTModel::PyDiTModel(const DiTModelContext& context)
    : context_(context),
      parallel_args_(context.get_parallel_args()),
      options_(context.get_tensor_options()),
      device_(options_.device()),
      dtype_(dtype_to_string(options_)),
      model_type_(context.model_type()) {
  ensure_python_interpreter();

  // Named subgroups must exist before the python pipeline resolves its
  // collectives; they are derived from the layouts the DiT communicator
  // recorded on ParallelArgs.
  py::gil_scoped_acquire gil;
  register_python_process_groups();

  if (context_.has_component("text_encoder")) {
    build_text_encoder_proxy();
  }
}

PyDiTModel::~PyDiTModel() { clear_python_object(py_executor_); }

void PyDiTModel::register_python_process_groups() {
  const std::string device_str = c10::str(device_);
  const int32_t global_rank = parallel_args_.rank();
  const int32_t global_world_size = parallel_args_.world_size();

  // Collect the kinds that actually span ranks; a single-rank deployment has
  // no layouts at all and needs no python groups.
  std::vector<std::pair<std::string, const std::vector<std::vector<int32_t>>*>>
      to_register;
  for (const char* kind : kGroupKinds) {
    const std::string group_name = std::string("dit_") + kind;
    auto it = parallel_args_.python_group_layouts_.find(group_name);
    if (it != parallel_args_.python_group_layouts_.end()) {
      to_register.emplace_back(group_name, &it->second);
    }
  }
  if (to_register.empty()) {
    return;
  }

  CHECK(!parallel_args_.python_rendezvous_host_.empty())
      << "python DiT backend requires the rendezvous endpoint on ParallelArgs";
  CHECK_GT(parallel_args_.python_rendezvous_port_, 0);
  py::object init_group =
      py::module_::import("xllm.python.distributed").attr("init_group");

  for (const auto& [group_name, layout] : to_register) {
    // Rank of this process inside the group that contains it.
    int32_t rank_in_group = -1;
    for (const auto& ranks : *layout) {
      auto found = std::find(ranks.begin(), ranks.end(), global_rank);
      if (found != ranks.end()) {
        rank_in_group = static_cast<int32_t>(found - ranks.begin());
        break;
      }
    }
    CHECK_GE(rank_in_group, 0)
        << "python group layout " << group_name
        << " does not contain global rank " << global_rank;
    init_group(py::str(group_name),
               parallel_args_.python_rendezvous_host_,
               parallel_args_.python_rendezvous_port_,
               py::cast(*layout),
               rank_in_group,
               device_str,
               global_rank,
               global_world_size);
  }
}

void PyDiTModel::build_text_encoder_proxy() {
  const ModelArgs& text_encoder_model_args =
      context_.get_model_args("text_encoder");
  ProcessGroup* tp_group = parallel_args_.dit_text_encoder_tp_group_;
  CHECK(tp_group != nullptr)
      << "JoyImageEditPlus requires an injected text encoder TP group.";
  ParallelArgs vlm_parallel_args(
      tp_group->rank(), tp_group->world_size(), tp_group);
  vlm_parallel_args.tp_size(tp_group->world_size());
  vlm_parallel_args.tp_group_ = tp_group;
  ModelContext vlm_context(vlm_parallel_args,
                           text_encoder_model_args,
                           context_.get_quant_args("text_encoder"),
                           options_);

  const std::string encoder_type = text_encoder_model_args.model_type();
  auto factory = ModelRegistry::get_causalvlm_factory(encoder_type);
  CHECK(factory != nullptr)
      << "No causal VLM factory registered for text encoder type: "
      << encoder_type;
  std::unique_ptr<CausalVLM> encoder = factory(vlm_context);
  CHECK(encoder != nullptr)
      << "Failed to create the JoyImageEditPlus text encoder: " << encoder_type;

  text_encoder_proxy_ = std::make_shared<VlmTextEncoderProxy>(
      std::move(encoder), text_encoder_model_args, options_);
  text_encoder_proxy_->set_mposition_generator(
      MPositionGeneratorFactory::get_instance().create_mposition_generator(
          "qwen3_vl"));
}

void PyDiTModel::load_model(std::unique_ptr<DiTModelLoader> loader) {
  LOG(INFO) << "PyDiTModel loading from " << loader->model_root_path();

  py::gil_scoped_acquire gil;

  py::dict config_dict;
  config_dict["pipeline"] = py::str(model_type_);
  config_dict["dtype"] = py::str(dtype_);
  config_dict["device"] = py::str(c10::str(device_));
  config_dict["dit_cfg_group"] = py::str("dit_cfg");
  config_dict["dit_sp_group"] = py::str("dit_sp");
  config_dict["dit_vae_group"] = py::str("dit_vae");

  // Forward the configured DiT cache policy (and its numeric options) into the
  // python executor, so the python pipeline takes the same reuse decisions as
  // the native C++ one (--dit_cache_policy). Only the selected policy's option
  // struct is populated; serialize exactly those fields and let the python
  // policies fall back to the same defaults as dit_cache_config.h for the rest.
  const DiTCacheConfig& dit_config = context_.get_dit_config();
  std::string cache_policy = "None";
  py::dict cache_options;
  switch (dit_config.selected_policy) {
    case PolicyType::FBCache:
      cache_policy = "FBCache";
      cache_options["warmup_steps"] = dit_config.fbcache.warmup_steps;
      cache_options["residual_diff_threshold"] =
          dit_config.fbcache.residual_diff_threshold;
      break;
    case PolicyType::TaylorSeer:
      cache_policy = "TaylorSeer";
      cache_options["warmup_steps"] = dit_config.taylorseer.warmup_steps;
      cache_options["n_derivatives"] = dit_config.taylorseer.n_derivatives;
      cache_options["skip_interval_steps"] =
          dit_config.taylorseer.skip_interval_steps;
      break;
    case PolicyType::FBCacheTaylorSeer:
      cache_policy = "FBCacheTaylorSeer";
      cache_options["warmup_steps"] = dit_config.fbcachetaylorseer.warmup_steps;
      cache_options["residual_diff_threshold"] =
          dit_config.fbcachetaylorseer.residual_diff_threshold;
      cache_options["n_derivatives"] =
          dit_config.fbcachetaylorseer.n_derivatives;
      break;
    case PolicyType::ResidualCache:
      cache_policy = "ResidualCache";
      cache_options["start_steps"] =
          dit_config.residual_cache.dit_cache_start_steps;
      cache_options["end_steps"] =
          dit_config.residual_cache.dit_cache_end_steps;
      cache_options["start_blocks"] =
          dit_config.residual_cache.dit_cache_start_blocks;
      cache_options["end_blocks"] =
          dit_config.residual_cache.dit_cache_end_blocks;
      cache_options["skip_interval_steps"] =
          dit_config.residual_cache.skip_interval_steps;
      break;
    case PolicyType::None:
      break;
  }
  config_dict["dit_cache_policy"] = py::str(cache_policy);
  config_dict["dit_cache_options"] = cache_options;

  py::dict components;
  for (const char* component : kPythonComponents) {
    if (!loader->has_component(component)) {
      continue;
    }
    py::dict spec;
    spec["config"] = read_component_config(component_dir(*loader, component));
    py::object component_config = spec["config"];
    CHECK(component_config.contains("_class_name"))
        << component << " config.json is missing _class_name";
    spec["class_name"] = component_config["_class_name"];
    components[component] = spec;
  }

  // Register the PyStateDict binding before casting any checkpoint wrapper so
  // py::cast can resolve the Python type (mirrors PyCausalLM::load_model).
  ensure_xllm_weight_loader_module();
  py::module_::import("xllm_weight_loader");

  // Hand the checkpoint state dicts to python; the PyStateDict wrappers stay
  // valid for the duration of the create() call below, which consumes them
  // synchronously (mirrors PyCausalLM::load_model lifetime rules). Keep the
  // component loaders (and therefore their StateDicts) alive until load_model
  // returns, since PyStateDict holds a raw StateDict*.
  std::vector<std::unique_ptr<DiTFolderLoader>> held_loaders;
  auto take_state_dicts = [&loader,
                           &held_loaders](const std::string& component) {
    held_loaders.push_back(loader->take_component_loader(component));
    DiTFolderLoader* component_loader = held_loaders.back().get();
    py::list state_dicts;
    for (const auto& state_dict : component_loader->get_state_dicts()) {
      state_dicts.append(py::cast(PyStateDict(state_dict.get()),
                                  py::return_value_policy::move));
    }
    return state_dicts;
  };
  for (const char* component : {"transformer", "vae", "audio_vae"}) {
    if (!components.contains(component)) {
      continue;
    }
    components[component]["state_dicts"] = take_state_dicts(component);
  }

  if (text_encoder_proxy_ != nullptr && loader->has_component("text_encoder")) {
    // Import the embedded module first so the proxy's python type exists
    // before py::cast registers it into the config dict.
    py::module_::import("xllm_dit");
    text_encoder_loader_ = loader->take_component_loader("text_encoder");
    std::shared_ptr<Tokenizer> tokenizer = text_encoder_loader_->tokenizer();
    CHECK(tokenizer != nullptr)
        << "Failed to load JoyImageEditPlus Qwen3-VL tokenizer";
    text_encoder_proxy_->set_tokenizer(tokenizer);
    text_encoder_proxy_->set_multimodal_processor(
        create_multimodal_processor(context_.get_model_args("text_encoder"),
                                    tokenizer,
                                    /*max_cache_items=*/0,
                                    text_encoder_loader_->tokenizer_args()));
    // Load the VLM weights on the C++ side (KV cache / TP reuse).
    text_encoder_proxy_->encoder()->load_model(std::move(text_encoder_loader_));
    text_encoder_loader_.reset();

    py::dict text_encoder_spec;
    text_encoder_spec["kind"] = py::str("external");
    text_encoder_spec["proxy"] = py::cast(text_encoder_proxy_);
    components["text_encoder"] = text_encoder_spec;
  }

  config_dict["components"] = components;

  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.dit_executor");
  py_executor_ =
      executor_module.attr("DiTPipelineExecutor").attr("create")(config_dict);
  LOG(INFO) << "PyDiTModel pipeline created: " << model_type_;
}

DiTForwardOutput PyDiTModel::forward(const DiTForwardInput& input) {
  torch::NoGradGuard no_grad;

  py::gil_scoped_acquire gil;
  CHECK(py_executor_) << "PyDiTModel pipeline is not initialized";

  const auto& gp = input.generation_params;
  py::dict request;
  request["prompts"] = py::cast(input.prompts);
  request["negative_prompts"] = py::cast(input.negative_prompts);
  request["batch_size"] = input.batch_size;
  request["images"] = py::none();
  py::list images_list;
  for (const auto& entry : input.image_sources.entries()) {
    images_list.append(py::cast(entry.tensor));
  }
  request["images_list"] = images_list;
  const auto prompt_embeds = input.tensor_sources.get("prompt_embed");
  request["prompt_embeds"] =
      prompt_embeds.has_value() ? py::cast(*prompt_embeds) : py::none();
  const auto negative_prompt_embeds =
      input.tensor_sources.get("negative_prompt_embed");
  request["negative_prompt_embeds"] = negative_prompt_embeds.has_value()
                                          ? py::cast(*negative_prompt_embeds)
                                          : py::none();
  const auto latents = input.tensor_sources.get("latent");
  request["latents"] = latents.has_value() ? py::cast(*latents) : py::none();

  py::dict gp_dict;
  gp_dict["width"] = gp.width;
  gp_dict["height"] = gp.height;
  gp_dict["num_inference_steps"] = gp.num_inference_steps;
  gp_dict["true_cfg_scale"] = gp.true_cfg_scale;
  gp_dict["guidance_scale"] = gp.guidance_scale;
  gp_dict["num_images_per_prompt"] = gp.num_images_per_prompt;
  gp_dict["seed"] = gp.seed;
  gp_dict["max_sequence_length"] = gp.max_sequence_length;
  gp_dict["strength"] = gp.strength;
  // Video / audio generation params consumed by video pipelines (JoyOV2
  // T2VA). Image pipelines ignore them.
  gp_dict["num_frames"] = gp.num_frames;
  gp_dict["video_fps"] = gp.video_fps;
  gp_dict["flow_shift"] = gp.flow_shift;
  gp_dict["audio_sampling_rate"] = gp.audio_sampling_rate;
  request["generation_params"] = gp_dict;

  py::object result = py_executor_.attr("execute")(request);

  DiTForwardOutput output;
  if (py::isinstance<py::dict>(result)) {
    // Pipelines with secondary outputs (JoyOV2 video+audio) return
    // {"tensors": [...], "audio_tensors": [...]}; plain tensor lists keep the
    // original single-output contract.
    if (result.contains("tensors")) {
      output.tensors = result["tensors"].cast<std::vector<torch::Tensor>>();
    }
    if (result.contains("audio_tensors")) {
      output.audio_tensors =
          result["audio_tensors"].cast<std::vector<torch::Tensor>>();
    }
  } else {
    output.tensors = result.cast<std::vector<torch::Tensor>>();
  }
  return output;
}

}  // namespace xllm
