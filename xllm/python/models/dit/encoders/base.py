# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text encoder contract for DiT pipelines.

The three-method protocol mirrors the C++ pipeline call sites
(``pipeline_joyimage_edit_plus.h``): ``get_multimodal_embeddings`` (optional,
vision tower prefill), ``get_input_embeddings`` (token embedding lookup), and
``forward`` (full pass returning pre-norm hidden states in
``TextEncoderOutput.residual`` — the JoyImageEditPlus contract).

Two implementations exist:
  * form A — a native Python encoder registered below (CLIP / UMT5 / ...),
    created and loaded entirely in Python;
  * form B — a C++-owned VLM (e.g. Qwen3-VL with paged attention and TP)
    wrapped by ``vlm_encoder_proxy.h`` and injected as an external object.
  Form-B proxies additionally expose a high-level ``encode_prompts`` that
  keeps tokenization and multimodal preprocessing on the C++ side, so a
  pipeline crosses the language boundary once per request instead of once per
  tensor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import torch

_ENCODER_REGISTRY: dict[str, tuple[str, str]] = {}


@dataclass
class TextEncoderOutput:
    """Output of the text-encoder forward pass.

    ``last_hidden_state`` is the final hidden state;
    ``residual`` is the pre-norm hidden state required by JoyImageEditPlus.
    """

    last_hidden_state: torch.Tensor
    residual: torch.Tensor | None = None


class TextEncoderProtocol(Protocol):
    """Structural contract every DiT text encoder satisfies."""

    def get_input_embeddings(self, tokens: torch.Tensor, params: object) -> torch.Tensor: ...

    def get_multimodal_embeddings(self, params: object) -> torch.Tensor: ...

    def forward(
        self,
        input_ids: torch.Tensor,
        params: object,
    ) -> TextEncoderOutput: ...


def register_dit_encoder(*names: str) -> Callable[[type], type]:
    """Register a native (form-A) encoder class under the given names."""

    def deco(cls: type) -> type:
        for name in names:
            _ENCODER_REGISTRY[name] = (cls.__module__, cls.__name__)
        return cls

    return deco


def create_text_encoder(spec: dict) -> object:
    """Build a text encoder from an executor spec dict.

    ``spec["kind"] == "external"`` returns the C++ proxy object as-is;
    otherwise the spec must carry ``class_name`` (a registered encoder),
    ``config`` (its config dict) and ``state_dicts`` (C++ StateDict wrappers).
    """
    kind = spec.get("kind", "native")
    if kind == "external":
        proxy = spec["proxy"]
        if proxy is None:
            raise ValueError("external text_encoder spec carries no proxy object")
        return proxy

    class_name = spec["class_name"]
    if class_name not in _ENCODER_REGISTRY:
        raise KeyError(f"DiT text encoder '{class_name}' not registered; available: {sorted(_ENCODER_REGISTRY)}")
    module_name, attribute = _ENCODER_REGISTRY[class_name]
    encoder_cls = getattr(import_module(module_name), attribute)
    encoder = encoder_cls(spec["config"])
    encoder.load_weights(spec["state_dicts"])
    return encoder
