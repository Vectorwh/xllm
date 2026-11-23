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

"""Qwen3.5 multimodal model for the Python executor.

Composition mirrors the native ``Qwen3VLForConditionalGenerationBase`` in
``models/vlm/qwen3_vl_base.h``: a Qwen3-VL vision tower feeding image/video
embeddings into the Qwen3.5 hybrid-attention language model.

Unlike Qwen3-VL, qwen3.5 checkpoints carry no DeepStack visual indexes (the
native loader rejects a non-empty ``vision_config.deepstack_visual_indexes``
with ``LOG(FATAL)``), so the tower runs with an empty deepstack list and
emits only the main merger output.
"""

from __future__ import annotations

import torch

from xllm.python.layers import ColumnParallelLinear
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    load_causal_lm_weights,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.multimodal import (
    EncoderOutputs,
    EncoderTokenCounts,
    MultimodalData,
    MultimodalInputs,
    encode_batched_vision,
    merge_multimodal_embeddings,
)
from xllm.python.models.qwen3_5 import Qwen3_5Config, Qwen3_5Model
from xllm.python.models.qwen3_vl import (
    Qwen3VLVisionConfig,
    Qwen3VLVisionTransformer,
    load_qwen3_vl_vision_weights,
)


def _vision_config_from_dict(config: dict) -> Qwen3VLVisionConfig:
    """Read the vision tower config, rejecting the Qwen3-VL-only DeepStack list.

    ``Qwen3VLVisionConfig.from_dict`` defaults ``deepstack_visual_indexes`` to
    the Qwen3-VL value ``[8, 16, 24]``. Substituting an explicit empty list when
    the checkpoint omits the key keeps an absent field meaning "no DeepStack",
    matching the native loader's ``LOAD_ARG_OR(..., std::vector<int64_t>())``;
    without it a checkpoint that simply has no DeepStack would be rejected.
    """
    deepstack = config.get("mm_deepstack_visual_indexes", config.get("deepstack_visual_indexes"))
    if deepstack is None:
        config = {**config, "mm_deepstack_visual_indexes": []}
    vision_config = Qwen3VLVisionConfig.from_dict(config)
    if vision_config.deepstack_visual_indexes:
        raise ValueError(
            f"qwen3_5 VLM does not support DeepStack visual indexes, got {vision_config.deepstack_visual_indexes}"
        )
    return vision_config


class Qwen3_5ForConditionalGeneration(PyModelBase):
    """Qwen3.5 top-level model: vision tower + hybrid-attention LLM + head."""

    cache_multimodal_embeddings = True

    model: Qwen3_5Model
    lm_head: ColumnParallelLinear

    def __init__(self, config: dict) -> None:
        super().__init__()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        # Share the Qwen3.5 execution policy: the EPLv2 checks and the resolved
        # sequence-parallel flag have to match the text host, because the same
        # metadata builders consume `sequence_parallel_enabled` from `config`.
        # Mirror the text host's speculative opt-in; the underlying Qwen3.5
        # model owns AuxHiddenCapture and the executor forwards aux states.
        text_cfg = Qwen3_5Config.prepare_execution_config(config, device, allow_speculative_decoding=True)
        if config.get("rope_scaling_mrope_interleaved") is None:
            # The VLM arg loader defaults the interleaved mRoPE layout to true
            # (the text-only loader defaults it to false).
            text_cfg.rope_scaling_mrope_interleaved = True
        self.cfg = text_cfg

        vision_cfg = _vision_config_from_dict(config)
        self.dtype = dtype
        self.device = device

        self.vision_cfg = vision_cfg
        self.text_cfg = text_cfg

        self.vision_model = Qwen3VLVisionTransformer(vision_cfg, dtype=dtype, device=device)

        self.model = Qwen3_5Model(text_cfg, dtype, device)  # pyright: ignore[reportIncompatibleVariableOverride]
        self.lm_head = ColumnParallelLinear(  # pyright: ignore[reportIncompatibleVariableOverride]
            text_cfg.hidden_size,
            text_cfg.vocab_size // text_cfg.tp_size,
            text_cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    # ------------------------------------------------------------------
    # ViT -> LLM connection
    # ------------------------------------------------------------------

    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Vision encode: flattened patches -> ``[total_tokens, projection_dim]``."""
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype, device=self.vision_model.device)
        grid_thw = grid_thw.to(dtype=torch.int32, device=self.vision_model.device)
        return self.vision_model(pixel_values, grid_thw)

    def encode_multimodal(self, data: MultimodalData, token_counts: EncoderTokenCounts) -> EncoderOutputs:
        return encode_batched_vision(self.encode, data, token_counts)

    def prepare_multimodal_inputs(self, input_ids: torch.Tensor, data: MultimodalData) -> MultimodalInputs:
        inputs_embeds = self.model.embed_tokens(input_ids)
        modalities_with_embeds = {key.removesuffix("|embedding") for key in data if key.endswith("|embedding")}
        modalities_with_masks = {key.removesuffix("|mask") for key in data if key.endswith("|mask")}
        # The gather visitor emits each modality's features and mask as a pair.
        # Half a pair means the other half was dropped: an orphan mask would
        # leave the image as text tokens, and an orphan embedding would reach
        # merge_multimodal_embeddings as a bare KeyError on its missing mask.
        if modalities_with_embeds != modalities_with_masks:
            raise ValueError(
                "multimodal embeddings and masks disagree: "
                f"embeddings without masks {sorted(modalities_with_embeds - modalities_with_masks)}, "
                f"masks without embeddings {sorted(modalities_with_masks - modalities_with_embeds)}"
            )
        if modalities_with_embeds:
            inputs_embeds = merge_multimodal_embeddings(inputs_embeds, data)
        return MultimodalInputs(inputs_embeds=inputs_embeds)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        # The parameters were shaped from text_cfg.tp_size, so a caller that
        # disagrees would shard against the wrong geometry. The shape check in
        # load_tensor does catch it, but it names the tensor rather than the
        # cause; this is the C++ -> Python boundary, so check the contract here.
        if tp_size != self.text_cfg.tp_size:
            raise ValueError(
                f"load_weights tp_size {tp_size} disagrees with the model's tp_size {self.text_cfg.tp_size}"
            )
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
        self._load_vision_weights(ScopedWeightLoader(state_dicts, src_prefixes=("model.visual.",)))
        all_weights = ScopedWeightLoader(
            state_dicts,
            src_prefixes=("model.language_model.", "model.", ""),
        )
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            all_weights,
            ParallelLoadContext.from_config(self.text_cfg, tp_rank, tp_size),
            tie_word_embeddings=self.text_cfg.tie_word_embeddings,
            # Mirrors qwen3_5.py:488, the owner of this stack, which also hardcodes
            # True: the same checkpoint has to load whether it runs as text or VLM.
            embed_fallback=True,
        )

    def _load_vision_weights(self, weights: ScopedWeightLoader) -> None:
        """Load the vision tower; replicated across ranks like the native path."""
        load_qwen3_vl_vision_weights(
            self.vision_model,
            weights,
            depth=self.vision_cfg.depth,
            deepstack_levels=0,  # qwen3_5 checkpoints carry no DeepStack mergers
        )
