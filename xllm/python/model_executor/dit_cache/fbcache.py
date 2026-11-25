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

"""FBCache block-level DiT cache (port of ``dit_cache/fbcache.cpp``).

Skips the remaining blocks of a step when the first block's residual is close
enough to the cached one; the skipped blocks' output is reconstructed by adding
the cached residual to the current inputs.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import DitCachePolicy, register_cache_policy


class FBCache(DitCachePolicy):
    def __init__(self, sp_group: str | None, options: dict[str, Any] | None) -> None:
        super().__init__(sp_group, options)
        self.residual_diff_threshold = float(options.get("residual_diff_threshold", 0.09))
        self.current_step = 0
        self.use_cache = False
        self.buffers: dict[str, torch.Tensor] = {}

    def on_before_block(self, block_index: int) -> bool:
        del block_index
        return self.use_cache

    def on_after_block(self, block_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = tensors.get("hidden_states")
        encoder = tensors.get("encoder_hidden_states")
        if block_index != 0 or "encoder_hidden_states" not in tensors:
            return {"hidden_states": hidden, "encoder_hidden_states": encoder}

        original_hidden = tensors.get("original_hidden_states")
        first_hidden_residual = hidden - original_hidden

        if self.can_use_cache(first_hidden_residual):
            self.use_cache = True
            output_hidden, output_encoder = self.apply_prev_hidden_states_residual(original_hidden, encoder)
        else:
            self.use_cache = False
            self.buffers["first_hidden_states_residual"] = first_hidden_residual
            output_hidden, output_encoder = hidden, encoder

        return {"hidden_states": output_hidden, "encoder_hidden_states": output_encoder}

    def on_before_step(self, step_index: int, tensors: dict[str, torch.Tensor]) -> bool:
        del tensors
        self.current_step = step_index
        self.use_cache = False
        if step_index == 1:
            self.buffers.clear()
        return False

    def on_after_step(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.use_cache:
            hidden = tensors.get("hidden_states")
            original_hidden = tensors.get("original_hidden_states")
            self.buffers["hidden_states_residual"] = hidden - original_hidden

            encoder = tensors.get("encoder_hidden_states")
            original_encoder = tensors.get("original_encoder_hidden_states")
            if encoder is not None and original_encoder is not None:
                self.buffers["encoder_hidden_states_residual"] = encoder - original_encoder

        return {"hidden_states": tensors.get("hidden_states")}

    def apply_prev_hidden_states_residual(
        self, hidden: torch.Tensor | None, encoder: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        residual = self.buffers.get("hidden_states_residual")
        if residual is not None and hidden is not None:
            return (hidden + residual).contiguous(), encoder
        return hidden, encoder

    def can_use_cache(self, first_hidden_residual: torch.Tensor | None) -> bool:
        enabled = self.current_step > self.warmup_steps and self.residual_diff_threshold > 0.0
        prev = self.buffers.get("first_hidden_states_residual") if enabled else None
        # A missing/disabled cache on any SP rank rejects reuse for the group.
        return self.is_similar(prev, first_hidden_residual, self.residual_diff_threshold)


register_cache_policy("FBCache", "xllm.python.model_executor.dit_cache.fbcache", "FBCache")
