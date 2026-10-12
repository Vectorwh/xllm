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

"""ResidualCache block-level DiT cache (port of ``dit_cache/residual_cache.cpp``).

Reuses the residual between hidden states and encoder hidden states for the
middle blocks of a step, bounded by start/end step and block windows.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import DitCachePolicy, register_cache_policy


class ResidualCache(DitCachePolicy):
    def __init__(self, sp_group: str | None, options: dict[str, Any] | None) -> None:
        super().__init__(sp_group, options)
        self.skip_interval_steps = int(options.get("skip_interval_steps", 3))
        self.start_steps = int(options.get("start_steps", 5))
        self.end_steps = int(options.get("end_steps", 5))
        self.start_blocks = int(options.get("start_blocks", 5))
        self.end_blocks = int(options.get("end_blocks", 5))
        self.current_step = 0
        self.use_cache = False
        self.update_cache = False
        self.buffers: dict[str, torch.Tensor] = {}

    def reset_cache(self) -> None:
        self.use_cache = False
        self.update_cache = False

    def get_residual(self, hidden: torch.Tensor, key: str) -> torch.Tensor:
        return hidden - self.buffers[key]

    def add_residual(self, hidden: torch.Tensor, key: str) -> torch.Tensor:
        return hidden + self.buffers[key]

    def update(self, residual: torch.Tensor, key: str) -> None:
        self.buffers[key] = residual

    def cache_validation(self) -> bool:
        step_valid = (
            self.infer_steps > self.start_steps + self.end_steps
            and self.infer_steps > self.start_steps
            and self.infer_steps > self.end_steps
        )
        block_valid = (
            self.num_blocks > self.start_blocks + self.end_blocks
            and self.num_blocks > self.start_blocks
            and self.num_blocks > self.end_blocks
        )
        return step_valid and block_valid

    def on_before_block(self, block_index: int) -> bool:
        if (
            not self.cache_validation()
            or not self.use_cache
            or block_index < self.start_blocks
            or block_index >= self.num_blocks - self.end_blocks - 1
        ):
            return False
        return True

    def on_after_block(self, block_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        hidden = tensors.get("hidden_states")
        encoder = tensors.get("encoder_hidden_states")

        if self.cache_validation():
            if self.use_cache:
                if block_index == self.num_blocks - self.end_blocks - 1:
                    out["hidden_states"] = self.add_residual(hidden, "hidden_states")
                    out["encoder_hidden_states"] = self.add_residual(encoder, "encoder_hidden_states")
                    return out
            elif self.update_cache:
                if block_index == self.start_blocks - 1:
                    self.update(hidden.clone(), "hidden_states")
                    self.update(encoder.clone(), "encoder_hidden_states")
                elif block_index == self.num_blocks - self.end_blocks - 1:
                    self.update(self.get_residual(hidden, "hidden_states"), "hidden_states")
                    self.update(self.get_residual(encoder, "encoder_hidden_states"), "encoder_hidden_states")

        out["hidden_states"] = hidden
        out["encoder_hidden_states"] = encoder
        return out

    def on_before_step(self, step_index: int, tensors: dict[str, torch.Tensor]) -> bool:
        del tensors
        self.current_step = step_index

        if (
            not self.cache_validation()
            or self.current_step < self.start_steps - 1
            or self.current_step >= self.infer_steps - self.end_steps
        ):
            self.reset_cache()
            return False

        self.use_cache = (self.current_step - (self.start_steps - 1)) % self.skip_interval_steps != 0
        self.update_cache = not self.use_cache
        return False

    def on_after_step(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return tensors


register_cache_policy(
    "ResidualCache",
    "xllm.python.model_executor.dit_cache.residual_cache",
    "ResidualCache",
)
