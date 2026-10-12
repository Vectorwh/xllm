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

"""TaylorSeer step-level DiT cache (port of ``dit_cache/taylorseer.cpp``).

Approximates the hidden states at skipped steps with a Taylor series over the
derivatives accumulated at the last non-cached step.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .base import DitCachePolicy, register_cache_policy


class TaylorSeer(DitCachePolicy):
    def __init__(self, sp_group: str | None, options: dict[str, Any] | None) -> None:
        super().__init__(sp_group, options)
        self.n_derivatives = int(options.get("n_derivatives", 3))
        self.skip_interval_steps = int(options.get("skip_interval_steps", 3))
        self.order = self.n_derivatives + 1
        self.dY_prev: list[torch.Tensor | None] = [None] * self.order
        self.dY_current: list[torch.Tensor | None] = [None] * self.order
        self.valid_prev: list[bool] = [False] * self.order
        self.valid_current: list[bool] = [False] * self.order
        self.current_step = 1
        self.last_non_approximated_step = 1
        self.use_cache = False

    def reset_cache(self) -> None:
        self.dY_prev = [None] * self.order
        self.dY_current = [None] * self.order
        self.valid_prev = [False] * self.order
        self.valid_current = [False] * self.order
        self.current_step = 1
        self.last_non_approximated_step = 1
        self.use_cache = False

    def mark_step_begin(self) -> None:
        self.current_step += 1

    def approximate_derivative(self, Y: torch.Tensor) -> tuple[list[torch.Tensor | None], list[bool]]:
        dY: list[torch.Tensor | None] = [None] * self.order
        valid: list[bool] = [False] * self.order
        dY[0] = Y
        valid[0] = True

        elapsed_steps = self.current_step - self.last_non_approximated_step
        if elapsed_steps <= 0:
            return dY, valid

        for i in range(self.n_derivatives):
            if i >= self.order or not self.valid_prev[i] or self.dY_prev[i] is None:
                break
            prev = self.dY_prev[i]
            if prev is None or dY[i] is None:
                break
            dY[i + 1] = (dY[i] - prev) / float(elapsed_steps)
            valid[i + 1] = True
        return dY, valid

    def approximate_value(self) -> torch.Tensor | None:
        if self.dY_current[0] is None:
            return None
        elapsed_steps = self.current_step - self.last_non_approximated_step
        if elapsed_steps < 0:
            return None

        output = torch.zeros_like(self.dY_current[0])
        for i in range(self.order):
            if not self.valid_current[i] or self.dY_current[i] is None:
                break
            coefficient = (elapsed_steps**i) / math.factorial(i)
            output = output + self.dY_current[i] * coefficient
        return output

    def update(self, Y: torch.Tensor) -> None:
        self.dY_prev = self.dY_current
        self.valid_prev = self.valid_current

        if self.dY_current[0] is None:
            self.dY_current = [None] * self.order
            self.valid_current = [False] * self.order
            self.dY_current[0] = Y
            self.valid_current[0] = True
        else:
            self.dY_current, self.valid_current = self.approximate_derivative(Y)
        self.last_non_approximated_step = self.current_step

    def on_before_block(self, block_index: int) -> bool:
        del block_index
        return False

    def on_after_block(self, block_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del block_index
        return {
            "hidden_states": tensors.get("hidden_states"),
            "encoder_hidden_states": tensors.get("encoder_hidden_states"),
        }

    def on_before_step(self, step_index: int, tensors: dict[str, torch.Tensor]) -> bool:
        del tensors
        self.current_step = step_index

        if step_index == 1 or step_index <= self.warmup_steps:
            if step_index == 1:
                self.reset_cache()
            self.use_cache = False
            return self.use_cache

        self.use_cache = (step_index - self.warmup_steps + 1) % self.skip_interval_steps != 0
        return self.use_cache

    def on_after_step(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.use_cache:
            self.update(tensors["hidden_states"])
            return dict(tensors)
        return {"hidden_states": self.approximate_value()}


register_cache_policy("TaylorSeer", "xllm.python.model_executor.dit_cache.taylorseer", "TaylorSeer")
