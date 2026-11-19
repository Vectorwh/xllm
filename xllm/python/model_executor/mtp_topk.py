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

from __future__ import annotations

import torch


class MtpTopkReuse:
    """Own the first draft's selection for one Python MTP iteration."""

    def __init__(self) -> None:
        self._indices: torch.Tensor | None = None
        self._last_step = -1

    def input_for_step(self, step: int, rows: int) -> torch.Tensor | None:
        if step == 0:
            self._indices = None
            self._last_step = -1
            return None
        if step < 0 or self._indices is None or self._last_step != step - 1:
            raise ValueError("MTP top-k reuse requires consecutive draft steps after step 0")
        if self._indices.shape[0] != rows:
            raise ValueError("MTP top-k request rows changed within a draft iteration")
        return self._indices

    def finish_step(self, step: int, indices: torch.Tensor, selected_rows: torch.Tensor | None) -> None:
        if step == 0:
            # Gather outside graph capture using the sampler's authoritative
            # rows, owning storage independent of the graph's output pool.
            self._indices = (
                indices.index_select(0, selected_rows.to(device=indices.device, dtype=torch.int64))
                if selected_rows is not None and selected_rows.numel() > 0
                else indices.clone()
            )
        self._last_step = step
