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

"""Fractal-NZ weight-layout primitive shared by the NPU kernel modules.

``torch_npu.npu_format_cast`` only accepts FRACTAL_NZ as a "private" internal
format, so every NZ conversion in this package must first enable
``torch.npu.config.allow_internal_format``. Centralizing both the toggle and
the cast here keeps the internal-format policy identical for linear and MoE
weights instead of split per call site.
"""

from __future__ import annotations

import torch
import torch_npu

from xllm.python.kernels_npu.constants import FRACTAL_NZ_FORMAT


def _enable_internal_format() -> None:
    """Allow the private NPU formats required by the NZ cast."""
    torch.npu.config.allow_internal_format = True


def format_cast_nz(weight: torch.Tensor) -> torch.Tensor:
    """Cast one weight tensor to the fractal-NZ format.

    Idempotent: a tensor already stored in FRACTAL_NZ is returned unchanged.
    """
    _enable_internal_format()
    if torch_npu.get_npu_format(weight) == FRACTAL_NZ_FORMAT:
        return weight
    return torch_npu.npu_format_cast(weight, FRACTAL_NZ_FORMAT)


__all__ = ["format_cast_nz"]
