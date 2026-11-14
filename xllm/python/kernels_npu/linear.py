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

"""NPU weight preparation for linear layers."""

from __future__ import annotations

import torch
import torch_npu

from xllm.python.kernels_npu.constants import FRACTAL_NZ_FORMAT
from xllm.python.kernels_npu.weight_layout import format_cast_nz


def prepare_row_parallel_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Lay out a row-parallel weight for the NPU matmul kernels.

    The matmul kernels read ``[K, N]`` in the fractal-NZ format, so the
    checkpoint layout is transposed and cast. A weight still on the host is
    returned untouched: the format cast needs device memory and runs after the
    weight is moved.

    Args:
        weight: Row-parallel weight of shape ``[N, K]``.

    Returns:
        The weight and whether it was transposed to ``[K, N]``.
    """
    if weight.device.type == "cpu":
        return weight, False
    transposed = weight.transpose(0, 1).contiguous()
    return torch_npu.npu_format_cast(transposed, FRACTAL_NZ_FORMAT), True


def prepare_quant_weight(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Pack an INT8 ``[out, in]`` weight for quantized matmul.

    ``torch_npu.npu_quant_matmul`` reads the right-hand operand as ``[K, N]``
    and benefits substantially from the NPU Fractal-NZ layout. Quantized Python
    linear layers use the same layout as the checkpoint before this helper is
    called, so keep the transpose and format conversion together here.  The
    CPU branch is intentionally a plain transpose for unit tests and weight
    loading without an initialized NPU.
    """
    transposed = weight.transpose(0, 1).contiguous()
    if weight.device.type == "cpu":
        return transposed
    return torch_npu.npu_format_cast(transposed, FRACTAL_NZ_FORMAT)


def prepare_weights_nz(weight: torch.Tensor) -> torch.Tensor:
    """Cast one NPU matmul weight into the fractal-NZ layout.

    Callers own the eligibility policy: dense BF16 weights pack here, while
    INT8 weights follow ``prepare_quant_weight`` instead. This helper only
    applies the layout change, so callers gate on device and dtype before
    calling. A matrix with N == 1 or K == 1 is returned unchanged because
    ``aclnnMatmulWeightNz`` rejects it.
    """
    if min(weight.shape[-2:]) <= 1:
        return weight
    return format_cast_nz(weight)


__all__ = [
    "prepare_quant_weight",
    "prepare_row_parallel_weight",
    "prepare_weights_nz",
]
