# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression coverage for clamped shared experts without grouped routing."""

from __future__ import annotations

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize(("alpha", "bias"), [(1.0, 0.0), (1.702, 1.0)])
@torch.inference_mode()
def test_shared_expert_dequant_swiglu_honors_clamp(alpha: float, bias: float, rows: int) -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "dequant_swiglu_quant"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    gate = torch.tensor([20, -20, 2, 0], dtype=torch.int32).repeat(rows, 32)
    up = torch.tensor([5, -5, 1, 2], dtype=torch.int32).repeat(rows, 32)
    accumulators = torch.cat((gate, up), dim=-1).npu()
    scale = torch.ones(accumulators.shape[-1], device="npu")
    activation_scale = torch.ones(rows, device="npu")
    quantized, output_scale = torch.ops.xllm_ops.dequant_swiglu_quant(
        accumulators,
        scale,
        activation_scale,
        None,
        None,
        None,
        None,
        True,
        1,
        1,
        1.0,
        alpha,
        bias,
    )
    clipped_gate = gate.float().clamp(max=1.0)
    values = clipped_gate * torch.sigmoid(alpha * clipped_gate) * (up.float().clamp(-1.0, 1.0) + bias)
    expected_scale = values.abs().amax(-1) / 127.0
    expected = (values / expected_scale[:, None]).round().to(torch.int8)
    torch.testing.assert_close(quantized.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(output_scale.cpu(), expected_scale, rtol=1e-6, atol=0)
