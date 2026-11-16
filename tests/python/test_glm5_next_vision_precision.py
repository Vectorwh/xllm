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

"""Regression coverage for GLM vision frequencies and single-round arithmetic."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xllm.python.models.glm5_next import Glm5NextConfig, Glm5NextMLP
from xllm.python.models.glm5_next_vl import (
    Glm5NextVisionConfig,
    Glm5NextVisionModel,
    VisionBlock,
    VisionRMSNorm,
    VisionRotaryEmbedding,
    VisionRowParallelLinear,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vision_frequencies_survive_model_dtype_conversion(dtype: torch.dtype) -> None:
    layer = VisionRotaryEmbedding(32)
    expected = layer.inv_freq.clone()
    parent = nn.Sequential(layer).to(dtype=dtype)
    assert layer.inv_freq.dtype == torch.float32
    torch.testing.assert_close(layer.inv_freq, expected, rtol=0, atol=0)
    positions = torch.tensor([[1023, 997], [47, 31]])
    expected_angles = (positions.unsqueeze(-1) * expected).flatten(1)
    torch.testing.assert_close(layer(positions), expected_angles, rtol=0, atol=0)
    parent.float().to(dtype=dtype)
    torch.testing.assert_close(layer.inv_freq, expected, rtol=0, atol=0)


def test_vision_rms_norm_rounds_after_weight_scaling() -> None:
    generator = torch.Generator().manual_seed(42)
    value = torch.randn(32, 64, generator=generator).bfloat16()
    layer = VisionRMSNorm(64).bfloat16()
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(0.5, 1.5, 64).bfloat16())
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + layer.variance_epsilon)
    expected = (normalized * layer.weight.float()).to(value.dtype)
    old_result = normalized.to(value.dtype) * layer.weight
    assert not torch.equal(old_result, expected)
    torch.testing.assert_close(layer(value), expected, rtol=0, atol=0)


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_vision_residual_norm_uses_unrounded_sum(eps: float, monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python import kernels

    def _add_norm(
        value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = value.float() + residual.float()
        normalized = total * torch.rsqrt(total.square().mean(-1, keepdim=True) + eps)
        return (normalized * weight.float()).to(value.dtype), total.to(value.dtype)

    monkeypatch.setattr(kernels, "fused_add_rms_norm", _add_norm, raising=False)
    cfg = Glm5NextVisionConfig(hidden_size=32, num_heads=4, intermediate_size=64, rms_norm_eps=eps)
    block = VisionBlock(cfg).bfloat16()
    assert block.norm1.variance_epsilon == block.norm2.variance_epsilon == eps
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(16, 32, generator=generator).bfloat16()
    attention = torch.randn(16, 32, generator=generator).bfloat16()
    monkeypatch.setattr(block.attn, "forward", lambda *args, **kwargs: attention)
    monkeypatch.setattr(block.mlp, "forward", lambda tensor: tensor)
    output = block(value, torch.tensor([0, 16]), (torch.empty(0), torch.empty(0)))
    normalized, residual = _add_norm(value, attention, block.norm2.weight, eps)
    torch.testing.assert_close(output, residual + normalized, rtol=0, atol=0)
    rounded_total = (value + attention).float()
    rounded_norm = (rounded_total * torch.rsqrt(rounded_total.square().mean(-1, keepdim=True) + eps)).bfloat16()
    assert not torch.equal(rounded_norm, normalized)


def test_dense_swiglu_rounds_after_activation_and_multiply() -> None:
    cfg = Glm5NextConfig(hidden_size=8, tp_size=1)
    layer = Glm5NextMLP(cfg, intermediate_size=4, dtype=torch.bfloat16, device=torch.device("cpu"))
    layer.gate_up_proj = nn.Identity()
    layer.down_proj = nn.Identity()
    generator = torch.Generator().manual_seed(42)
    value = (torch.randn(64, 8, generator=generator) * 12).bfloat16()
    gate, up = value.chunk(2, -1)
    gate = gate.clamp(max=cfg.swiglu_limit)
    up = up.clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
    expected = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
    old_result = gate * gate.sigmoid() * up
    assert not torch.equal(old_result, expected)
    torch.testing.assert_close(layer(value), expected, rtol=0, atol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_vision_row_bias_is_added_before_reduction(rank: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python import distributed

    layer = VisionRowParallelLinear(2, 2, 2, tp_rank=rank, bias=True).bfloat16()
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.fill_(0.00390625)
    value = torch.ones(1, 2, dtype=torch.bfloat16)
    partial = torch.nn.functional.linear(value, layer.weight, layer.bias if rank == 0 else None)

    def _all_reduce(output: torch.Tensor) -> None:
        torch.testing.assert_close(output, partial, rtol=0, atol=0)
        output.add_(0.5)

    monkeypatch.setattr(distributed, "tp_all_reduce", _all_reduce, raising=False)
    torch.testing.assert_close(layer(value), partial + 0.5, rtol=0, atol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_vision_loader_packs_each_gate_up_shard(rank: int) -> None:
    cfg = Glm5NextVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        out_hidden_size=16,
        projection_intermediate_size=32,
        tp_size=2,
        tp_rank=rank,
    )
    full_cfg = Glm5NextVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        out_hidden_size=16,
        projection_intermediate_size=32,
    )
    full_model = Glm5NextVisionModel(full_cfg, dtype=torch.float32, device=torch.device("cpu"))
    weights: dict[str, torch.Tensor] = {}
    for name, parameter in full_model.named_parameters():
        tensor = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        if "gate_up_proj" in name:
            gate, up = tensor.chunk(2, dim=0)
            weights[name.replace("gate_up_proj", "gate_proj")] = gate
            weights[name.replace("gate_up_proj", "up_proj")] = up + 10000
        else:
            weights[name] = tensor

    class _StateDict:
        def has(self, name: str) -> bool:
            return name.removeprefix("model.visual.") in weights

        def get_tensor(self, name: str) -> torch.Tensor:
            return weights[name.removeprefix("model.visual.")]

    model = Glm5NextVisionModel(cfg, dtype=torch.float32, device=torch.device("cpu"))
    model.load_weights([_StateDict()])
    for prefix in ["blocks.0.mlp", "merger"]:
        for suffix in ["weight", "bias"]:
            gate_name = f"{prefix}.gate_proj.{suffix}"
            if gate_name not in weights:
                continue
            expected = torch.cat(
                [
                    weights[gate_name].chunk(2, dim=0)[rank],
                    weights[f"{prefix}.up_proj.{suffix}"].chunk(2, dim=0)[rank],
                ]
            )
            actual = model.get_parameter(f"{prefix}.gate_up_proj.{suffix}")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        model.merger.proj.weight,
        weights["merger.proj.weight"].chunk(2, dim=0)[rank],
        rtol=0,
        atol=0,
    )
