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

from typing import Optional

import pytest
import torch
from torch import nn

from xllm.python.layers.qlinear import QLinear, QLinearWeightLoader
from xllm.python.models import deepseek_v32
from xllm.python.models.glm5_next import Glm5NextForCausalLM


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


@pytest.mark.parametrize("quant_format", ["static", "dynamic"])
@pytest.mark.parametrize("shard_dim", [None, 0, 1])
@pytest.mark.parametrize("rank", [0, 1])
def test_attention_quant_format_and_tp_shards(
    quant_format: str, shard_dim: Optional[int], rank: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CPU tests provide the quantized-matmul layout without initializing NPU ops.
    monkeypatch.setattr(
        deepseek_v32.kernels, "prepare_quant_weight", lambda weight: weight.T.contiguous(), raising=False
    )
    weight = torch.arange(32, dtype=torch.int8).reshape(8, 4)
    scales = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
    tensors = {"projection.weight": weight}
    if quant_format == "dynamic":
        tensors["projection.weight_scale"] = scales
    else:
        tensors.update(
            {
                "projection.deq_scale": scales.flatten(),
                "projection.quant_bias": torch.arange(8, dtype=torch.int32),
                "projection.input_scale": torch.ones(1),
                "projection.input_offset": torch.zeros(1),
            }
        )
    model = nn.Module()
    model.projection = QLinear(
        2 if shard_dim == 1 else 4,
        4 if shard_dim == 0 else 8,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kind="static",
        row_parallel=shard_dim == 1,
    )
    loader = QLinearWeightLoader(model, [_StateDict(tensors)], 2, rank)
    Glm5NextForCausalLM._load_qlinear(model, loader, "", "projection", shard_dim)
    expected = weight if shard_dim is None else weight.chunk(2, dim=shard_dim)[rank]
    torch.testing.assert_close(model.projection._w8a8.weight, expected.transpose(0, 1).contiguous())
    assert model.projection.kind == quant_format
    assert model.projection.weight.numel() == 0
    if quant_format == "dynamic":
        expected_scales = scales.chunk(2, dim=0)[rank].flatten() if shard_dim == 0 else scales.flatten()
        torch.testing.assert_close(model.projection._w8a8.weight_scale, expected_scales)
        assert torch.count_nonzero(model.projection._w8a8.weight_offset) == 0
    else:
        expected_scales = scales.flatten()
        if shard_dim == 0:
            expected_scales = expected_scales.chunk(2)[rank]
        torch.testing.assert_close(model.projection._w8a8.deq_scale, expected_scales)


def test_dynamic_qlinear_prequantized_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    def quant_matmul(
        x: torch.Tensor,
        weight: torch.Tensor,
        transpose: bool,
        weight_scale: torch.Tensor,
        _offset: object,
        pertoken: Optional[torch.Tensor],
        _bias: object,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        accumulated = x.int() @ (weight.t() if transpose else weight).int()
        if dtype == torch.int32:
            return accumulated
        return (accumulated.float() * weight_scale * pertoken[:, None]).to(dtype)

    monkeypatch.setattr(deepseek_v32.kernels, "quant_matmul", quant_matmul, raising=False)
    projection = QLinear(4, 2, device=torch.device("cpu"), dtype=torch.bfloat16, kind="dynamic")
    projection.resolve_quant(True)
    projection._w8a8.weight.data.copy_(torch.arange(8, dtype=torch.int8).reshape(2, 4))
    projection._w8a8.weight_scale.data = torch.tensor([0.5, 0.25])
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.int8)
    scales = torch.tensor([0.125])
    expected = x.int() @ projection._w8a8.weight.t().int()
    torch.testing.assert_close(projection.forward_accumulated(x), expected)
    torch.testing.assert_close(
        projection.forward_quantized(x, scales),
        (expected.float() * projection.weight_scale * scales[:, None]).bfloat16(),
    )
