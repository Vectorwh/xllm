# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test V4.1 candidate-block expansion and W8A8 linear dispatch contracts.

Extracted from the monolithic ``test_deepseek_v41_fused_ops.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
import xllm.python.models.deepseek_v41 as v41

# ---------------------------------------------------------------------------
# R1: indexer candidate-block expansion
# ---------------------------------------------------------------------------


def _select_candidate_blocks_repeat_interleave(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """The legacy ``repeat_interleave``-based expansion, as a frozen reference."""
    width = score.size(-1)
    padded = torch.nn.functional.pad(score, (0, -width % block_size), value=float("-inf"))
    blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = blocks.size(-1)
    last = torch.div(compress_lens - 1, block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(
        torch.arange(num_blocks, device=score.device) == last.unsqueeze(-1),
        float("inf"),
    )
    top = blocks.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def test_indexer_candidate_expand_matches_repeat_interleave() -> None:
    """unsqueeze->expand->flatten is bitwise the legacy repeat_interleave."""
    generator = torch.Generator().manual_seed(3)
    # 37 columns -> 10 padded blocks of 4 (last block partially padded).
    score = torch.randn(12, 37, generator=generator)
    compress_lens = torch.tensor([6, 12, 25, 31, 37, 1, 2, 3, 40, 9, 10, 11])

    fused = csa2._select_candidate_blocks(score, compress_lens, topk_blocks=3, block_size=4)
    legacy = _select_candidate_blocks_repeat_interleave(score, compress_lens, topk_blocks=3, block_size=4)

    assert torch.equal(fused, legacy)
    assert fused.shape == score.shape

    # block_size larger than the width and an exact-multiple width.
    small = torch.randn(3, 5, generator=generator)
    lens = torch.tensor([0, 3, 5])
    small_fused = csa2._select_candidate_blocks(small, lens, topk_blocks=2, block_size=8)
    small_legacy = _select_candidate_blocks_repeat_interleave(small, lens, topk_blocks=2, block_size=8)
    assert torch.equal(small_fused, small_legacy)

    exact = torch.randn(3, 8, generator=generator)
    exact_lens = torch.tensor([0, 4, 8])
    exact_fused = csa2._select_candidate_blocks(exact, exact_lens, topk_blocks=2, block_size=4)
    exact_legacy = _select_candidate_blocks_repeat_interleave(exact, exact_lens, topk_blocks=2, block_size=4)
    assert torch.equal(exact_fused, exact_legacy)


# ---------------------------------------------------------------------------
# R2: merged attention-input quant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device_type", "dimensions", "expected"),
    [
        ("npu", 2, True),
        ("npu", 3, False),
        ("privateuseone", 2, True),
        ("cpu", 2, False),
    ],
)
def test_merged_attn_quant_guard_is_strict(device_type: str, dimensions: int, expected: bool) -> None:
    x = SimpleNamespace(device=SimpleNamespace(type=device_type), dim=lambda: dimensions)
    assert v41._merged_attn_quant_supported(x) is expected


def test_w8a8_dynamic_linear_eager_and_quantized_paths_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eager dynamic-quant route matches the prequantized projection input."""
    import xllm.python.kernels as kernels_ns

    from xllm.python.models.deepseek_v32 import W8A8DynamicLinear

    quant_calls: list[torch.Tensor] = []

    def fake_dynamic_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quant_calls.append(value.clone())
        scale = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
        quantized = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
        return quantized, scale

    def fake_quant_matmul(
        x1: torch.Tensor,
        x2: torch.Tensor,
        transpose2: bool,
        scale: torch.Tensor,
        offset: torch.Tensor | None,
        pertoken: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del offset, bias, output_dtype
        weight = x2.float()
        out = x1.float() @ (weight.t() if transpose2 else weight)
        return (out * scale.view(1, -1) * pertoken.view(-1, 1)).to(torch.bfloat16)

    monkeypatch.setattr(kernels_ns, "dynamic_quant", fake_dynamic_quant, raising=False)
    monkeypatch.setattr(kernels_ns, "quant_matmul", fake_quant_matmul, raising=False)

    generator = torch.Generator().manual_seed(11)
    proj = W8A8DynamicLinear(8, 4, torch.device("cpu"), transpose_weight_after_loading=False)
    with torch.no_grad():
        proj.weight.copy_(torch.randint(-127, 127, (4, 8), dtype=torch.int8, generator=generator))
        proj.weight_scale.copy_(torch.ones(4, 1))
        proj.weight_offset.zero_()

    x = (torch.randn(6, 8, generator=generator) * 0.7).to(torch.bfloat16)

    quant_calls.clear()
    quantized_input, input_scale = fake_dynamic_quant(x)
    merged = proj.forward_quantized(quantized_input, input_scale)
    assert len(quant_calls) == 1

    quant_calls.clear()
    fallback = proj(x)
    assert len(quant_calls) == 1
    assert torch.equal(merged, fallback)
