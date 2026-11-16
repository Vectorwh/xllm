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

"""Golden layout and numerical tests for DeepSeek-V4.1 image embeddings.

LOCKED PAIR twin of ``tests/processors/deepseek_v41_prompt_processor_test.cpp``:
both pin the image-block role layout to the same golden masks, so the Python
``_vision_block_slots`` (consumed by ``encode``) and the C++
``DeepseekV41PromptProcessor::image_block_roles`` (consumed by ``process`` /
``adjust_token_layout`` / ``find_mm_spans``) cannot drift apart. Any layout
change must regenerate both golden tables in the same commit.

The ten numerical values are pinned together with deepseek_v41_vision_test.
They come from the production Python tower with CPU substitutes at NPU kernel
boundaries; this checks model semantics and BF16 rounding, not NPU kernels.

Pure-Python: does not load compiled operators.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.models.deepseek_v41 import (
    _IMAGE,
    _IMAGE_END,
    _IMAGE_NEWLINE,
    _IMAGE_PAD,
    _IMAGE_START,
    DeepseekV41Aligner,
    DeepseekV41ForCausalLM,
    DeepseekV41VisionConfig,
    DeepseekV41VisionTower,
    _vision_block_slots,
)

# (n_vit_h, n_vit_w, golden mask) triples. The mask is the per-image block
# role sequence: 'S' for a sentinel slot (any non-pad role, token 129264) and
# 'P' for a compressor-alignment pad (token 129265). n_llm = ceil(n_vit / 3).
LAYOUT_CASES = [
    (6, 6, "SSSSSSSPPS"),
    (9, 9, "SSSSSSSSSSPSPSPSPS"),
    (3, 3, "SSPSPS"),
    (6, 15, "SSSSSSSSSSSSSS"),
    (7, 8, "SSSSSSSSSSPSPSPSPS"),
    (12, 9, "SSSSSSSSSSSSSSSSSS"),
    (9, 3, "SSSSSSPSPS"),
    (5, 5, "SSSSSSSPPS"),
]


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


@pytest.mark.parametrize(("n_vit_h", "n_vit_w", "mask"), LAYOUT_CASES)
def test_block_slots_match_cpp_golden(n_vit_h: int, n_vit_w: int, mask: str) -> None:
    n_llm_h = _ceil_div(n_vit_h, 3)
    n_llm_w = _ceil_div(n_vit_w, 3)
    slots = _vision_block_slots(n_llm_h, n_llm_w)
    got = "".join("S" if role != _IMAGE_PAD else "P" for role, _ in slots)
    assert got == mask


@pytest.mark.parametrize(("n_vit_h", "n_vit_w", "mask"), LAYOUT_CASES)
def test_block_slots_feature_rows_are_dense_and_ordered(n_vit_h: int, n_vit_w: int, mask: str) -> None:
    """Every aligner row appears exactly once, in the interleaved slot order.

    This locks the row order ``encode`` emits; the C++ side scatters those
    rows at the matching sentinel positions, so the permutation must be
    exactly [perm index of each IMAGE slot].
    """
    n_llm_h = _ceil_div(n_vit_h, 3)
    n_llm_w = _ceil_div(n_vit_w, 3)
    slots = _vision_block_slots(n_llm_h, n_llm_w)
    feat_rows = [idx for role, idx in slots if role == _IMAGE]
    assert len(feat_rows) == n_llm_h * n_llm_w
    assert sorted(feat_rows) == list(range(n_llm_h * n_llm_w))
    # Every non-IMAGE slot carries no feature row.
    assert all(idx == -1 for role, idx in slots if role != _IMAGE)
    # Block ends with the IMAGE_END sentinel.
    assert slots[-1][0] == _IMAGE_END
    assert slots[0][0] == _IMAGE_START


@pytest.mark.parametrize("compress_pad", [0, 1, 2, 3])
def test_block_slots_compress_pad_prefixes_pads(compress_pad: int) -> None:
    slots = _vision_block_slots(2, 2, compress_pad)
    leading = [role for role, _ in slots[:compress_pad]]
    assert leading == [_IMAGE_PAD] * compress_pad
    # The block itself is the compress_pad=0 layout.
    assert slots[compress_pad:] == _vision_block_slots(2, 2)


def test_block_slots_roles_are_known_codes() -> None:
    for n_llm_h, n_llm_w, _ in LAYOUT_CASES:
        for role, _ in _vision_block_slots(n_llm_h, n_llm_w):
            assert role in (_IMAGE_START, _IMAGE_PAD, _IMAGE, _IMAGE_NEWLINE, _IMAGE_END)


def _vision_rotary(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    left, right = value.chunk(2, -1)
    rotated = torch.cat((-right, left), dim=-1)
    return (value.unsqueeze(0) * cos + rotated.unsqueeze(0) * sin).squeeze(0)


def _vision_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    actual_seq_qlen: list[int],
    actual_seq_kvlen: list[int],
    num_heads: int,
    scale: float,
    input_layout: str,
) -> torch.Tensor:
    assert input_layout == "TND"
    assert actual_seq_qlen == actual_seq_kvlen == [q.size(0)]
    assert q.size(1) == num_heads
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    return torch.einsum("hts,shd->thd", scores.softmax(-1), v.float()).to(q.dtype)


def _vision_silu_and_mul(value: torch.Tensor) -> torch.Tensor:
    gate, up = value.float().chunk(2, -1)
    return (F.silu(gate) * up).to(value.dtype)


@torch.inference_mode()
def test_vision_numerics_match_native_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin ten image features from identical inputs and FP32 norm weights."""
    monkeypatch.setattr(kernels, "vision_rotary_mul", _vision_rotary, raising=False)
    monkeypatch.setattr(kernels, "vision_fusion_attention", _vision_attention, raising=False)
    monkeypatch.setattr(kernels, "silu_and_mul", _vision_silu_and_mul, raising=False)
    cfg = DeepseekV41VisionConfig(
        n_layers=1,
        hidden_size=8,
        num_heads=1,
        intermediate_size=8,
        patch_size=2,
        num_channels=3,
        out_hidden_size=8,
        rope_theta=10000.0,
        downsample_ratio=2,
        head_dim=8,
    )
    assert cfg.norm_eps == 1e-6
    holder = torch.nn.Module()
    holder.vision_cfg = cfg
    holder.vision_model = DeepseekV41VisionTower(cfg, torch.bfloat16, torch.device("cpu"))
    holder.aligner = DeepseekV41Aligner(cfg, torch.bfloat16, torch.device("cpu"))
    # LOCKED PAIR with Bfloat16MatchesPythonGoldenWithFloat32Norms: keep names
    # in the C++ fixture's order, along with the formula, pixels and ten values.
    shapes = {
        "vision.patch_embed.proj.weight": (8, 3, 2, 2),
        "vision.patch_embed.proj.bias": (8,),
        "vision.norm.weight": (8,),
        "image_start": (8,),
        "image_end": (8,),
        "image_newline": (8,),
        "aligner.w1.weight": (8, 32),
        "aligner.w1.bias": (8,),
        "aligner.w2.weight": (8, 8),
        "aligner.w2.bias": (8,),
        "vision.blocks.0.attn.wqkv.weight": (24, 8),
        "vision.blocks.0.attn.wqkv.bias": (24,),
        "vision.blocks.0.attn.wo.weight": (8, 8),
        "vision.blocks.0.attn.wo.bias": (8,),
        "vision.blocks.0.mlp.w1.weight": (16, 8),
        "vision.blocks.0.mlp.w2.weight": (8, 8),
        "vision.blocks.0.norm1.weight": (8,),
        "vision.blocks.0.norm2.weight": (8,),
    }
    for index, (name, shape) in enumerate(shapes.items()):
        value = ((torch.arange(torch.Size(shape).numel()) * 7 + index).remainder(29) - 14).reshape(shape) / 32
        if "norm" in name:
            value = 1 + (torch.arange(8) - 4) * 0.0013
        elif "patch_embed" in name:
            value.mul_(0.001)
        if name.startswith("vision."):
            parameter = holder.vision_model.get_parameter(
                name.removeprefix("vision.").replace("patch_embed.proj.", "patch_embed.")
            )
        elif name.startswith("aligner."):
            parameter = holder.aligner.get_parameter(name.removeprefix("aligner."))
        else:
            holder.register_parameter(name, torch.nn.Parameter(value.to(torch.bfloat16), requires_grad=False))
            continue
        parameter.copy_(value.reshape(parameter.shape))
        if "norm" in name:
            assert parameter.dtype == torch.float32
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    pixels = torch.arange(24 * 12).remainder(31).reshape(24, 12).float() / 31 - 0.5
    output = DeepseekV41ForCausalLM.encode(holder, pixels, torch.tensor([1, 6, 4]))
    assert output.shape == (11, 8)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    # Skip image_start; the next two rows contain image features, not newlines.
    actual = output[1:3].flatten()[:10].float()
    expected = torch.tensor(
        [
            -0.0947265625,
            0.8984375,
            0.91015625,
            0.0137939453125,
            0.06591796875,
            0.0179443359375,
            0.028076171875,
            -0.8671875,
            -0.08642578125,
            0.88671875,
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
