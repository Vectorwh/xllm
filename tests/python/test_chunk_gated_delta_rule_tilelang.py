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

"""Tests for optional-gate and optional-state TileLang GDN paths."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch


def _chunk_gated_delta_rule_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small CPU oracle for per-chunk residual and state updates."""
    output_dtype = u.dtype
    k, w, u = (tensor.float().squeeze(0) for tensor in (k, w, u))
    g = g.float().squeeze(0) if g is not None else None
    initial_state = initial_state.float().squeeze(0) if initial_state is not None else None
    boundaries = cu_seqlens.to(torch.int64).tolist()
    num_tokens, num_key_heads, key_dim = k.shape
    num_heads, value_dim = u.shape[1:]
    h_chunks: list[torch.Tensor] = []
    v_new = torch.empty((num_tokens, num_heads, value_dim), dtype=torch.float32)
    final_states = []

    for sequence, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = (
            initial_state[sequence].clone()
            if initial_state is not None
            else torch.zeros((num_heads, key_dim, value_dim), dtype=torch.float32)
        )
        for chunk_begin in range(begin, end, chunk_size):
            chunk_end = min(chunk_begin + chunk_size, end)
            h_chunks.append(state.to(output_dtype).clone())
            for head in range(num_heads):
                key_head = head // (num_heads // num_key_heads)
                residual = u[chunk_begin:chunk_end, head] - torch.matmul(w[chunk_begin:chunk_end, head], state[head])
                v_new[chunk_begin:chunk_end, head] = residual
                update = residual
                if g is not None:
                    gate = g[chunk_begin:chunk_end, head]
                    last_gate = gate[-1]
                    delta = last_gate - gate
                    decay = torch.where(delta <= 0, delta, float("-inf")).exp()
                    update = residual * decay[:, None]
                    state[head] = state[head] * torch.exp(last_gate)
                state[head] = state[head] + torch.matmul(k[chunk_begin:chunk_end, key_head].transpose(0, 1), update)
        final_states.append(state)

    return (
        torch.stack(h_chunks).unsqueeze(0),
        v_new.to(output_dtype).unsqueeze(0),
        torch.stack(final_states),
    )


def _chunk_local_gates(
    cu_seqlens: list[int], total_tokens: int, num_heads: int, chunk_size: int, seed: int
) -> torch.Tensor:
    torch.manual_seed(seed)
    raw = torch.randn((1, total_tokens, num_heads), dtype=torch.float32) * 0.002
    gates = torch.empty_like(raw)
    for begin, end in zip(cu_seqlens, cu_seqlens[1:]):
        for chunk_begin in range(begin, end, chunk_size):
            chunk_end = min(chunk_begin + chunk_size, end)
            gates[:, chunk_begin:chunk_end] = raw[:, chunk_begin:chunk_end].cumsum(dim=1)
    return gates


def _tilelang_chunk_module():
    pytest.importorskip("torch_npu", reason="NPU TileLang tests require torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU TileLang tests require an available NPU")
    pytest.importorskip("tilelang", reason="NPU TileLang tests require TileLang")

    python_root = Path(__file__).parents[2] / "xllm" / "python"
    package_name = "xllm.python.kernels_npu.tilelang"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(python_root / "kernels_npu" / "tilelang")]
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.chunk_gated_delta_rule_fwd_h")


def test_chunk_gated_delta_rule_reference_without_gate_or_initial_state() -> None:
    k = torch.tensor([[[[1.0]]]], dtype=torch.bfloat16)
    w = torch.tensor([[[[0.5]]]], dtype=torch.bfloat16)
    u = torch.tensor([[[[2.0]]]], dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

    h, v_new, final_state = _chunk_gated_delta_rule_reference(k, w, u, None, None, cu_seqlens, chunk_size=2)

    torch.testing.assert_close(h, torch.zeros_like(h))
    torch.testing.assert_close(v_new, torch.tensor([[[[2.0]]]], dtype=torch.bfloat16))
    torch.testing.assert_close(final_state, torch.tensor([[[[2.0]]]]))


@pytest.mark.parametrize(
    ("use_gate", "use_initial_state"),
    [(False, True), (True, False), (True, True), (False, False)],
)
def test_chunk_gated_delta_rule_optional_paths_match_reference(use_gate: bool, use_initial_state: bool) -> None:
    module = _tilelang_chunk_module()
    torch.manual_seed(51)
    device = "npu:0"
    chunk_size = 64
    sequence_lengths = [65, 7]
    cu_seqlens_list = [0]
    for length in sequence_lengths:
        cu_seqlens_list.append(cu_seqlens_list[-1] + length)
    total_tokens = cu_seqlens_list[-1]
    batch, num_heads, num_key_heads, key_dim, value_dim = 2, 16, 8, 128, 128
    k = torch.randn((1, total_tokens, num_key_heads, key_dim), device=device, dtype=torch.bfloat16) * 0.01
    w = torch.randn((1, total_tokens, num_heads, key_dim), device=device, dtype=torch.bfloat16) * 0.01
    u = torch.randn((1, total_tokens, num_heads, value_dim), device=device, dtype=torch.bfloat16) * 0.01
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    g = (
        _chunk_local_gates(cu_seqlens_list, total_tokens, num_heads, chunk_size, seed=52).to(device)
        if use_gate
        else None
    )
    initial_state = (
        torch.randn((1, batch, num_heads, key_dim, value_dim), dtype=torch.float32) * 0.01
        if use_initial_state
        else None
    )
    if initial_state is not None:
        initial_state = initial_state.to(device)

    expected = _chunk_gated_delta_rule_reference(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g.cpu() if g is not None else None,
        initial_state.cpu() if initial_state is not None else None,
        cu_seqlens.cpu(),
        chunk_size,
    )
    actual = module.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), reference, rtol=5e-2, atol=5e-4)
