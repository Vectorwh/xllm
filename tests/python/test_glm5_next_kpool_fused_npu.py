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

"""Check paged pooling and prefix restoration against persistent CANN state."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from xllm.python.models.glm5_next_kpool import (  # noqa: E402
    _store_compressed_kpool_torch,
    prepare_fused_kpool_states,
    update_compressed_kpool,
)


@pytest.mark.parametrize("chunks", [(64,), (1, 2, 3, 4, 17, 37), (4096,), (3, 4093)])
@pytest.mark.parametrize("restore_prefix", [False, True])
@torch.inference_mode()
def test_fused_kpool_preserves_fp32_state(chunks: tuple[int, ...], restore_prefix: bool) -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "key_pool"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _check_pooling(chunks, restore_prefix)
    finally:
        torch.use_deterministic_algorithms(deterministic)


def _check_pooling(chunks: tuple[int, ...], restore_prefix: bool) -> None:
    tokens = sum(chunks)
    hidden = torch.randn(tokens, 4096, device="npu", dtype=torch.bfloat16)
    key_weight = torch.randn(128, 4096, device="npu", dtype=torch.bfloat16) * 0.02
    gate_weight = torch.randn_like(key_weight) * 0.02
    ape = torch.randn(4, 128, device="npu")
    norm_weight = torch.randn(128, device="npu")
    norm_bias = torch.randn_like(norm_weight)
    tail = torch.zeros(4, 2, 9, 128, device="npu")
    tail[:, 1].fill_(float("-inf"))
    pool_capacity = (tokens + 3) // 4 + 1
    cache = torch.zeros(2, pool_capacity, 1, 128, device="npu", dtype=torch.bfloat16)
    table = torch.tensor([[1]], device="npu", dtype=torch.int32)
    oracle_state = torch.zeros(pool_capacity + 1, 4, 256, device="npu")
    oracle_table = torch.arange(1, pool_capacity + 1, device="npu", dtype=torch.int32)[None]
    offset = 0
    read_slot = 1
    for chunk_index, length in enumerate(chunks):
        write_slot = 2 if restore_prefix and chunk_index > 0 else 1
        read_ids = torch.tensor([read_slot], device="npu", dtype=torch.int32)
        write_ids = torch.tensor([write_slot], device="npu", dtype=torch.int32)
        old_tail = tail[read_slot].clone()
        positions = torch.arange(offset, offset + length, device="npu")[None]
        inputs = hidden[offset : offset + length][None]
        valid = torch.ones(1, length, device="npu", dtype=torch.bool)
        states = prepare_fused_kpool_states(
            inputs,
            key_weight,
            gate_weight,
            ape,
            norm_weight,
            norm_bias,
            tail,
            read_ids,
            positions,
            [length],
            4,
            1e-6,
        )
        raw = F.layer_norm(F.linear(inputs, key_weight).float(), (128,), norm_weight, norm_bias, 1e-6)
        gates = F.linear(inputs, gate_weight).float()
        update_compressed_kpool(
            raw,
            gates,
            valid,
            valid,
            positions,
            cache,
            tail,
            read_ids,
            write_ids,
            table,
            [length],
            ape,
            4,
            fused_states=states,
        )
        expected = torch.ops.xllm_ops.key_pool(
            inputs[0],
            key_weight,
            gate_weight,
            ape,
            oracle_state,
            oracle_table,
            torch.tensor([offset], device="npu", dtype=torch.int32),
            norm_weight,
            norm_bias,
            torch.tensor([0, length], device="npu", dtype=torch.int32),
            4,
            1e-6,
        )
        count = (offset % 4 + length) // 4
        first_pool = offset // 4
        torch.testing.assert_close(
            cache[1, first_pool : first_pool + count, 0].cpu(),
            expected[:count].cpu(),
            rtol=0,
            atol=0,
        )
        expected_tail = old_tail.cpu()
        retained = min(length, tail.shape[2])
        retained_positions = positions[0, -retained:].cpu() % tail.shape[2]
        expected_tail[0].index_copy_(0, retained_positions, raw[0, -retained:].cpu())
        expected_tail[1].index_copy_(0, retained_positions, gates[0, -retained:].cpu())
        offset += length
        remainder = offset % 4
        if remainder:
            rows = torch.arange(offset - remainder, offset, device="npu") % tail.shape[2]
            partial = oracle_state[1 + offset // 4, :remainder].reshape(remainder, 2, 128).permute(1, 0, 2)
            expected_tail.index_copy_(1, rows.cpu(), partial.cpu())
        torch.testing.assert_close(tail[write_slot].cpu(), expected_tail, rtol=0, atol=0)
        if read_slot != write_slot:
            torch.testing.assert_close(tail[read_slot].cpu(), old_tail.cpu(), rtol=0, atol=0)
        read_slot = write_slot


@torch.inference_mode()
def test_batched_fused_kpool_keeps_requests_and_cached_geometry_independent() -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "key_pool"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    query_lens = [1, 0, 3, 5]
    hidden = torch.randn(sum(query_lens), 4096, dtype=torch.bfloat16, device="npu")
    key_weight = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu") * 0.02
    gate_weight = torch.randn_like(key_weight) * 0.02
    ape = torch.randn(4, 128, device="npu")
    norm_weight = torch.randn(128, device="npu")
    norm_bias = torch.randn_like(norm_weight)
    tail = torch.randn(4, 2, 9, 128, device="npu")
    ids = torch.tensor([3, 0, 2, 1], device="npu", dtype=torch.int32)
    positions = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13], device="npu")
    old_tail = tail.clone()
    batched = prepare_fused_kpool_states(
        hidden,
        key_weight,
        gate_weight,
        ape,
        norm_weight,
        norm_bias,
        tail,
        ids,
        positions,
        query_lens,
        4,
        1e-6,
    )
    offset = 0
    for request_idx, length in enumerate(query_lens):
        expected = prepare_fused_kpool_states(
            hidden[offset : offset + length],
            key_weight,
            gate_weight,
            ape,
            norm_weight,
            norm_bias,
            tail,
            ids[request_idx : request_idx + 1],
            positions[offset : offset + length],
            [length],
            4,
            1e-6,
        )[0]
        if length:
            pool_count = (int(positions[offset].item()) % 4 + length) // 4
            torch.testing.assert_close(batched[request_idx][0][:pool_count], expected[0][:pool_count], rtol=0, atol=0)
            torch.testing.assert_close(batched[request_idx][1], expected[1], rtol=0, atol=0)
        else:
            assert batched[request_idx][0].shape == expected[0].shape == (0, 128)
        offset += length
    torch.testing.assert_close(tail, old_tail, rtol=0, atol=0)


@pytest.mark.parametrize("query_width", [1, 3, 8])
@pytest.mark.parametrize("restore_prefix", [False, True])
@pytest.mark.parametrize("mask_pool_member", [False, True])
@torch.inference_mode()
def test_fused_kpool_graph_replay_matches_eager_with_padding(
    query_width: int, restore_prefix: bool, mask_pool_member: bool
) -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None or not torch.npu.is_available():
        pytest.skip("requires an NPU and XLLM_EXPORT_PATH")
    if not hasattr(torch.ops.xllm_ops, "key_pool"):
        torch.ops.load_library(artifact)
    torch.npu.set_device(0)
    torch.manual_seed(42)
    batch = 3
    key_weight = torch.randn(128, 4096, dtype=torch.bfloat16, device="npu") * 0.02
    gate_weight = torch.randn_like(key_weight) * 0.02
    ape = torch.randn(4, 128, device="npu")
    norm_weight = torch.randn(128, device="npu")
    norm_bias = torch.randn_like(norm_weight)
    hidden = torch.zeros(batch, query_width, 4096, dtype=torch.bfloat16, device="npu")
    positions = torch.full((batch, query_width), -1, device="npu", dtype=torch.int64)
    valid = torch.zeros(batch, query_width, device="npu", dtype=torch.bool)
    pool_valid = valid.clone()
    read_ids = torch.zeros(batch, device="npu", dtype=torch.int32)
    write_ids = torch.zeros_like(read_ids) if restore_prefix else read_ids
    cache = torch.zeros(4, 32, 1, 128, dtype=torch.bfloat16, device="npu")
    tail = torch.randn(8, 2, 11, 128, device="npu")
    table = torch.arange(1, 4, dtype=torch.int32, device="npu")[:, None]
    oracle_cache = cache.clone()
    oracle_tail = tail.clone()

    def step() -> None:
        states = prepare_fused_kpool_states(
            hidden,
            key_weight,
            gate_weight,
            ape,
            norm_weight,
            norm_bias,
            tail,
            read_ids,
            positions,
            [query_width] * batch,
            4,
            1e-6,
            valid_rows=valid,
        )
        raw = F.layer_norm(F.linear(hidden, key_weight).float(), (128,), norm_weight, norm_bias, 1e-6)
        gates = F.linear(hidden, gate_weight).float()
        update_compressed_kpool(
            raw,
            gates,
            pool_valid,
            valid,
            positions,
            cache,
            tail,
            read_ids,
            write_ids,
            table,
            [query_width] * batch,
            ape,
            4,
            graph_mode=True,
            fused_states=states,
        )

    step()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        step()
    lengths = [
        [query_width, query_width, 0],
        [1, max(1, query_width - 1), 0],
        [0, 0, 0],
        [query_width, 1, query_width],
    ]
    starts = [5, 7, 12]
    for iteration, counts in enumerate(lengths):
        hidden.normal_()
        positions.fill_(-1)
        valid.zero_()
        read_ids.zero_()
        write_ids.zero_()
        for request_idx, count in enumerate(counts):
            if not count:
                continue
            positions[request_idx, :count] = torch.arange(
                starts[request_idx], starts[request_idx] + count, device="npu"
            )
            valid[request_idx, :count] = True
            read_slot = request_idx + 1
            write_slot = request_idx + 4 if restore_prefix else read_slot
            read_ids[request_idx] = read_slot
            write_ids[request_idx] = write_slot
            inputs = hidden[request_idx, :count]
            request_positions = positions[request_idx, :count]
            request_read = read_ids[request_idx : request_idx + 1]
            request_write = write_ids[request_idx : request_idx + 1] if restore_prefix else request_read
            expected_states = prepare_fused_kpool_states(
                inputs,
                key_weight,
                gate_weight,
                ape,
                norm_weight,
                norm_bias,
                oracle_tail,
                request_read,
                request_positions,
                [count],
                4,
                1e-6,
            )
            raw = F.layer_norm(F.linear(inputs, key_weight).float(), (128,), norm_weight, norm_bias, 1e-6)
            gates = F.linear(inputs, gate_weight).float()
            mask = torch.ones(count, dtype=torch.bool, device="npu")
            pool_mask = mask.clone()
            if mask_pool_member and request_idx == 1:
                pool_mask[0] = False
            _store_compressed_kpool_torch(
                raw.reshape(-1, 128),
                gates.reshape(-1, 128),
                pool_mask,
                mask,
                request_positions,
                oracle_cache,
                oracle_tail,
                request_read,
                request_write,
                table[request_idx : request_idx + 1],
                [count],
                ape,
                4,
                fused_states=expected_states,
            )
            starts[request_idx] += count
        pool_valid.copy_(valid)
        if mask_pool_member:
            pool_valid[1, 0] = False
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(cache.cpu(), oracle_cache.cpu(), rtol=0, atol=0, msg=f"cache replay {iteration}")
        torch.testing.assert_close(tail.cpu(), oracle_tail.cpu(), rtol=0, atol=0, msg=f"tail replay {iteration}")
