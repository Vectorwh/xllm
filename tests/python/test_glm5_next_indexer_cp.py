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

from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import glm5_next
from xllm.python.models.glm5_next_kpool import update_compressed_kpool

# One 512-token sequence pooled at rate 4: 128 pools, 8 pools per cache
# block, 16 blocks.
KV_LEN = 512
RATE = 4
HEAD_DIM = 128
POOLS_PER_BLOCK = 8
NUM_BLOCKS = KV_LEN // RATE // POOLS_PER_BLOCK


def _indexer(device: torch.device) -> glm5_next.Glm5NextIndexer:
    config = glm5_next.Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=128,
        index_kpool=4,
        index_topk=8,
        index_kpool_compress=True,
    )
    return glm5_next.Glm5NextIndexer(config, 1, torch.bfloat16, device).to(device=device, dtype=torch.bfloat16)


def _pool_cache_and_table() -> tuple[torch.Tensor, torch.Tensor]:
    """Write a complete compressed pool cache for one KV_LEN-token sequence."""
    cache = torch.zeros(NUM_BLOCKS, POOLS_PER_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16)
    table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    update_compressed_kpool(
        raw_k=torch.randn(KV_LEN, HEAD_DIM, dtype=torch.bfloat16),
        gate_scores=torch.zeros(KV_LEN, HEAD_DIM, dtype=torch.bfloat16),
        valid_rows=torch.ones(KV_LEN, dtype=torch.bool),
        tail_valid_rows=torch.ones(KV_LEN, dtype=torch.bool),
        positions=torch.arange(KV_LEN),
        compressed_cache=cache,
        tail_cache=torch.zeros(1, 2, RATE, HEAD_DIM, dtype=torch.bfloat16),
        tail_read_ids=torch.tensor([0], dtype=torch.int32),
        tail_write_ids=torch.tensor([0], dtype=torch.int32),
        block_table=table,
        query_lens=[KV_LEN],
        ape=torch.zeros(RATE, HEAD_DIM, dtype=torch.float32),
        rate=RATE,
    )
    return cache, table


@torch.inference_mode()
def test_select_topk_row_slicing_is_bit_exact() -> None:
    torch.manual_seed(42)
    indexer = _indexer(torch.device("cpu"))
    indexer.process_weights_after_loading()
    cache, table = _pool_cache_and_table()
    qr = torch.randn(1, KV_LEN, 16, dtype=torch.bfloat16)
    hidden = torch.randn(1, KV_LEN, 16, dtype=torch.bfloat16)
    mask = torch.ones(1, KV_LEN, dtype=torch.bool)
    positions = torch.arange(KV_LEN).unsqueeze(0)
    q_full = indexer.wq_b(qr)

    def _select(
        rows: torch.Tensor | None, kv_seq_lens: torch.Tensor, projected_q: torch.Tensor | None = None
    ) -> torch.Tensor:
        return indexer.select_topk(
            qr if rows is None else qr[:, rows],
            hidden if rows is None else hidden[:, rows],
            mask if rows is None else torch.ones(1, rows.numel(), dtype=torch.bool),
            KV_LEN,
            KV_LEN,
            pool_cache=cache,
            pool_block_table=table,
            kv_seq_lens=kv_seq_lens,
            projected_q=projected_q,
            query_positions=positions if rows is None else positions[:, rows],
        )

    full = _select(None, torch.tensor([KV_LEN]))
    cases: dict[str, torch.Tensor] = {
        "contig": torch.arange(128, 384),
        "perm": torch.randperm(KV_LEN)[:256],
        "boundary": torch.cat((torch.arange(0, 16), torch.arange(KV_LEN - 16, KV_LEN))),
        "single": torch.tensor([KV_LEN // 2 - 1]),
    }
    for name, rows in cases.items():
        # A single sliced row mirrors the fused path's query_len == 1 branch,
        # which reads causal visibility from kv_seq_lens.
        kv_seq_lens = torch.tensor([rows[-1] + 1]) if rows.numel() == 1 else torch.tensor([KV_LEN])
        sliced = _select(rows, kv_seq_lens)
        assert torch.equal(sliced, full[:, rows]), f"row case {name} diverged from the full-stream selection"
        # Reusing the full-M wq_b output via projected_q — the CP α branch's
        # bit-exactness contract — must reproduce the same rows.
        reused = _select(rows, kv_seq_lens, projected_q=q_full[:, rows])
        assert torch.equal(reused, full[:, rows]), f"row case {name} diverged with projected_q"


def _wired_select_qli_inputs() -> tuple[
    glm5_next.Glm5NextIndexer,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SimpleNamespace,
    SimpleNamespace,
]:
    """Indexer + CPU-path ctx/backend stubs for select_qli (packed history)."""
    indexer = _indexer(torch.device("cpu"))
    indexer.process_weights_after_loading()
    hidden = torch.randn(1, KV_LEN, 16, dtype=torch.bfloat16)
    qr = torch.randn(1, KV_LEN, 16, dtype=torch.bfloat16)
    positions = torch.arange(KV_LEN).unsqueeze(0)
    mask = torch.ones(1, KV_LEN, dtype=torch.bool)
    history = torch.randn(1, KV_LEN, HEAD_DIM * 2 + 1, dtype=torch.bfloat16)
    history[..., -1] = 1
    ctx = SimpleNamespace(
        index_cache=None,
        slot_mapping=None,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        actual_seq_kv=torch.tensor([KV_LEN]),
        kpool_tail=None,
        # Full-replica index cache: every context field MlaIndexContext
        # defaults to, spelled out because this is a duck-typed stand-in.
        has_kv_shard=False,
        materialized_block_table=None,
        localize_pool_block_table=lambda table: table,
    )
    backend = SimpleNamespace(gather_index_history=lambda *_args: history)
    return indexer, hidden, qr, positions, mask, ctx, backend


@torch.inference_mode()
def test_select_qli_query_rows_partition_matches_full_stream() -> None:
    torch.manual_seed(42)
    indexer, hidden, qr, positions, mask, ctx, backend = _wired_select_qli_inputs()

    def _select_qli(query_rows: torch.Tensor | None) -> torch.Tensor:
        return indexer.select_qli(hidden, qr, positions, mask, ctx, None, backend, query_rows=query_rows)

    full = _select_qli(None)
    rows = torch.arange(128, 384)
    sliced = _select_qli(rows)
    assert torch.equal(sliced[rows], full[rows])
    outside = torch.ones(KV_LEN, dtype=torch.bool)
    outside[rows] = False
    assert bool((sliced[outside] == -1).all()), "rows outside query_rows must stay -1 in the full-layout output"

    local_rows = torch.cat((torch.arange(128), torch.arange(KV_LEN - 128, KV_LEN)))
    shard_index = torch.cat((local_rows, torch.tensor([-1])))
    cp_context = SimpleNamespace(
        total_local=shard_index.numel(),
        shard_index=shard_index,
        shard_gather_index=shard_index.clamp_min(0),
        shard_valid_mask=shard_index >= 0,
        query_index=torch.arange(local_rows.numel()),
        q_cu_seqlens=[128, local_rows.numel()],
    )
    with forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=cp_context)):
        local = indexer.select_qli(
            hidden, qr, positions, mask, ctx, None, backend, query_rows=local_rows, cp_local_output=True
        )
    assert local.shape[0] == cp_context.total_local
    torch.testing.assert_close(local[cp_context.query_index], full[local_rows], rtol=0, atol=0)

    rows = torch.tensor([2, 0, 5, 7, KV_LEN])
    with forward_context(ForwardContext(None, torch.device("cpu"), None, [])):
        first = glm5_next._kpool_query_rows_by_sequence(rows, [4, 4], 8)
        repeated = glm5_next._kpool_query_rows_by_sequence(rows, [4, 4], 8)
        assert repeated is first
        assert [part.tolist() for part in first] == [[2, 0], [5, 7]]
        trimmed = glm5_next._kpool_query_rows_by_sequence(rows, [4, 2], 6)
        assert [part.tolist() for part in trimmed] == [[2, 0], [5]]
    rows[0] = 3
    with forward_context(ForwardContext(None, torch.device("cpu"), None, [])):
        next_forward = glm5_next._kpool_query_rows_by_sequence(rows, [4, 4], 8)
        assert next_forward is not first
        assert [part.tolist() for part in next_forward] == [[3, 0], [5, 7]]


@torch.inference_mode()
def test_select_qli_query_rows_empty_fills_invalid() -> None:
    torch.manual_seed(42)
    indexer, hidden, qr, positions, mask, ctx, backend = _wired_select_qli_inputs()

    topk_indices = indexer.select_qli(
        hidden,
        qr,
        positions,
        mask,
        ctx,
        None,
        backend,
        query_rows=torch.empty(0, dtype=torch.long),
    )
    width = indexer.topk + (indexer.index_kpool - 1 if indexer.index_kpool_always_select_tail else 0)
    assert topk_indices.shape == (KV_LEN, 1, width)
    assert bool((topk_indices == -1).all())


@torch.inference_mode()
def test_select_qli_query_rows_active_under_kv_split_replicated(monkeypatch: pytest.MonkeyPatch) -> None:
    """kv_split > 1 with the DEFAULT replicated index-write mode keeps the
    per-rank index cache a full replica and reads through per-sequence
    page-expanded tables, so the alpha slice stays active; the non-default
    sharded mode falls back to the full stream."""
    torch.manual_seed(42)
    indexer, hidden, qr, positions, mask, ctx, backend = _wired_select_qli_inputs()
    ctx.has_kv_shard = True
    rows = torch.arange(128, 384)

    def _select_qli(query_rows: torch.Tensor | None) -> torch.Tensor:
        return indexer.select_qli(hidden, qr, positions, mask, ctx, None, backend, query_rows=query_rows)

    full = _select_qli(None)
    monkeypatch.setattr(glm5_next, "cp_index_write_replicated", lambda: True)
    sliced = _select_qli(rows)
    assert torch.equal(sliced[rows], full[rows])
    outside = torch.ones(KV_LEN, dtype=torch.bool)
    outside[rows] = False
    assert bool((sliced[outside] == -1).all()), "replicated kv_split must keep the alpha slice active"

    monkeypatch.setattr(glm5_next, "cp_index_write_replicated", lambda: False)
    fallback = _select_qli(rows)
    assert torch.equal(fallback, full), "sharded kv_split must fall back to the full stream verbatim"


@pytest.mark.parametrize("query_lens", [(4, 4), (5, 3)])
@torch.inference_mode()
def test_select_qli_partition_preserves_request_histories(
    monkeypatch: pytest.MonkeyPatch, query_lens: tuple[int, int]
) -> None:
    """CP rows spanning two requests must use their respective causal histories."""
    torch.manual_seed(43)
    indexer, _, _, _, _, ctx, _ = _wired_select_qli_inputs()
    total = sum(query_lens)
    hidden = torch.randn(1, total, 16, dtype=torch.bfloat16)
    qr = torch.randn(total, 16, dtype=torch.bfloat16)
    positions = torch.cat([torch.arange(KV_LEN - length, KV_LEN) for length in query_lens]).unsqueeze(0)
    mask = torch.ones(1, total, dtype=torch.bool)
    history = torch.randn(2, KV_LEN, HEAD_DIM * 2 + 1, dtype=torch.bfloat16)
    history[..., -1] = 1
    ctx.block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    ctx.actual_seq_kv = torch.tensor([KV_LEN, KV_LEN])
    metadata = SimpleNamespace(q_cu_seq_lens=torch.tensor([0, query_lens[0], total]))
    monkeypatch.setattr(
        glm5_next, "get_forward_context", lambda: SimpleNamespace(layer_shared_cache={}, metadata=metadata)
    )
    backend = SimpleNamespace(gather_index_history=lambda *_: history)
    rows = torch.tensor([1, query_lens[0] - 1, query_lens[0] + 1, total - 1])
    full = indexer.select_qli(hidden, qr, positions, mask, ctx, None, backend)
    partitioned = indexer.select_qli(hidden, qr, positions, mask, ctx, None, backend, query_rows=rows)
    assert torch.equal(partitioned[rows], full[rows])
    outside = torch.ones(total, dtype=torch.bool)
    outside[rows] = False
    assert bool((partitioned[outside] == -1).all())


@torch.inference_mode()
def test_select_qli_partition_preserves_paged_request_caches() -> None:
    """Unequal requests must retain distinct pool-cache pages during CP scoring."""
    torch.manual_seed(44)
    indexer = _indexer(torch.device("cpu"))
    indexer.index_kpool_compress_gate.zero_()
    indexer.index_kpool_compress_ape.zero_()
    indexer.process_weights_after_loading()
    indexer._uses_npu_compressed_tail = True
    query_lens = (8, 12)
    total = sum(query_lens)
    hidden = torch.randn(1, total, 16, dtype=torch.bfloat16)
    qr = torch.randn(total, 16, dtype=torch.bfloat16)
    positions = torch.cat([torch.arange(length) for length in query_lens]).unsqueeze(0)
    mask = torch.ones(1, total, dtype=torch.bool)
    cache = torch.zeros(3, POOLS_PER_BLOCK, 1, HEAD_DIM, dtype=torch.bfloat16)
    tail = torch.zeros(3, 2, RATE, HEAD_DIM, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        index_cache=cache,
        slot_mapping=torch.arange(1, total + 1, dtype=torch.int64),
        block_table=torch.tensor([[1], [2]], dtype=torch.int32),
        actual_seq_kv=torch.tensor(query_lens),
        kpool_tail=tail,
        kpool_tail_read_indices=torch.tensor([1, 2], dtype=torch.int32),
        kpool_tail_write_indices=torch.tensor([1, 2], dtype=torch.int32),
        kpool_query_lens=query_lens,
        has_kv_shard=False,
        localize_pool_block_table=lambda table: table,
    )
    rows = torch.tensor([3, 7, 11, 15, 19])
    full = indexer.select_qli(hidden, qr, positions, mask, ctx, None, None)
    assert not torch.equal(cache[1], cache[2])
    cache.zero_()
    tail.zero_()
    partitioned = indexer.select_qli(hidden, qr, positions, mask, ctx, None, None, query_rows=rows)
    assert torch.equal(partitioned[rows], full[rows])
    outside = torch.ones(total, dtype=torch.bool)
    outside[rows] = False
    assert bool((partitioned[outside] == -1).all())
