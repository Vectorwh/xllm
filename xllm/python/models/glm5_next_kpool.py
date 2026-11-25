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
"""Fast torch implementation of kPool pooling (bit-exact equivalent to the original get_pooled_states).

Uses ``F.embedding`` (a small [B, P, rate] index) to replace the flattened
``gather(1, flat_idx)`` (a large [B, P*rate*D] int64 index): the set of
selected elements is identical (safe already clamped), only saving the
construction of and random access into a 4.19M-element index tensor.
Microbenchmark 2.50ms -> 0.05ms (B=1, T=32768, D=128, rate=4, npu:1).
CPU tests can load the portable functions without importing NPU dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import torch
import torch.nn.functional as F


def _gather_rows(src: torch.Tensor, safe_indices: torch.Tensor) -> torch.Tensor:
    """src [B, T, D] (may be a strided view), safe_indices [B, P, R] -> [B, P, R, D]."""
    if src.shape[0] == 1:
        return F.embedding(safe_indices[0], src[0]).unsqueeze(0)
    return torch.stack([F.embedding(safe_indices[b], src[b]) for b in range(src.shape[0])])


def pooled_states(packed_states: torch.Tensor, key_valid: torch.Tensor, ape: torch.Tensor, head_dim: int, rate: int):
    """Bit-exact identical to the full output of Glm5NextIndexer.get_pooled_states."""
    keys, gate_scores, _ = torch.split(packed_states, [head_dim, head_dim, 1], dim=-1)
    batch_size, total_len = keys.shape[:2]
    device = keys.device
    first_key = torch.where(
        key_valid.any(-1),
        key_valid.to(torch.int32).argmax(-1),
        torch.full((batch_size,), total_len, dtype=torch.long, device=device),
    )
    n_pools = (total_len + rate - 1) // rate
    pool_offsets = torch.arange(n_pools, device=device) * rate
    slot_offsets = torch.arange(rate, device=device)
    pool_indices = first_key[:, None, None] + pool_offsets[None, :, None] + slot_offsets[None, None, :]
    slot_in_range = pool_indices < total_len
    safe_indices = pool_indices.clamp(0, total_len - 1)
    grouped_keys = _gather_rows(keys, safe_indices)  # [B, P, R, D]
    grouped_gate_scores = _gather_rows(gate_scores, safe_indices)
    slot_valid = (
        key_valid.to(torch.uint8)
        .gather(1, safe_indices.reshape(batch_size, -1))
        .reshape(batch_size, n_pools, rate)
        .to(torch.bool)
        & slot_in_range
    )
    pool_valid = slot_valid.all(-1)
    logits = grouped_gate_scores.float() + ape.float()[None, None]
    logits = logits.masked_fill(~slot_valid[..., None], float("-inf"))
    weights = torch.nan_to_num(logits.softmax(2)).to(grouped_keys.dtype)
    pool_keys = (weights * grouped_keys).sum(2)
    pool_indices = pool_indices.masked_fill(~slot_valid, -1)
    return pool_keys, pool_indices, pool_valid


def update_compressed_kpool(
    raw_k: torch.Tensor,
    gate_scores: torch.Tensor,
    valid_rows: torch.Tensor,
    tail_valid_rows: torch.Tensor,
    positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    tail_cache: torch.Tensor,
    tail_read_ids: torch.Tensor,
    tail_write_ids: torch.Tensor,
    block_table: torch.Tensor,
    query_lens: Sequence[int],
    ape: torch.Tensor,
    rate: int,
    graph_mode: bool = False,
    fused_states: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    """Complete compressed pools and then persist the current raw tail.

    ``valid_rows`` gates POOL completion -- under sharded CP writes it is the
    owner-localized mask, so each rank completes only the pools whose members
    it owns and the mixed-ownership boundary pools stay for the decode side.
    ``tail_valid_rows`` gates the tail persistence and must be the GLOBAL
    real-token mask: the tail is a per-sequence linear accumulator replicated
    on every CP rank, so a rank zeroing the peer tokens it does not own would
    leave a divergent copy whose write-order in the PD handoff decides
    whether the boundary pool can ever complete.

    ``raw_k``/``gate_scores``/``positions`` contain flattened, sequence-major
    query rows, with increasing real-token positions per request. Padding is
    excluded by ``tail_valid_rows``. ``query_lens`` maps those rows back to
    logical requests; the block table may either have one row per request or
    one repeated row per query token (MTP verify). Read/write tail ids may differ
    for an out-of-place prefix-cache restore. Pool completion always runs before
    tail writes so a circular tail slot cannot overwrite a value still needed
    by this step.
    """
    if rate <= 0:
        raise ValueError("kPool compression rate must be positive")
    if compressed_cache.ndim != 4 or compressed_cache.shape[2] != 1:
        raise ValueError("compressed kPool cache must have shape [blocks, pools_per_block, 1, dim]")
    if tail_cache.ndim != 4 or tail_cache.shape[1] != 2:
        raise ValueError("kPool tail must have shape [slots, 2, capacity, dim]")

    flat_k = raw_k.reshape(-1, raw_k.shape[-1])
    flat_gate = gate_scores.reshape(-1, gate_scores.shape[-1])
    flat_valid = valid_rows.reshape(-1).to(torch.bool)
    flat_tail_valid = tail_valid_rows.reshape(-1).to(torch.bool)
    if flat_tail_valid.numel() != flat_valid.numel():
        raise ValueError(
            "kPool tail validity must cover the same rows as the pool "
            f"validity: tail={flat_tail_valid.numel()}, pools={flat_valid.numel()}"
        )
    flat_positions = positions.reshape(-1).to(torch.int64)
    if flat_k.shape != flat_gate.shape:
        raise ValueError("raw K and gate rows must have identical shapes")
    if flat_k.shape[0] != flat_positions.numel() or flat_k.shape[0] != flat_valid.numel():
        raise ValueError("kPool rows, positions, and validity must have equal lengths")
    if len(query_lens) > tail_read_ids.numel() or len(query_lens) > tail_write_ids.numel():
        raise ValueError("kPool query groups exceed available linear-state read/write ids")
    if sum(query_lens) != flat_k.shape[0]:
        raise ValueError("kPool query groups must cover every token row")

    if block_table.ndim != 2 or block_table.shape[0] not in (len(query_lens), sum(query_lens)):
        raise ValueError("kPool block table must have one row per request or query token")

    head_dim = flat_k.shape[-1]
    if compressed_cache.shape[-1] != head_dim or tail_cache.shape[-1] != head_dim:
        raise ValueError("kPool cache and tail dimensions must match projected K")
    if tail_cache.shape[2] < rate:
        raise ValueError("kPool tail capacity must cover one complete pool")

    if (
        fused_states is not None
        and raw_k.device.type == "npu"
        and query_lens
        and all(length > 0 for length in query_lens)
        and head_dim <= 128
        and compressed_cache.is_contiguous()
        and tail_cache.is_contiguous()
        and block_table.is_contiguous()
    ):
        from xllm.python.kernels_npu.triton.kpool_compress import store_fused_kpool

        # Prefix restores can read a slot another request will replace.
        # Snapshot both old read/write slots for this batch before any launch;
        # storage is O(batch), independent of the total tail-cache capacity.
        read_tail = None
        if tail_read_ids is not tail_write_ids:
            snapshot_ids = torch.stack(
                (tail_read_ids.reshape(-1)[: len(query_lens)], tail_write_ids.reshape(-1)[: len(query_lens)]), dim=1
            )
            read_tail = tail_cache.index_select(0, snapshot_ids.reshape(-1).long().clamp(0, tail_cache.shape[0] - 1))
        if all(length == query_lens[0] for length in query_lens):
            store_fused_kpool(
                raw_k,
                gate_scores,
                valid_rows,
                tail_valid_rows,
                positions,
                compressed_cache,
                tail_cache,
                tail_read_ids,
                tail_write_ids,
                block_table,
                query_lens[0],
                rate,
                list(fused_states),
                read_tail_cache=read_tail,
            )
        else:
            row_start = 0
            for request, query_len in enumerate(query_lens):
                row_end = row_start + query_len
                read_ids = tail_read_ids.reshape(-1)[request : request + 1]
                write_ids = (
                    read_ids if tail_read_ids is tail_write_ids else tail_write_ids.reshape(-1)[request : request + 1]
                )
                table_row = request if block_table.shape[0] == len(query_lens) else row_start
                store_fused_kpool(
                    flat_k[row_start:row_end],
                    flat_gate[row_start:row_end],
                    flat_valid[row_start:row_end],
                    flat_tail_valid[row_start:row_end],
                    flat_positions[row_start:row_end],
                    compressed_cache,
                    tail_cache,
                    read_ids,
                    write_ids,
                    block_table[table_row : table_row + 1],
                    query_len,
                    rate,
                    [fused_states[request]],
                    read_tail_cache=None if read_tail is None else read_tail[2 * request : 2 * request + 2],
                )
                row_start = row_end
        return

    _store_compressed_kpool_torch(
        flat_k,
        flat_gate,
        flat_valid,
        flat_tail_valid,
        flat_positions,
        compressed_cache,
        tail_cache,
        tail_read_ids,
        tail_write_ids,
        block_table,
        query_lens,
        ape,
        rate,
        graph_mode,
        fused_states,
    )


def _store_compressed_kpool_torch(
    flat_k: torch.Tensor,
    flat_gate: torch.Tensor,
    flat_valid: torch.Tensor,
    flat_tail_valid: torch.Tensor,
    flat_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    tail_cache: torch.Tensor,
    tail_read_ids: torch.Tensor,
    tail_write_ids: torch.Tensor,
    block_table: torch.Tensor,
    query_lens: Sequence[int],
    ape: torch.Tensor,
    rate: int,
    graph_mode: bool = False,
    fused_states: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    """Store validated rows using the portable tensor implementation."""
    head_dim = flat_k.shape[-1]
    pool_block_size = compressed_cache.shape[1]
    cache_flat = compressed_cache.reshape(-1, head_dim)
    tail_capacity = tail_cache.shape[2]
    rate_offsets = torch.arange(rate, dtype=torch.int64, device=flat_positions.device)
    flat_tail_read_ids = tail_read_ids.reshape(-1)
    flat_tail_write_ids = tail_write_ids.reshape(-1)
    has_distinct_tail_io = tail_read_ids is not tail_write_ids
    read_tail_snapshots = (
        [
            tail_cache.index_select(
                0,
                flat_tail_read_ids[request_idx].to(torch.int64).clamp(0, tail_cache.shape[0] - 1).reshape(1),
            )
            .squeeze(0)
            .clone()
            for request_idx in range(len(query_lens))
        ]
        if has_distinct_tail_io
        else []
    )
    row_start = 0

    # Phase 1: every completion reads the old tail before any row overwrites it.
    for request_idx, query_len in enumerate(query_lens):
        if query_len < 0:
            raise ValueError("kPool query lengths must be non-negative")
        row_end = row_start + query_len
        if query_len == 0:
            row_start = row_end
            continue
        request_positions = flat_positions[row_start:row_end]
        request_k = flat_k[row_start:row_end]
        request_gate = flat_gate[row_start:row_end]
        request_valid = flat_valid[row_start:row_end]
        tail_read_id = flat_tail_read_ids[request_idx].to(torch.int64)
        safe_tail_read_id = tail_read_id.clamp(0, tail_cache.shape[0] - 1)
        table_row = request_idx if block_table.shape[0] == len(query_lens) else row_start
        request_table = block_table[table_row].reshape(-1).to(torch.int64)
        tail_read_state = tail_cache.index_select(0, safe_tail_read_id.reshape(1)).squeeze(0)

        member_positions = request_positions[:, None] - rate + 1 + rate_offsets[None, :]
        source_rows = member_positions - request_positions[0]
        safe_source_rows = source_rows.clamp(0, query_len - 1)
        source_positions = request_positions.index_select(0, safe_source_rows.reshape(-1)).view(query_len, rate)
        from_current = (source_rows >= 0) & (source_rows < query_len) & (source_positions == member_positions)
        current_keys = request_k.index_select(0, safe_source_rows.reshape(-1)).view(query_len, rate, head_dim)
        current_gates = request_gate.index_select(0, safe_source_rows.reshape(-1)).view(
            query_len,
            rate,
            head_dim,
        )
        current_valid = request_valid.index_select(0, safe_source_rows.reshape(-1)).view(query_len, rate)
        tail_rows = torch.remainder(member_positions, tail_capacity)
        old_keys = (
            tail_read_state[0]
            .index_select(0, tail_rows.reshape(-1))
            .view(
                query_len,
                rate,
                head_dim,
            )
        )
        old_gates = (
            tail_read_state[1]
            .index_select(0, tail_rows.reshape(-1))
            .view(
                query_len,
                rate,
                head_dim,
            )
        )
        keys = torch.where(from_current[..., None], current_keys, old_keys)
        gates = torch.where(from_current[..., None], current_gates, old_gates)
        old_valid = (member_positions >= 0) & torch.isfinite(old_gates[..., 0])
        members_valid = torch.where(from_current, current_valid, old_valid)
        complete = (
            (tail_read_id > 0)
            & (request_positions >= rate - 1)
            & (torch.remainder(request_positions + 1, rate) == 0)
            & members_valid.all(dim=1)
        )

        if fused_states is None:
            logits = gates.float() + ape.float()[None]
            logits = logits.masked_fill(~members_valid[..., None], float("-inf"))
            probabilities = torch.nan_to_num(torch.softmax(logits, dim=1)).to(compressed_cache.dtype)
            products = (probabilities.float() * keys.float()).to(keys.dtype).float()
            compressed = products.sum(1).to(compressed_cache.dtype)
        else:
            pooled = fused_states[request_idx][0]
            relative_positions = torch.remainder(request_positions[0], rate) + torch.arange(
                query_len, device=flat_positions.device
            )
            output_rows = torch.div(relative_positions + 1, rate, rounding_mode="floor") - 1
            compressed = pooled.index_select(0, output_rows.clamp(0, pooled.shape[0] - 1))

        pool_ids = torch.div(request_positions.clamp_min(0), rate, rounding_mode="floor")
        logical_blocks = torch.div(pool_ids, pool_block_size, rounding_mode="floor")
        table_in_range = logical_blocks < request_table.numel()
        safe_logical_blocks = logical_blocks.clamp(0, request_table.numel() - 1)
        physical_blocks = request_table.index_select(0, safe_logical_blocks)
        cache_slots = (
            physical_blocks.clamp_min(0) * pool_block_size + torch.remainder(pool_ids, pool_block_size)
        ).clamp(0, cache_flat.shape[0] - 1)
        writes = complete & table_in_range & (physical_blocks >= 0)
        if graph_mode:
            # Fixed masked writes avoid data-dependent nonzero output during
            # graph capture. Speculative windows keep this loop small.
            for local_row in range(query_len):
                row_slice = slice(local_row, local_row + 1)
                cache_slot = cache_slots[row_slice]
                cache_flat.index_copy_(
                    0,
                    cache_slot,
                    torch.where(
                        writes[row_slice, None],
                        compressed[row_slice],
                        cache_flat.index_select(0, cache_slot),
                    ),
                )
        else:
            selected = writes.nonzero().flatten()
            cache_flat.index_copy_(
                0,
                cache_slots.index_select(0, selected),
                compressed.index_select(0, selected),
            )
        row_start = row_end

    # Phase 2: persist current rows after every completion has consumed the old
    # tail. Sequential writes make wraparound deterministic for long chunks.
    row_start = 0
    for request_idx, query_len in enumerate(query_lens):
        row_end = row_start + query_len
        if query_len == 0:
            row_start = row_end
            continue
        tail_write_id = flat_tail_write_ids[request_idx].to(torch.int64)
        safe_tail_write_id = tail_write_id.clamp(0, tail_cache.shape[0] - 1)
        tail_write_index = safe_tail_write_id.reshape(1)
        destination_tail = tail_cache.index_select(0, tail_write_index).squeeze(0)
        if has_distinct_tail_io:
            tail_read_id = flat_tail_read_ids[request_idx].to(torch.int64)
            copy_state = (tail_read_id > 0) & (tail_write_id > 0)
            destination_tail = torch.where(copy_state, read_tail_snapshots[request_idx], destination_tail)
        stash_start = max(row_start, row_end - tail_capacity)
        stash_positions = flat_positions[stash_start:row_end]
        tail_rows = torch.remainder(stash_positions.clamp_min(0), tail_capacity)
        active_rows = (tail_write_id > 0) & (stash_positions >= 0)
        writes = active_rows & flat_tail_valid[stash_start:row_end]
        if graph_mode:
            # Padded positions can alias an active row in the circular buffer.
            # Read each destination after the preceding write so an inactive
            # alias preserves its current value instead of restoring a stale
            # snapshot from before this request.
            for local_row in range(row_end - stash_start):
                row = slice(local_row, local_row + 1)
                tail_row = tail_rows[row]
                active = active_rows[row, None]
                write = writes[row, None]
                source_row = slice(stash_start + local_row, stash_start + local_row + 1)
                for channel, values, fill in ((0, flat_k, 0.0), (1, flat_gate, float("-inf"))):
                    previous = destination_tail[channel].index_select(0, tail_row)
                    current = values[source_row].to(tail_cache.dtype)
                    value = torch.where(write, current, torch.full_like(current, fill))
                    destination_tail[channel].index_copy_(0, tail_row, torch.where(active, value, previous))
        else:
            old_keys = destination_tail[0].index_select(0, tail_rows)
            old_gates = destination_tail[1].index_select(0, tail_rows)
            destination_tail[0].index_copy_(
                0,
                tail_rows,
                torch.where(
                    active_rows[:, None],
                    torch.where(
                        writes[:, None],
                        flat_k[stash_start:row_end].to(tail_cache.dtype),
                        torch.zeros_like(flat_k[stash_start:row_end], dtype=tail_cache.dtype),
                    ),
                    old_keys,
                ),
            )
            destination_tail[1].index_copy_(
                0,
                tail_rows,
                torch.where(
                    active_rows[:, None],
                    torch.where(
                        writes[:, None],
                        flat_gate[stash_start:row_end].to(tail_cache.dtype),
                        torch.full_like(flat_gate[stash_start:row_end], float("-inf"), dtype=tail_cache.dtype),
                    ),
                    old_gates,
                ),
            )
        if fused_states is not None:
            # KeyPool retains normalized K in FP32. Keep that exact state for
            # the next partial-pool completion, including historical members.
            request_valid = flat_tail_valid[row_start:row_end] & (flat_positions[row_start:row_end] >= 0)
            end_position = torch.where(request_valid, flat_positions[row_start:row_end], -1).amax() + 1
            tail_length = torch.remainder(end_position, rate)
            last_pool = end_position - tail_length
            state_positions = last_pool + rate_offsets
            state_rows = torch.remainder(state_positions, tail_capacity)
            state = fused_states[request_idx][1]
            live = (rate_offsets < tail_length) & (tail_write_id > 0)
            for channel in range(2):
                previous = destination_tail[channel].index_select(0, state_rows)
                destination_tail[channel].index_copy_(
                    0, state_rows, torch.where(live[:, None], state[channel], previous)
                )
        tail_cache.index_copy_(0, tail_write_index, destination_tail.unsqueeze(0))
        row_start = row_end


@lru_cache(maxsize=128)
def _fused_kpool_metadata(
    query_len: int, rate: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reuse immutable request geometry across layers and eager steps."""
    offsets = torch.arange(rate, device=device)
    table = torch.full((1, 1 + (query_len + rate - 1) // rate), 2, dtype=torch.int32, device=device)
    table[:, 0] = 1
    cu_seqlens = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    return offsets, table, cu_seqlens


def prepare_fused_kpool_states(
    hidden_states: torch.Tensor,
    key_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    tail_cache: torch.Tensor,
    tail_read_ids: torch.Tensor,
    positions: torch.Tensor,
    query_lens: Sequence[int],
    rate: int,
    norm_epsilon: float,
    *,
    valid_rows: torch.Tensor | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Project contiguous requests with replayable, device-resident geometry.

    Graph buckets may contain empty requests or a padded query suffix. Keep
    one harmless token for empty CANN calls and gate every persistent write
    using the original masks. No device value controls Python execution.
    """
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_positions = positions.reshape(-1)
    flat_valid = valid_rows.reshape(-1).to(torch.bool) if valid_rows is not None else None
    head_dim = key_weight.shape[0]
    key_weight = key_weight.contiguous()
    gate_weight = gate_weight.to(hidden.dtype).contiguous()
    ape = ape.float().contiguous()
    norm_weight = norm_weight.float()
    norm_bias = norm_bias.float()
    if (
        flat_valid is not None
        and hidden.device.type == "npu"
        and query_lens
        and 0 < query_lens[0] <= 8
        and all(length == query_lens[0] for length in query_lens)
        and rate == 4
        and head_dim <= 128
        and tail_cache.is_contiguous()
    ):
        from xllm.python.kernels_npu.triton.kpool_compress import prepare_fused_kpool_graph

        _, table, _ = _fused_kpool_metadata(query_lens[0], rate, hidden.device)
        return prepare_fused_kpool_graph(
            hidden.contiguous(),
            key_weight,
            gate_weight,
            ape,
            norm_weight,
            norm_bias,
            tail_cache,
            tail_read_ids,
            positions,
            flat_valid,
            query_lens[0],
            rate,
            norm_epsilon,
            table,
        )
    states = torch.zeros((len(query_lens), 3, rate, 2 * head_dim), dtype=torch.float32, device=hidden.device)
    results = []
    row_start = 0
    for request_idx, query_len in enumerate(query_lens):
        if query_len == 0:
            results.append((hidden.new_empty((0, head_dim)), tail_cache.new_empty((2, rate, head_dim))))
            continue
        row_end = row_start + query_len
        if flat_valid is None:
            offsets, table, cu_seqlens = _fused_kpool_metadata(query_len, rate, hidden.device)
            actual_length = query_len
        else:
            offsets = torch.arange(rate, device=hidden.device)
            table = torch.full((1, 1 + (query_len + rate - 1) // rate), 2, dtype=torch.int32, device=hidden.device)
            table[:, 0] = 1
            actual_length = flat_valid[row_start:row_end].sum(dtype=torch.int32)
            cu_seqlens = torch.stack((torch.zeros_like(actual_length), actual_length.clamp_min(1)))
        first_position = flat_positions[row_start].clamp_min(0)
        start = torch.remainder(first_position, rate).to(torch.int32).reshape(1)
        history_positions = first_position - start[0] + offsets
        tail_id = tail_read_ids.reshape(-1)[request_idx].long().clamp(0, tail_cache.shape[0] - 1)
        history = tail_cache.index_select(0, tail_id.reshape(1))[0]
        history = history.index_select(1, torch.remainder(history_positions, tail_cache.shape[2]))
        # Separate request-local scratch from persistent tail storage. CANN
        # writes only the unfinished pool, while MTP retains a larger window.
        state = states[request_idx]
        state[1].copy_(history.permute(1, 0, 2).reshape(rate, 2 * head_dim))
        pooled = torch.ops.xllm_ops.key_pool(
            hidden[row_start:row_end].contiguous(),
            key_weight,
            gate_weight,
            ape,
            state,
            table,
            start,
            norm_weight,
            norm_bias,
            cu_seqlens,
            rate,
            norm_epsilon,
        )
        final_block = torch.where(start[0] + actual_length <= rate, 1, 2).long().reshape(1)
        tail = state.index_select(0, final_block)[0].reshape(rate, 2, head_dim).permute(1, 0, 2)
        results.append((pooled, tail))
        row_start = row_end
    return results


def append_causal_tail(
    selected_indices: torch.Tensor,
    query_positions: torch.Tensor,
    rate: int,
) -> torch.Tensor:
    """Append the query-visible incomplete pool and keep valid slots leading."""
    tail_width = rate - 1
    if tail_width <= 0:
        return selected_indices
    positions = query_positions.to(torch.int64)
    offsets = torch.arange(tail_width, dtype=torch.int64, device=positions.device)
    tail_start = torch.div(positions + 1, rate, rounding_mode="floor") * rate
    tail = tail_start[..., None] + offsets
    tail = tail.masked_fill((tail > positions[..., None]) | (positions[..., None] < 0), -1)
    combined = torch.cat([selected_indices, tail], dim=-1)

    # SFA stops consuming a row at its first -1. Causal filtering can leave
    # holes in the scored pool prefix, so stably pack every valid slot before
    # the invalid suffix after appending the unscored tail.
    width = combined.shape[-1]
    original_order = torch.arange(width, dtype=torch.float32, device=combined.device).expand_as(combined)
    pack_keys = original_order + (combined < 0).to(torch.float32) * width
    _, pack_order = torch.sort(pack_keys, dim=-1)
    return torch.gather(combined, dim=-1, index=pack_order)


def read_pools(pool_cache: torch.Tensor, block_table: torch.Tensor, kv_lens: torch.Tensor, n_pools: int, rate: int):
    """Direct read of the pool cache -> ``(pool_keys [B,P,D], pool_indices [B,P,rate] int64,
    pool_valid [B,P] bool)``.

    ``pool_valid = 4(p+1) <= kv_len`` is equivalent to the old path's "4 slots all valid";
    the -1 mask of ``pool_indices`` reuses the old ``slot_valid`` semantics
    (``4p+s >= kv_len[b]`` -> set to -1, per sequence).
    """
    pool_bs = pool_cache.shape[1]
    head_dim = pool_cache.shape[-1]
    device = pool_cache.device
    B = block_table.shape[0]
    pool_flat = pool_cache.reshape(-1, head_dim)
    offs = torch.arange(n_pools, device=device)
    # [B, P] = bt[b, p // pool_bs] * pool_bs + p % pool_bs
    blk = (offs // pool_bs).clamp(max=block_table.shape[1] - 1)
    pool_slots = block_table[:, blk].clamp(min=0) * pool_bs + offs[None, :] % pool_bs
    pool_keys = pool_flat[pool_slots.reshape(-1)].reshape(B, n_pools, head_dim)
    slot_off = torch.arange(rate, device=device)
    member = offs[:, None] * rate + slot_off[None, :]  # [P, rate]
    # -1 mask aligns with the old slot_valid (key_valid & in_range): mask per sequence by kv_len
    member_b = member[None].expand(B, -1, -1)
    pool_indices = member_b.masked_fill(member_b >= kv_lens.reshape(-1, 1, 1), -1).contiguous()
    pool_valid = (offs[None, :] + 1) * rate <= kv_lens.reshape(-1, 1)
    return pool_keys, pool_indices, pool_valid
