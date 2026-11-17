# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Resident rows and causal metadata for sparse KV offloading.

The caller owns the mapped Host allocation and its lifetime. Query ordinals
keep separate resident selections, even when several queries share a request.
Current-window KV comes from the supplied device rows rather than racing Host
appends performed by other Serve cores.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from xllm.python.kernels_npu.sparse_attention import RESIDENT_HEAD_DIM, RESIDENT_PAGE_SIZE, SELECTION_CAPACITY


class ResidentRows:
    """Assign fixed selection rows without moving surviving requests' KV."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Resident row capacity must be positive.")
        self._capacity = capacity
        self._owners: tuple[tuple[str, int, int] | None, ...] = (None,) * capacity

    @property
    def owners(self) -> tuple[tuple[str, int, int] | None, ...]:
        return self._owners

    def prepare(
        self,
        request_ids: Sequence[str],
        widths: Sequence[int],
        instance_ids: Sequence[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return query addresses and rows to invalidate before using them.

        Instance ids distinguish new sequences reusing a client request id.
        An empty/dummy step preserves owners; it is not a retirement signal.
        A nonempty schedule releases absent identities and query ordinals.
        Validation finishes before the ownership map changes.
        """
        ids = tuple(request_ids)
        counts = tuple(int(width) for width in widths)
        instances = tuple(int(instance) for instance in instance_ids)
        if len(counts) != len(ids) or len(instances) != len(ids):
            raise ValueError("KV offloading requests require one query width and instance id per request.")
        if len(set(ids)) != len(ids) or any(not request for request in ids):
            raise ValueError("KV offloading live request ids must be nonempty and distinct.")
        if any(count <= 0 for count in counts):
            raise ValueError("KV offloading live query widths must be positive.")
        if sum(counts) > self._capacity:
            raise ValueError("KV offloading scheduled query rows exceed the resident capacity.")
        identities = tuple(
            (request, instance, ordinal)
            for request, instance, count in zip(ids, instances, counts)
            for ordinal in range(count)
        )
        if not identities:
            return (), ()

        existing = {owner: row for row, owner in enumerate(self._owners) if owner is not None}
        wanted = set(identities)
        owners = [owner if owner in wanted else None for owner in self._owners]
        released = {row for row, (old, new) in enumerate(zip(self._owners, owners)) if old != new}
        free = iter(row for row, owner in enumerate(owners) if owner is None)
        active = []
        for identity in identities:
            row = existing.get(identity)
            if row is None:
                row = next(free)
                owners[row] = identity
                released.add(row)
            active.append(row)
        self._owners = tuple(owners)
        return tuple(active), tuple(sorted(released))


def is_supported_host_kv_layout(host: torch.Tensor, topk_width: int) -> bool:
    """Check geometry; the native adapter checks the mapped device alias."""
    return (
        host.device.type == "cpu"
        and host.dtype == torch.bfloat16
        and host.is_contiguous()
        and host.ndim == 4
        and host.shape[0] > 0
        and tuple(host.shape[1:]) == (RESIDENT_PAGE_SIZE, 1, RESIDENT_HEAD_DIM)
        and 0 < topk_width <= SELECTION_CAPACITY
    )


def pad_topk_indices(buffer: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Pad this layer's top-k to the fixed vendor capacity, clearing old tails."""
    rows = indices.shape[0]
    out = buffer[:rows]
    width = indices.shape[-1]
    out[:, :, :width].copy_(indices)
    out[:, :, width:].fill_(-1)
    return out


def build_decode_layout(kv_len: torch.Tensor, query_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One current token per request, with a square current-row-map."""
    if kv_len.ndim != 1 or kv_len.numel() != query_rows or query_rows <= 0:
        raise ValueError("KV offloading single-token decode requires one KV length per query row.")
    rows = torch.arange(query_rows, dtype=torch.int32, device=kv_len.device)
    descriptor = torch.stack((rows, torch.ones_like(rows), kv_len.to(torch.int32), rows), dim=1)
    return descriptor.contiguous(), rows[:, None].expand(query_rows, query_rows).contiguous()


def build_verify_layout(
    kv_len: torch.Tensor,
    query_lens: Sequence[int],
    *,
    table_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bound each verify query to its own request's causal current prefix.

    KV lengths and block tables may be sequence-scoped or expanded per query.
    Padding columns in the square row map repeat the group's first row; only
    the descriptor's causal window is consumed by the native operator.
    """
    widths = tuple(int(width) for width in query_lens)
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("KV offloading verify query groups must be positive.")
    rows = sum(widths)
    if kv_len.ndim != 1 or kv_len.numel() not in (len(widths), rows):
        raise ValueError("KV offloading verify KV lengths must cover sequences or every query row.")
    if table_rows not in (len(widths), rows):
        raise ValueError("KV offloading verify block table must cover sequences or every query row.")
    device = kv_len.device
    counts = torch.tensor(widths, dtype=torch.int64, device=device)
    row_ids = torch.arange(rows, dtype=torch.int64, device=device)
    requests = torch.repeat_interleave(
        torch.arange(len(widths), dtype=torch.int64, device=device), counts, output_size=rows
    )
    starts = torch.cat((counts.new_zeros(1), counts.cumsum(0)[:-1]))[requests]
    offsets = row_ids - starts
    visible = kv_len if kv_len.numel() == rows else kv_len[requests] - counts[requests] + offsets + 1
    cache_rows = row_ids if table_rows == rows else requests
    descriptor = torch.stack((cache_rows, offsets + 1, visible, row_ids), dim=1).to(torch.int32).contiguous()
    columns = torch.arange(rows, dtype=torch.int64, device=device)
    row_map = torch.where(columns[None, :] < counts[requests, None], starts[:, None] + columns, starts[:, None])
    return descriptor, row_map.to(torch.int32).contiguous()
