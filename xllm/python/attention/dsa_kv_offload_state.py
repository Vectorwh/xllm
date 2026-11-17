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

"""Selection storage and same-step RoPE plans; no Host pool or transport."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.attention.sparse_kv_offload import ResidentRows
from xllm.python.kernels_npu.sparse_attention import (
    RESIDENT_HEAD_DIM,
    RESIDENT_PAGE_SIZE,
    RESIDENT_ROPE_DIM,
    SELECTION_BLOCKS,
    SELECTION_CAPACITY,
)


@dataclass(frozen=True)
class DsaKvStepPlan:
    """Read-only owner plan reused against each shared layer's own KV."""

    owner: int
    rows: int
    topk_count: int
    metadata: AttentionMetadata
    cache_is_preprocessed: bool
    topk: torch.Tensor
    append: torch.Tensor
    tokens: torch.Tensor
    slots: torch.Tensor
    indices: torch.Tensor
    actual: torch.Tensor
    table: torch.Tensor


@dataclass(frozen=True)
class DsaKvLayerState:
    selection: torch.Tensor
    selection_rope: torch.Tensor | None
    resident: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    kernel_indices: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    compact_indices: torch.Tensor | None = None
    actual: torch.Tensor | None = None
    table: torch.Tensor | None = None
    plan_tokens: torch.Tensor | None = None
    plan_slots: torch.Tensor | None = None


class DsaKvOffloadState:
    """An explicitly configured backend's computation-local ownership."""

    def __init__(self, device: torch.device, capacity: int) -> None:
        self.device = device
        self.capacity = capacity
        self.enabled = False
        self.rope_dim = 0
        self.plan_owners: tuple[int, ...] = ()
        self.layers: dict[int, DsaKvLayerState] = {}
        self.resident_rows: ResidentRows | None = None
        self.active_rows: tuple[int, ...] = ()
        self.query_lens: tuple[int, ...] = ()
        self.requests_prepared = False
        self.step_metadata: AttentionMetadata | None = None
        self.step_descriptor: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.step_plan: DsaKvStepPlan | None = None
        self.last_layer = -1

    def reset(self) -> None:
        """Invalidate old selections when their source cache binding changes."""
        self.layers.clear()
        self.resident_rows = ResidentRows(self.capacity) if self.enabled else None
        self.active_rows = ()
        self.query_lens = ()
        self.requests_prepared = False
        self.step_metadata = None
        self.step_descriptor = None
        self.step_plan = None
        self.last_layer = -1

    def prepare_requests(
        self,
        request_ids: Sequence[str],
        num_scheduled_tokens: Sequence[int],
        instance_ids: Sequence[int],
    ) -> None:
        if not self.enabled:
            return
        self.require_complete_step()
        resident_rows = cast(ResidentRows, self.resident_rows)
        active, reset = resident_rows.prepare(request_ids, num_scheduled_tokens, instance_ids)
        for row in reset:
            for layer in self.layers.values():
                if layer.resident is not None:
                    layer.resident[row].fill_(-1)
                    layer.slots[row].fill_(-1)
                    layer.valid[row].zero_()
        self.active_rows = active
        self.query_lens = tuple(int(width) for width in num_scheduled_tokens)
        self.requests_prepared = True
        self.step_metadata = None
        self.step_descriptor = None
        self.step_plan = None
        self.last_layer = -1

    def require_complete_step(self) -> None:
        """A shared layer cannot inherit hits from a forward it never served."""
        if self.rope_dim and 0 <= self.last_layer < len(self.plan_owners) - 1:
            raise RuntimeError(
                "RoPE KV offloading forwards must execute every configured full/shared layer before the next step."
            )

    def write_resident_rows(self, descriptor: torch.Tensor, active_rows: int) -> None:
        """Fill resident addresses from the caller's request/instance identities."""
        rows = descriptor.shape[0]
        if descriptor.ndim != 2 or descriptor.shape[1] != 4 or not 0 <= active_rows <= rows <= self.capacity:
            raise ValueError("KV offloading descriptor exceeds the resident row capacity.")
        if len(self.active_rows) != active_rows:
            raise ValueError("KV offloading live query rows require prepared request identities.")
        addresses = (*self.active_rows, *((0,) * (rows - active_rows)))
        descriptor[:, 3].copy_(torch.tensor(addresses, device=descriptor.device, dtype=torch.int32))

    def ensure_layer(self, layer_id: int, query_rows: int) -> DsaKvLayerState:
        if not 0 < query_rows <= self.capacity:
            raise ValueError("KV offloading query rows exceed the configured selection capacity.")
        existing = self.layers.get(layer_id)
        if existing is not None:
            return existing
        device = self.device
        capacity = self.capacity
        selection = torch.empty(
            capacity * SELECTION_BLOCKS, RESIDENT_PAGE_SIZE, 1, RESIDENT_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        selection_rope = (
            torch.empty(
                capacity * SELECTION_BLOCKS,
                RESIDENT_PAGE_SIZE,
                1,
                RESIDENT_ROPE_DIM,
                dtype=torch.bfloat16,
                device=device,
            )
            if self.rope_dim
            else None
        )
        if self.rope_dim and self.plan_owners[layer_id] != layer_id:
            state = DsaKvLayerState(selection, selection_rope)
        else:
            shape = (capacity, 1, SELECTION_CAPACITY)
            state = DsaKvLayerState(
                selection=selection,
                selection_rope=selection_rope,
                resident=torch.full(shape, -1, dtype=torch.int32, device=device),
                slots=torch.full(shape, -1, dtype=torch.int32, device=device),
                valid=torch.zeros(capacity, dtype=torch.int32, device=device),
                kernel_indices=torch.full(shape, -1, dtype=torch.int32, device=device),
                indices=torch.empty(shape, dtype=torch.int32, device=device),
                compact_indices=torch.empty(shape, dtype=torch.int32, device=device) if not self.rope_dim else None,
                actual=torch.empty(capacity, dtype=torch.int32, device=device),
                table=torch.empty(capacity, SELECTION_BLOCKS, dtype=torch.int32, device=device),
                plan_tokens=torch.empty(shape, dtype=torch.int32, device=device) if self.rope_dim else None,
                plan_slots=torch.empty(shape, dtype=torch.int32, device=device) if self.rope_dim else None,
            )
        self.layers[layer_id] = state
        return state
