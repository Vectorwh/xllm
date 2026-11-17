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

"""NoPE/RoPE attention over caller-owned mapped Host KV.

The paged backend supplies its existing metadata and sparse attention entry.
This mixin owns only selection/resident computation, never PD allocations,
network transfers, physical page retirement, or ACL Graph capture/replay.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

from scripts.logger import logger
from xllm.python import kernels
from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.attention.dsa_kv_offload_state import DsaKvOffloadState, DsaKvStepPlan
from xllm.python.attention.sparse_kv_offload import (
    build_decode_layout,
    build_verify_layout,
    is_supported_host_kv_layout,
    pad_topk_indices,
)
from xllm.python.kernels_npu.sparse_attention import (
    RESIDENT_HEAD_DIM,
    RESIDENT_PAGE_SIZE,
    RESIDENT_ROPE_DIM,
    SELECTION_CAPACITY,
)
from xllm.python.model_executor.forward_context import get_forward_context, in_acl_graph

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


class DsaKvOffloadMixin:
    """KV offloading methods used only by explicitly configured NPU paged backends."""

    _offload_state: DsaKvOffloadState | None = None
    _offload_topk_width: int
    dtype: torch.dtype
    _kv_caches: list[LayerCache]
    _metadata: AttentionMetadata | None
    _block_table_i32: torch.Tensor | None

    if TYPE_CHECKING:

        @property
        def num_kv_blocks(self) -> int: ...

        def _mla_sparse(
            self,
            q_latent: torch.Tensor,
            q_pe: torch.Tensor | None,
            nope_cache: torch.Tensor,
            rope_cache: torch.Tensor | None,
            topk: torch.Tensor,
            block_table: torch.Tensor,
            actual_seq_q: torch.Tensor,
            actual_seq_kv: torch.Tensor,
            layer_id: int,
        ) -> torch.Tensor: ...

    @property
    def uses_host_main_kv(self) -> bool:
        return self._offload_state is not None and self._offload_state.enabled

    def configure_kv_offload_layers(self, layers: Sequence[Attention]) -> None:
        """Bind full/shared roles before Host caches or selection are bound."""
        state = self._offload_state
        if state is None:
            raise ValueError("KV offloading layer roles require offload-capable backend construction.")
        if state.enabled:
            raise ValueError("KV offloading layer roles must be configured before binding Host caches.")
        ordered = sorted(layers, key=lambda layer: layer.layer_id)
        if not ordered:
            return
        if [layer.layer_id for layer in ordered] != list(range(len(ordered))):
            raise ValueError("KV offloading layer roles must cover contiguous cache layer ids.")
        rope_dims = {int(layer.qk_rope_head_dim) for layer in ordered}
        if len(rope_dims) != 1 or rope_dims.pop() not in (0, RESIDENT_ROPE_DIM):
            raise ValueError("KV offloading layer roles require a uniform NoPE or RoPE64 layout.")
        state.rope_dim = int(ordered[0].qk_rope_head_dim)
        owner = -1
        owners = []
        for layer in ordered:
            if not bool(getattr(layer, "is_shared", False)):
                owner = layer.layer_id
            if owner < 0:
                raise ValueError("Shared KV offloading layer requires a preceding full indexer owner.")
            owners.append(owner)
        state.plan_owners = tuple(owners)

    def _bind_kv_offload_caches(self, kv_caches: list[LayerCache]) -> None:
        """Validate and allocate all layer storage before admitting a Host binding."""
        state = cast(DsaKvOffloadState, self._offload_state)
        host = any(cache.key is not None and cache.key.device.type == "cpu" for cache in kv_caches)
        if host:
            if self.dtype != torch.bfloat16:
                raise ValueError("Host KV offloading requires BF16 queries and selection storage.")
            full = [cache for cache in kv_caches if cache.key is not None]
            if not all(is_supported_host_kv_layout(cache.key, self._offload_topk_width) for cache in full):
                raise ValueError(
                    "Host KV offloading requires contiguous BF16 page128/latent512 caches on every attention layer."
                )
            if state.rope_dim:
                if len(state.plan_owners) != len(kv_caches):
                    raise ValueError(
                        "RoPE Host KV offloading requires configured full/shared roles for every cache layer."
                    )
                if any(
                    cache.value is None
                    or cache.value.device.type != "cpu"
                    or cache.value.dtype != torch.bfloat16
                    or not cache.value.is_contiguous()
                    or tuple(cache.value.shape) != (*cache.key.shape[:-1], RESIDENT_ROPE_DIM)
                    for cache in full
                ):
                    raise ValueError(
                        "RoPE Host KV offloading requires matching contiguous BF16 latent512/RoPE64 caches."
                    )
                if len(full) != len(kv_caches) or any(
                    (cache.index is not None) != (state.plan_owners[layer_id] == layer_id)
                    for layer_id, cache in enumerate(kv_caches)
                ):
                    raise ValueError("RoPE KV offloading roles disagree with the full/shared indexer cache allocation.")
                native_ready = kernels.has_stable_rope_resident_slots()
            else:
                if any(cache.value is not None for cache in full):
                    raise ValueError(
                        "NoPE Host KV offloading requires latent-only caches; configure RoPE roles for rotary KV."
                    )
                native_ready = kernels.has_stable_resident_slots()
            if not native_ready:
                raise ValueError(
                    f"Host KV offloading requires the complete stable resident vendor ABI with capacity {SELECTION_CAPACITY}."
                )
        unchanged = len(kv_caches) == len(self._kv_caches) and all(
            new.key is old.key and new.value is old.value and new.index is old.index
            for new, old in zip(kv_caches, self._kv_caches)
        )
        if unchanged:
            return
        if host:
            replacement = DsaKvOffloadState(state.device, state.capacity)
            replacement.rope_dim = state.rope_dim
            replacement.plan_owners = state.plan_owners
            replacement.enabled = True
            replacement.reset()
            payload_bytes = len(full) * state.capacity * SELECTION_CAPACITY * (RESIDENT_HEAD_DIM + state.rope_dim) * 2
            logger.info(
                f"Allocating Host KV offloading selection: layers={len(full)}, query_capacity={state.capacity}, "
                f"payload_bytes={payload_bytes}, device={state.device}; resident metadata is additional."
            )
            for layer_id, cache in enumerate(kv_caches):
                if cache.key is not None:
                    replacement.ensure_layer(layer_id, replacement.capacity)
            self._offload_state = replacement
        else:
            state.enabled = False
            state.reset()

    def prepare_kv_offload_requests(
        self,
        request_ids: Sequence[str],
        num_scheduled_tokens: Sequence[int],
        instance_ids: Sequence[int],
    ) -> None:
        """Supply request identities before each KV offloading prepare/execute step.

        Instance ids must change when a new sequence reuses a request id.
        Calls on a default or Device-KV backend perform no KV offloading work.
        """
        if self.uses_host_main_kv:
            self._offload_state.prepare_requests(request_ids, num_scheduled_tokens, instance_ids)

    def _prepare_kv_offload_step(self, metadata: AttentionMetadata, *, graph_mode: bool) -> None:
        state = cast(DsaKvOffloadState, self._offload_state)
        if graph_mode or in_acl_graph():
            raise ValueError(
                "Host KV offloading attention currently requires Eager execution; ACL Graph is a separate integration."
            )
        if (metadata.is_prefill or metadata.is_chunked_prefill or metadata.is_mixed) and not (metadata.is_spec_verify):
            raise ValueError(
                "Host KV offloading computes Decode/target verify; ordinary prefill keeps main KV on device."
            )
        if metadata.has_kv_shard:
            raise ValueError("Host KV offloading requires an unsharded Decode KV layout.")
        if not state.requests_prepared:
            raise ValueError("Prepare KV offloading request identities before each attention step.")
        slot_mapping = metadata.slot_mapping
        kv_len = metadata.kv_seq_lens
        block_table = metadata.block_table
        if slot_mapping is None or slot_mapping.ndim != 1 or slot_mapping.dtype not in (torch.int32, torch.int64):
            raise ValueError("Host KV offloading requires one integer append slot per query row.")
        rows = slot_mapping.numel()
        if not 0 < rows <= state.capacity:
            raise ValueError("Host KV offloading query rows exceed the configured resident capacity.")
        if kv_len is None or kv_len.ndim != 1 or kv_len.dtype not in (torch.int32, torch.int64):
            raise ValueError("Host KV offloading requires integer KV lengths.")
        if (
            block_table is None
            or block_table.ndim != 2
            or block_table.shape[1] == 0
            or block_table.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Host KV offloading requires an integer paged block table.")
        for tensor in (slot_mapping, kv_len, block_table):
            if tensor.device.type != state.device.type or state.device.index not in (None, tensor.device.index):
                raise ValueError("KV offloading metadata and selection must share the configured execution device.")
        violations = [((kv_len < 0) | (kv_len > torch.iinfo(torch.int32).max)).any()]
        errors = ["KV offloading KV lengths must fit the nonnegative native INT32 layout."]
        dummy = bool(metadata.is_dummy)
        if dummy:
            if state.active_rows:
                raise ValueError("A dummy KV offloading step requires an empty request schedule.")
            descriptor, row_map = build_decode_layout(kv_len, rows)
            descriptor[:, 1:3].zero_()
        else:
            if len(state.active_rows) != rows:
                raise ValueError("KV offloading live query rows disagree with the prepared request identities.")
            if metadata.is_spec_verify:
                descriptor, row_map = build_verify_layout(kv_len, state.query_lens, table_rows=block_table.shape[0])
            else:
                if any(width != 1 for width in state.query_lens) or block_table.shape[0] != rows:
                    raise ValueError("KV offloading multi-query groups require causal target-verify metadata.")
                descriptor, row_map = build_decode_layout(kv_len, rows)
            violations.append(
                (
                    (descriptor[:, 2] < descriptor[:, 1])
                    | (descriptor[:, 2] > block_table.shape[1] * RESIDENT_PAGE_SIZE)
                ).any()
            )
            needed = torch.zeros(block_table.shape[0], dtype=torch.int64, device=block_table.device)
            needed.scatter_reduce_(0, descriptor[:, 0].long(), descriptor[:, 2].long(), reduce="amax")
            columns = torch.arange(block_table.shape[1], device=block_table.device) * RESIDENT_PAGE_SIZE
            required = columns[None, :] < needed[:, None]
            page_limit = min(self.num_kv_blocks, torch.iinfo(torch.int32).max + 1)
            violations.append((((block_table < 0) | (block_table >= page_limit)) & required).any())
            slot_limit = min(self.num_kv_blocks * RESIDENT_PAGE_SIZE, torch.iinfo(torch.int32).max + 1)
            violations.append(((slot_mapping < -1) | (slot_mapping >= slot_limit)).any())
            appends = slot_mapping.sort().values
            violations.append(((appends[1:] >= 0) & (appends[1:] == appends[:-1])).any())
            positions = (descriptor[:, 2].long() - 1).clamp(0, block_table.shape[1] * RESIDENT_PAGE_SIZE - 1)
            physical = (
                block_table[descriptor[:, 0].long(), positions // RESIDENT_PAGE_SIZE].long() * RESIDENT_PAGE_SIZE
                + positions % RESIDENT_PAGE_SIZE
            )
            violations.append(((slot_mapping >= 0) & (slot_mapping != physical)).any())
            errors.extend(
                (
                    "KV offloading visible KV lengths exceed the current window or Host block-table capacity.",
                    "KV offloading block table addresses an invalid physical Host page.",
                    "KV offloading append slots address an invalid physical Host row.",
                    "KV offloading current rows must not append to the same physical Host slot.",
                    "KV offloading append slots disagree with the causal query positions.",
                )
            )
        for violated, error in zip(torch.stack(violations).cpu().tolist(), errors, strict=True):
            if violated:
                raise ValueError(error)
        state.write_resident_rows(descriptor, 0 if dummy else rows)
        state.step_metadata = metadata
        query_ends = torch.arange(1, rows + 1, dtype=torch.int32, device=descriptor.device)
        state.step_descriptor = (descriptor, row_map, query_ends)
        state.step_plan = None
        state.last_layer = -1
        state.requests_prepared = False

    def _execute_kv_offload(
        self,
        query: torch.Tensor,
        query_rope: torch.Tensor | None,
        current: torch.Tensor | None,
        current_rope: torch.Tensor | None,
        layer: Attention,
        topk: torch.Tensor | None,
        cache_is_preprocessed: bool,
    ) -> torch.Tensor:
        state = cast(DsaKvOffloadState, self._offload_state)
        metadata = self._metadata
        if metadata is None or state.step_metadata is not metadata or state.step_descriptor is None:
            raise ValueError(
                "Host KV offloading execute requires this step's prepared metadata and request identities."
            )
        if in_acl_graph() or get_forward_context().cp_context is not None:
            raise ValueError("Host KV offloading currently supports Eager Decode without context parallelism.")
        rows = state.step_descriptor[0].shape[0]
        if topk is None:
            raise ValueError("Host KV offloading requires sparse DSA top-k indices.")
        indices = topk
        if indices.ndim == 4 and indices.shape[1:3] == (1, 1):
            indices = indices[:, 0]
        if indices.ndim == 2:
            indices = indices[:, None]
        if (
            indices.ndim != 3
            or indices.shape[:2] != (rows, 1)
            or not 0 < indices.shape[-1] <= self._offload_topk_width
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.device != query.device
        ):
            raise ValueError("KV offloading top-k must be integer [query, 1, width] within the configured top-k width.")
        if indices.dtype == torch.int64 and bool(
            ((indices < torch.iinfo(torch.int32).min) | (indices > torch.iinfo(torch.int32).max)).any()
        ):
            raise ValueError("KV offloading top-k token ids must fit the native INT32 layout.")
        indices = indices.to(torch.int32)
        if (
            query.ndim != 3
            or query.shape[0] != rows
            or query.shape[1] <= 0
            or query.shape[-1] != RESIDENT_HEAD_DIM
            or query.dtype != torch.bfloat16
            or current is None
            or current.shape != (rows, 1, RESIDENT_HEAD_DIM)
            or current.dtype != query.dtype
            or current.device != query.device
            or query.device != self._block_table_i32.device
        ):
            raise ValueError(
                "KV offloading requires BF16 latent512 queries and complete current KV rows on the metadata device."
            )
        layer_id = layer.layer_id
        rope = int(getattr(layer, "qk_rope_head_dim", 0))
        if rope != state.rope_dim:
            raise ValueError("KV offloading layer rotary dimension disagrees with the admitted Host layout.")
        if rope and (
            query_rope is None
            or query_rope.shape != (*query.shape[:2], RESIDENT_ROPE_DIM)
            or query_rope.dtype != query.dtype
            or query_rope.device != query.device
            or current_rope is None
            or current_rope.shape != (rows, 1, RESIDENT_ROPE_DIM)
            or current_rope.dtype != query.dtype
            or current_rope.device != query.device
        ):
            raise ValueError("RoPE KV offloading requires this layer's BF16 query/current RoPE64 rows.")
        if rope and layer_id != state.last_layer + 1:
            raise ValueError("KV offloading RoPE full/shared layers must execute in configured order within each step.")
        cache = self._kv_caches[layer_id]
        descriptor, current_map, query_ends = state.step_descriptor
        current = current.contiguous()
        if rope:
            current_rope = current_rope.contiguous()
        topk_count = indices.shape[-1]
        shared = bool(rope and state.plan_owners[layer_id] != layer_id)
        if shared:
            plan = state.step_plan
            if (
                plan is None
                or plan.owner != state.plan_owners[layer_id]
                or plan.metadata is not metadata
                or plan.rows != rows
                or plan.topk_count != topk_count
                or plan.cache_is_preprocessed != cache_is_preprocessed
                or not torch.equal(plan.topk, indices)
            ):
                raise ValueError("Shared KV offloading layer requires its full owner's same-step top-k and plan.")
            storage = state.ensure_layer(layer_id, rows)
            kernels.dsa_kv_offloading_serve_rope_planned_out(
                cache.key,
                cache.value,
                self._block_table_i32,
                descriptor,
                current_map,
                plan.tokens,
                plan.slots,
                plan.append,
                current,
                current_rope,
                storage.selection,
                storage.selection_rope,
            )
            sparse_indices, actual, table = plan.indices, plan.actual, plan.table
        else:
            storage = state.ensure_layer(layer_id, rows)
            padded = pad_topk_indices(storage.kernel_indices, indices)
            append = (
                torch.full((rows,), -1, dtype=torch.int32, device=query.device)
                if cache_is_preprocessed or metadata.is_dummy
                else metadata.slot_mapping.to(torch.int32).contiguous()
            )
            if rope:
                kernels.dsa_kv_offloading_update_rope_with_plan_out(
                    cache.key,
                    self._block_table_i32,
                    descriptor,
                    current_map,
                    padded,
                    append,
                    current,
                    storage.selection,
                    storage.indices[:rows],
                    storage.actual[:rows],
                    storage.table[:rows],
                    storage.resident,
                    storage.slots,
                    storage.valid,
                    cache.value,
                    current_rope,
                    storage.selection_rope,
                    storage.plan_tokens[:rows],
                    storage.plan_slots[:rows],
                )
                sparse_indices = storage.indices[:rows, :, :topk_count].contiguous()
                state.step_plan = DsaKvStepPlan(
                    owner=layer_id,
                    rows=rows,
                    topk_count=topk_count,
                    metadata=metadata,
                    cache_is_preprocessed=cache_is_preprocessed,
                    topk=padded[:, :, :topk_count],
                    append=append,
                    tokens=storage.plan_tokens[:rows],
                    slots=storage.plan_slots[:rows],
                    indices=sparse_indices,
                    actual=storage.actual[:rows],
                    table=storage.table[:rows],
                )
            else:
                kernels.dsa_kv_offloading_update_nope_out(
                    cache.key,
                    self._block_table_i32,
                    descriptor,
                    current_map,
                    padded,
                    append,
                    current,
                    storage.selection,
                    storage.indices[:rows],
                    storage.actual[:rows],
                    storage.table[:rows],
                    storage.resident,
                    storage.slots,
                    storage.valid,
                    storage.compact_indices[:rows],
                )
                sparse_indices = storage.compact_indices[:rows, :, :topk_count].contiguous()
            actual, table = storage.actual[:rows], storage.table[:rows]
        result = self._mla_sparse(
            query,
            query_rope if rope else None,
            storage.selection,
            storage.selection_rope,
            sparse_indices,
            table,
            query_ends,
            actual,
            layer_id,
        )
        if rope:
            state.last_layer = layer_id
        return result
