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

"""Unit tests for the Python DSA metadata builder.

Validates the faithful port of ``DSAMetadataBuilder`` (core/layers/common/
dsa_metadata_builder.cpp) against the cache-spec and slot-expansion rules the
C++ implementation enforces. These tests are pure Python: they do not load the
compiled NPU operators, so they run anywhere.
"""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    READ_TABLE_LAYOUT_RIGHT_ALIGNED,
    READ_TABLE_LAYOUT_RING,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs,
    build_cache_specs_v41,
    dsa_block_table_groups,
)


@pytest.mark.parametrize("ratios", [[4, 128], [128, 4], [1, 2], [2, 1], [1], [2]])
@pytest.mark.parametrize("enable_graph", [False, True])
def test_logical_compressed_table_populates_all_physical_groups(ratios: list[int], enable_graph: bool) -> None:
    if max(ratios) <= 2:
        caches_info, groups = build_cache_specs_v41(ratios, list(range(len(ratios))), [], 128, len(ratios))
        token_span = 128
    else:
        caches_info, groups = build_cache_specs(ratios, 128, len(ratios))
        token_span = 2048
    # Physical group order must not select the logical source table.
    for layer in caches_info:
        for cache in layer:
            cache.group_id = len(groups) - cache.group_id - 1
    groups = list(reversed(groups))
    builder = DsaMetadataBuilder(caches_info, groups)
    swa = torch.arange(32, dtype=torch.int32).view(1, -1)
    compressed = torch.tensor([[7, 2, 9]], dtype=torch.int32)
    length = token_span + 128
    kwargs = dict(
        kv_seq_lens=[length],
        q_seq_lens=[128],
        positions=torch.arange(length - 128, length, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=True,
        enable_graph=enable_graph,
        graph_block_table_capacity_cols=32 if enable_graph else 0,
    )
    logical = builder.build(multi_block_tables=[swa, compressed], **kwargs)
    assert len(dsa_block_table_groups(groups)) == 2
    for layer_id, ratio in enumerate(ratios):
        slots = logical.slot_mappings[layer_id][0]
        physical_dim = next(
            group.physical_dim for group in groups if group.cache_type == DSA_CACHE_TOKEN and group.ratio == ratio
        )
        committed_row = token_span // ratio
        assert slots[0].item() == 2 * physical_dim + committed_row % physical_dim


def test_v41_kv_source_rejects_ratio_4() -> None:
    """V4.1 history planes exist only for ratios 1 and 2."""
    with pytest.raises(ValueError, match="compress ratio 1 or 2"):
        build_cache_specs_v41([4], [0], [], 128, 1)


def test_build_cache_specs_groups() -> None:
    """Group 0 is always SWA; TOKEN groups register in first-seen order."""
    compress_ratios = [0, 0, 128, 4, 128, 4, 128, 0]
    caches_info, group_infos = build_cache_specs(compress_ratios, swa_block_size=128, n_layers=8)

    # Non-sorted first occurrence matters: SWA, C128, then C4.
    assert len(group_infos) == 3
    assert group_infos[0].cache_type == DSA_CACHE_SLIDING_WINDOW
    assert group_infos[0].ratio == 1
    assert group_infos[1].cache_type == DSA_CACHE_TOKEN
    assert group_infos[1].ratio == 128
    assert group_infos[1].physical_dim == 16
    assert group_infos[2].cache_type == DSA_CACHE_TOKEN
    assert group_infos[2].ratio == 4
    assert group_infos[2].physical_dim == 512


def test_build_cache_specs_per_layer_cache_counts() -> None:
    """C1 -> 1 cache, C4 -> 8 caches, C128 -> 4 caches."""
    compress_ratios = [0, 4, 128]
    caches_info, _ = build_cache_specs(compress_ratios, 128, 3)

    assert [[cache.group_id for cache in layer] for layer in caches_info] == [
        [0],
        [1, 1, 0, 0, 0, 0, 0, 1],
        [2, 0, 0, 0],
    ]


def test_build_cache_specs_ignores_trailing_swa_layer() -> None:
    caches_info, group_infos = build_cache_specs([0, 4, 0], 128, 2)
    assert len(group_infos) == 2
    assert [[cache.group_id for cache in layer] for layer in caches_info] == [[0], [1, 1, 0, 0, 0, 0, 0, 1]]


def _make_builder(
    read_table_layout: str = READ_TABLE_LAYOUT_RIGHT_ALIGNED,
) -> tuple[DsaMetadataBuilder, list, list]:
    compress_ratios = [0, 4, 128, 4]
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, 4)
    return (
        DsaMetadataBuilder(caches_info, group_infos, read_table_layout=read_table_layout),
        caches_info,
        group_infos,
    )


@pytest.mark.parametrize("in_place", [False, True])
def test_host_slot_planning_avoids_per_token_tensor_reads(monkeypatch: pytest.MonkeyPatch, in_place: bool) -> None:
    """Strided tables, missing blocks and ring wrap need no scalar Tensor reads."""
    builder, _, _ = _make_builder()
    raw = torch.tensor([[2, 99, -1, 99, 7, 99], [9, 99, 10, 99, 11, 99]], dtype=torch.int32)[:, ::2]
    token_dst = torch.empty(9, dtype=torch.int32) if in_place else None
    swa_dst = torch.empty(16, dtype=torch.int32) if in_place else None

    def reject_scalar_read(_tensor: torch.Tensor) -> None:
        raise AssertionError("slot planning must read the block table in one batch")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", reject_scalar_read)
        _, token_slots = builder._process_token_group(raw, 2, 2, [10, 7], [8, 5], 2, 9, 0, dst_slots=token_dst)
        _, swa_slots = builder._process_swa_group(raw, 2, [10, 7], [8, 5], 2, 16, 0, None, dst_slots=swa_dst)

    assert token_slots.tolist() == [5, -1, -1, 14, 19, 20, -1, -1, -1]
    assert swa_slots.tolist() == [-1, -1, 14, 15, 4, 5, -1, -1, 20, 21, 22, 23, 18, -1, -1, -1]
    if in_place:
        assert token_slots is token_dst
        assert swa_slots is swa_dst


def test_build_seq_lengths_and_start_pos() -> None:
    """start_pos = kv_len - q_len per sequence."""
    builder, _, _ = _make_builder()
    # batch=2, decode (q_len=1 each).
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[6, 8],
        q_seq_lens=[1, 1],
        positions=torch.tensor([5, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    assert dsa.seq_lens.tolist() == [6, 8]
    assert dsa.seq_lens_q.tolist() == [1, 1]
    assert dsa.start_pos.tolist() == [5, 7]
    # actual_seq_lengths_query is cumsum(q_lens) with a leading zero.
    assert dsa.actual_seq_lengths_query.tolist() == [0, 1, 2]
    assert dsa.kv_cu_seq_lens.tolist() == [0, 6, 14]
    assert dsa.max_seqlen_q.ndim == 0 and dsa.max_seqlen_q.item() == 1
    assert dsa.max_seqlen_kv.ndim == 0 and dsa.max_seqlen_kv.item() == 8


@pytest.mark.parametrize(
    ("max_query_len", "max_seq_len", "expected"), [(16, 32, (16, 32)), (0, 0, (2, 8)), (1, 7, (2, 8))]
)
def test_build_max_lengths_include_attention_metadata_capacity(
    max_query_len: int, max_seq_len: int, expected: tuple[int, int]
) -> None:
    """C++ takes max(params.meta.max_*, max(host sequence lengths))."""
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[6, 8],
        q_seq_lens=[1, 2],
        positions=torch.tensor([5, 6, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
    )

    assert (dsa.max_query_len, dsa.max_seq_len) == expected


def test_build_token_group_slot_committed_rows() -> None:
    """A TOKEN cache commits one row per ratio boundary crossed this step."""
    builder, caches_info, group_infos = _make_builder()
    # group 0 = SWA, group 1 = TOKEN(4). Give each a [batch=1, cols=4] table.
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    # kv_len=8, q_len=1 (decode): prev_ctx_len=7, committed = 8//4 - 7//4 = 2 - 1 = 1.
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 1 (cr=4): cmp cache is caches_info[1][0] -> group 1 (TOKEN4).
    cmp_slot = dsa.slot_mappings[1][0]
    # One committed row: compressed_idx = prev_committed = 7//4 = 1.
    # block_idx = 1 // 512 = 0, block_id = token4_bt[0,0] = 20.
    # slot = 20 * 512 + 1 = 10241.
    assert cmp_slot.tolist() == [20 * 512 + 1]
    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 7]


def test_build_token_group_slot_empty_between_boundaries() -> None:
    """Eager decode uses an actual empty tensor when no row is committed."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[129],
        q_seq_lens=[1],
        positions=torch.tensor([128], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[1][0].numel() == 0


def test_build_token_group_slot_commits_at_later_boundary() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 37, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[2052],
        q_seq_lens=[1],
        positions=torch.tensor([2051], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[1][0].tolist() == [37 * 512]


def test_build_swa_group_uses_scheduler_resolved_slots_for_expanded_decode() -> None:
    """Expanded MTP decode preserves the scheduler's ring-buffer slot order."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10], [11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20], [21]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8, 8],
        q_seq_lens=[2, 2],
        positions=torch.tensor([6, 7, 6, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=[301, 302, 401, 402],
    )

    assert dsa.slot_mappings[0][0].tolist() == [301, 302, 401, 402]


def test_swa_device_path_routes_through_dst_slots_kwarg() -> None:
    """Vector context-length route preserves in-place slots and values."""
    builder, _, _ = _make_builder()
    raw_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    device_ctx_lens = torch.tensor([8], dtype=torch.int32)
    dst_slots = torch.full((1,), -1, dtype=torch.int32)

    _, out_slots = builder._process_swa_group(
        raw_bt=raw_bt,
        block_size=128,
        ctx_lens=[8],
        q_lens=[1],
        batch_size=1,
        graph_slot_capacity=0,
        graph_block_table_capacity_cols=0,
        new_cache_slots=None,
        swa_device_ctx_lens=device_ctx_lens,
        dst_slots=dst_slots,
    )

    assert out_slots is dst_slots
    # kv_len=8, q_len=1 -> q_start=7, block_idx=0, block_id=10 -> 10*128+7.
    assert out_slots[0].item() == 10 * 128 + 7


def _swa_read_table(
    raw_bt: torch.Tensor,
    ctx_lens: list[int],
    *,
    graph_cols: int = 0,
    layout: str = READ_TABLE_LAYOUT_RING,
) -> torch.Tensor:
    """Drive the real builder's SWA read-side table construction."""
    builder, _, _ = _make_builder(read_table_layout=layout)
    read_bt, _ = builder._process_swa_group(
        raw_bt=raw_bt,
        block_size=128,
        ctx_lens=list(ctx_lens),
        q_lens=[1] * len(ctx_lens),
        batch_size=len(ctx_lens),
        graph_slot_capacity=len(ctx_lens) if graph_cols > 0 else 0,
        graph_block_table_capacity_cols=graph_cols,
        new_cache_slots=None,
    )
    return read_bt


@pytest.mark.parametrize("ctx_len", [513])
def test_swa_read_table_rings_every_addressed_column(ctx_len: int) -> None:
    """A non-zero ring block must appear in EVERY column a reader addresses.

    The "ring" layout (requested by DeepSeek-V4.1) fills every column with the
    block for its own logical index, ``min(semantic_cols, ceil(ctx / block_size))``
    right-aligned columns carrying the ring is NOT enough: every read-side
    consumer addresses columns absolutely, so the older half of a multi-block
    window would resolve to block 0. With ``raw_bt = [[8]]`` (one ring block,
    id 8) every addressed column must read 8.
    """
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [ctx_len])
    addressed = (ctx_len + 127) // 128
    assert read_bt.shape == (1, addressed)
    assert read_bt[0, :addressed].tolist() == [8] * addressed


def test_swa_read_table_resolves_each_sequence_ring_block() -> None:
    """batch > 1: each row resolves to its OWN ring block, never block 0."""
    rings = [8, 9, 10, 11]
    raw_bt = torch.tensor([[ring] for ring in rings], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257] * len(rings))
    addressed = (257 + 127) // 128  # 3
    assert read_bt.shape == (len(rings), addressed)
    for seq, ring in enumerate(rings):
        assert read_bt[seq].tolist() == [ring] * addressed


def test_swa_read_table_modulos_by_semantic_cols() -> None:
    """Several ring blocks per sequence: column ``d`` carries ``raw[s, d % sc]``."""
    raw_bt = torch.tensor([[8, 9]], dtype=torch.int32)  # semantic_cols == 2
    read_bt = _swa_read_table(raw_bt, [385])
    addressed = (385 + 127) // 128  # 4
    assert read_bt.shape == (1, addressed)
    assert read_bt[0].tolist() == [8, 9, 8, 9]


def test_swa_read_table_rings_the_full_graph_storage_width() -> None:
    """Production graph width: all 8192 columns carry the sequence ring block.

    Production runs ``graph_block_table_capacity_cols=8192`` with one ring block
    per sequence, so the read table is 8192 wide and every addressed column must
    equal that single ring block (not just the right-aligned tail).
    """
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257], graph_cols=8192)
    assert read_bt.shape == (1, 8192)
    assert bool((read_bt[0] == 8).all())


def test_default_swa_read_table_stays_right_aligned() -> None:
    """The shared (non-V4.1) default keeps the historical right-aligned layout."""
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257], layout=READ_TABLE_LAYOUT_RIGHT_ALIGNED)
    # dst_lens = ceil(257 / 128) = 3, semantic_cols = 1 -> one retained column at
    # the tail, the leading columns stay -1 (the DSV4 / V4-dspark consumers
    # compensate for this shift via _build_dspark_swa_indices).
    assert read_bt.shape == (1, 3)
    assert read_bt[0].tolist() == [-1, -1, 8]


def test_builder_rejects_unknown_read_table_layout() -> None:
    caches_info, group_infos = build_cache_specs([0], 128, 1)
    with pytest.raises(ValueError, match="unknown SWA read-table layout"):
        DsaMetadataBuilder(caches_info, group_infos, read_table_layout="bogus")


def test_build_block_tables_shared_within_group() -> None:
    """Caches in the same group share the same underlying tensor."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 1 (cr=4): caches 0,1,7 are TOKEN4 (group 1) -> same slot tensor.
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][1].data_ptr()
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][7].data_ptr()
    # Caches 2-6 are SWA (group 0) -> same slot tensor.
    assert dsa.slot_mappings[1][2].data_ptr() == dsa.slot_mappings[1][3].data_ptr()


# -- group_limit: a sibling model that owns fewer groups than the request ----


def _make_swa_only_builder() -> tuple[DsaMetadataBuilder, list, list]:
    """A pure-SWA model (the DSpark draft): exactly one deduplicated group."""
    caches_info, group_infos = build_cache_specs_v41([0] * 3, [], [], swa_block_size=128, n_layers=3)
    return DsaMetadataBuilder(caches_info, group_infos), caches_info, group_infos


def _v41_tables() -> list[torch.Tensor]:
    """Logical SWA/KV tables, shared by every physical history geometry."""
    return [
        torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
        torch.tensor([[20, 21]], dtype=torch.int32),
    ]


def _build_v41(builder: DsaMetadataBuilder, group_limit: int | None = None) -> DsaMetadata:
    """One decode step over the fixed V4.1 tables, optionally group-limited."""
    return builder.build(
        multi_block_tables=_v41_tables(),
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        group_limit=group_limit,
    )


def test_manager_count_beyond_the_model_groups_is_rejected() -> None:
    """Managers must fit the initialized builder cache groups."""
    builder, _, _ = _make_swa_only_builder()
    with pytest.raises(ValueError, match="logical table count 2 does not match model bindings 1"):
        _build_v41(builder)


def test_group_limit_selects_only_the_leading_managers() -> None:
    """Explicit shared-request limit keeps only the owner SWA storage."""
    builder, _, _ = _make_swa_only_builder()
    dsa = _build_v41(builder, group_limit=1)
    # Every draft layer owns one SWA cache -> the same processed manager tensor.
    for layer_bt in dsa.block_tables:
        assert len(layer_bt) == 1
        assert layer_bt[0].data_ptr() == dsa.block_tables[0][0].data_ptr()
    for layer_slots in dsa.slot_mappings:
        assert len(layer_slots) == 1
    # Slot math comes from the SWA table (block 10, ring offset 7), never C2/C1.
    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 7]
    assert dsa.block_tables[0][0][0, 0].item() == 10


def test_group_limit_rejects_a_negative_value() -> None:
    builder, _, _ = _make_swa_only_builder()
    with pytest.raises(ValueError, match="group_limit must be non-negative"):
        _build_v41(builder, group_limit=-1)


def test_build_c4_pad_positions() -> None:
    """c4_pad_positions records next_pos-4 when (pos+1) % 4 == 0."""
    builder, _, _ = _make_builder()
    # q_len=4, q_start=3 -> positions 3,4,5,6. (pos+1)%4==0 at pos=3 (next=4).
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[11],
        q_seq_lens=[4],
        positions=torch.tensor([7, 8, 9, 10], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )
    # Eager ABI reserves two rows: committed position4 plus safe zero padding.
    assert dsa.c4_pad_positions.tolist() == [4, 0]


def test_graph_compressed_positions_use_zero_padding() -> None:
    """ACL graph position buffers match C++ vector::resize zero fill."""
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[11],
        q_seq_lens=[4],
        positions=torch.tensor([7, 8, 9, 10], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
        enable_graph=True,
    )

    assert dsa.c4_pad_positions.tolist() == [4, 0, 0, 0]
    assert dsa.c128_pad_positions.tolist() == [0, 0, 0, 0]


def test_empty_batch_preserves_cpp_zero_length_buffers() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[],
        q_seq_lens=[],
        positions=torch.empty(0, dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert dsa.actual_seq_lengths_query.tolist() == [0]
    assert dsa.kv_cu_seq_lens.tolist() == [0]
    assert dsa.max_seqlen_q.shape == (1,) and dsa.max_seqlen_q.tolist() == [0]
    assert dsa.max_seqlen_kv.shape == (1,) and dsa.max_seqlen_kv.tolist() == [0]
    assert dsa.max_query_len == 0
    assert dsa.max_seq_len == 0
    assert dsa.c4_pad_positions.numel() == 0
    assert dsa.c128_pad_positions.numel() == 0


def test_build_c128_slot_at_compression_boundary() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[128],
        q_seq_lens=[1],
        positions=torch.tensor([255], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    # Layer 2 uses TOKEN(128) for cache 0. The first compressed row is offset 0.
    assert dsa.slot_mappings[2][0].tolist() == [20 * 16]
    assert dsa.c128_pad_positions.tolist() == [0]


def test_multi_batch_slots_are_concatenated_by_sequence() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21], [22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[4, 8],
        q_seq_lens=[1, 1],
        positions=torch.tensor([3, 7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 3, 12 * 128 + 7]
    assert dsa.slot_mappings[1][0].tolist() == [20 * 512, 22 * 512 + 1]


def test_packed_physical_groups_are_rejected() -> None:
    builder, _, _ = _make_builder()
    with pytest.raises(ValueError, match="logical table count 1 does not match model bindings 2"):
        builder.build(
            multi_block_tables=[torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)],
            kv_seq_lens=[128],
            q_seq_lens=[1],
            positions=torch.tensor([127], dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=False,
        )


def test_graph_slots_and_block_tables_use_bucket_capacity() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7, 0, 0, 0], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[1][0].tolist() == [20 * 512 + 1, -1, -1, -1]
    assert dsa.block_tables[1][0].shape == (1, 4)
    assert dsa.block_tables[1][0].tolist() == [[20, 21, -1, -1]]
    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 7, -1, -1, -1]
    assert dsa.block_tables[0][0].shape == (1, 4)


def test_rope_cache_is_split_into_contiguous_cos_and_sin_tables() -> None:
    builder, _, _ = _make_builder()
    cos_sin = torch.arange(24, dtype=torch.float32).view(3, 8)
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[3],
        q_seq_lens=[3],
        positions=torch.arange(3, dtype=torch.int64),
        dsa_cos_sin=cos_sin,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert torch.equal(dsa.cos_table, cos_sin[:, :4])
    assert torch.equal(dsa.sin_table, cos_sin[:, 4:])
    assert dsa.cos_table.is_contiguous()
    assert dsa.sin_table.is_contiguous()


def test_compressed_positions_preserve_position_dtype() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[8],
        q_seq_lens=[4],
        positions=torch.arange(4, 8, dtype=torch.int32),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert dsa.c4_pad_positions.dtype == torch.int32
    assert dsa.c128_pad_positions.dtype == torch.int32
    assert dsa.c4_pad_positions.tolist() == [4, 0]


def test_graph_rejects_block_table_larger_than_bucket_capacity() -> None:
    builder, _, _ = _make_builder()
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)

    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        builder.build(
            multi_block_tables=[block_table, block_table],
            kv_seq_lens=[8],
            q_seq_lens=[1],
            positions=torch.tensor([7], dtype=torch.int64),
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=False,
            enable_graph=True,
            graph_block_table_capacity_cols=2,
        )


@pytest.mark.parametrize("non_blocking", [False, True])
def test_async_upload_retains_staging_until_copy_finishes(monkeypatch: pytest.MonkeyPatch, non_blocking: bool) -> None:
    """Identically shaped fields must not overwrite an in-flight host copy."""
    from types import SimpleNamespace

    submission_order: list[str] = []

    class Device:
        type = "npu"

    class Completion:
        def __init__(self) -> None:
            self.finished = False

        def query(self) -> bool:
            return self.finished

        def record(self, stream: object) -> None:
            assert stream is current_stream
            submission_order.append("record")
            self.finished = False

    class Destination:
        device = Device()

        def __init__(self) -> None:
            self.copies: list[tuple[torch.Tensor, bool]] = []

        def copy_(self, source: torch.Tensor, *, non_blocking: bool) -> None:
            submission_order.append("copy")
            self.copies.append((source, non_blocking))

    current_stream = object()
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Event=Completion, current_stream=lambda device: current_stream),
        raising=False,
    )
    original_empty = torch.empty
    allocations: list[torch.Tensor] = []

    def allocate(*args: object, **kwargs: object) -> torch.Tensor:
        assert kwargs.pop("pin_memory") is True
        tensor = original_empty(*args, **kwargs)
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "empty", allocate)
    builder = DsaMetadataBuilder([], [], async_graph_refresh=True)
    destination = Destination()
    first = torch.arange(8, dtype=torch.int32).reshape(2, 4)[:, ::2]
    second = first + 100
    builder._copy_to_destination(destination, first, non_blocking=non_blocking)
    builder._copy_to_destination(destination, second, non_blocking=non_blocking)
    assert [flag for _, flag in destination.copies] == [non_blocking, non_blocking]
    torch.testing.assert_close(destination.copies[0][0], first)
    torch.testing.assert_close(destination.copies[1][0], second)
    if not non_blocking:
        assert submission_order == ["copy", "copy"]
        assert not allocations
        assert not builder._pinned_uploads
        return
    assert submission_order == ["copy", "record", "copy", "record"]
    assert len(allocations) == 2
    assert destination.copies[0][0].data_ptr() != destination.copies[1][0].data_ptr()
    uploads = next(iter(builder._pinned_uploads.values()))
    uploads[0][1].finished = True
    builder._copy_to_destination(destination, second, non_blocking=True)
    assert len(allocations) == 2
    assert submission_order == ["copy", "record", "copy", "record", "copy", "record"]
    assert destination.copies[2][0] is destination.copies[0][0]
    torch.testing.assert_close(destination.copies[2][0], second)
