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

"""Tests for the NPU paged-attention backend."""

import sys
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="NPU paged-attention tests require torch_npu")

from xllm.python.attention.backend import (  # noqa: E402
    LayerCache,
    build_speculative_ssm_state_indices,
)
from xllm.python.attention.dsa_kv_offload_state import DsaKvOffloadState  # noqa: E402
from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
    _pad_dense_mla_queries,
)
from xllm.python.kernels_npu import sparse_attention  # noqa: E402
from xllm.python.model_executor.forward_context import (  # noqa: E402
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)


def test_speculative_ssm_indices_keep_all_bootstrap_checkpoints() -> None:
    state_indices = torch.tensor([3, 7], dtype=torch.int64)

    indices = build_speculative_ssm_state_indices(state_indices, checkpoint_stride=4)

    assert indices.tolist() == [[12, 13, 14, 15], [28, 29, 30, 31]]


pytestmark = pytest.mark.usefixtures("causal_conv1d_reference")


@pytest.mark.parametrize("heads,padded_heads", [(6, 8), (24, 32), (32, 32), (96, 128)])
def test_dense_mla_padding_preserves_real_heads(heads: int, padded_heads: int) -> None:
    query = torch.randn(2, heads, 512)
    position = torch.randn(2, heads, 64)
    padded_query, padded_position = _pad_dense_mla_queries(query, position)
    assert padded_query.shape == (2, padded_heads, 512)
    torch.testing.assert_close(padded_query[:, :heads], query)
    torch.testing.assert_close(padded_position[:, :heads], position)
    assert torch.count_nonzero(padded_query[:, heads:]) == 0
    if heads == padded_heads:
        assert padded_query is query and padded_position is position


def test_dense_mla_padding_leaves_other_head_dimensions_unchanged() -> None:
    query, position = torch.randn(2, 24, 128), torch.randn(2, 24, 64)
    padded_query, padded_position = _pad_dense_mla_queries(query, position)
    assert padded_query is query and padded_position is position


def test_dense_mla_padding_reuses_one_eager_bucket_and_refreshes_inputs() -> None:
    buffers: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = {}
    query, position = torch.randn(2, 24, 512), torch.randn(2, 24, 64)
    first = _pad_dense_mla_queries(query, position, eager_buffers=buffers)
    query.add_(1)
    position.sub_(1)
    second = _pad_dense_mla_queries(query, position, eager_buffers=buffers)
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(first, second))
    torch.testing.assert_close(second[0][:, :24], query)
    torch.testing.assert_close(second[1][:, :24], position)
    assert torch.count_nonzero(second[0][:, 24:]) == 0
    assert torch.count_nonzero(second[1][:, 24:]) == 0
    _pad_dense_mla_queries(query[:1], position[:1], eager_buffers=buffers)
    assert len(buffers) == 1


def test_dense_mla_padding_graph_storage_is_owned_by_entry_and_layer() -> None:
    query, position = torch.randn(2, 24, 512), torch.randn(2, 24, 64)
    states = [AclGraphExecutionState(persistent_buffers={}) for _ in range(2)]
    storage: list[tuple[torch.Tensor, torch.Tensor]] = []
    for state in states:
        context = ForwardContext(None, torch.device("cpu"), None, [], execution_state=state)
        with forward_context(context):
            first = _pad_dense_mla_queries(query, position, layer_id=0)
            again = _pad_dense_mla_queries(query + 1, position + 1, layer_id=0)
            other_layer = _pad_dense_mla_queries(query, position, layer_id=1)
        assert first[0].data_ptr() == again[0].data_ptr()
        assert first[0].data_ptr() != other_layer[0].data_ptr()
        torch.testing.assert_close(first[0][:, :24], query + 1)
        storage.append(first)
    assert storage[0][0].data_ptr() != storage[1][0].data_ptr()


@pytest.mark.parametrize("pad_query_heads", [False, True])
def test_dense_mla_backend_preserves_queries_without_padding_opt_in(
    monkeypatch: pytest.MonkeyPatch, pad_query_heads: bool
) -> None:
    backend = object.__new__(NpuPagedAttentionBackend)
    backend._use_fia_v2 = True
    backend._mla_actual_seq_q_host = [2]
    backend._mla_actual_seq_kv_host = [2]
    backend._mla_eager_query_buffers = {}
    backend._metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    backend._causal_mask = None
    backend.scale = 0.125
    query, position = torch.randn(2, 6, 512), torch.randn(2, 6, 64)
    seen: list[torch.Tensor] = []
    kernel_outputs: list[torch.Tensor] = []

    def attention(q: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(q)
        assert kwargs["num_query_heads"] == q.shape[1]
        result = q.clone()
        kernel_outputs.append(result)
        return result, torch.empty(0)

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score_v2", attention, raising=False)
    # Omitting the argument exercises the default used by existing callers.
    options = {"pad_query_heads": True} if pad_query_heads else {}
    with forward_context(ForwardContext(backend, torch.device("cpu"), None, [])):
        output = backend._mla_dense_fia_v2(
            query,
            position,
            torch.empty(1, 128, 1, 512),
            torch.empty(1, 128, 1, 64),
            torch.zeros(1, 1, dtype=torch.int32),
            0,
            **options,
        )
    assert seen[0].shape[1] == (8 if pad_query_heads else 6)
    if not pad_query_heads:
        assert seen[0] is query
        assert output is kernel_outputs[0]
        assert not backend._mla_eager_query_buffers
    torch.testing.assert_close(output, query, rtol=0, atol=0)


def test_uses_first_nonempty_key_cache() -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        scale=0.125,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    linear_cache = LayerCache(
        key=None,
        value=None,
        conv=torch.empty(8, 3, 64),
        ssm=torch.empty(8, 2, 4, 4),
    )
    key_cache = torch.empty(17, 128, 2, 64)
    value_cache = torch.empty_like(key_cache)

    backend.bind_kv_caches(
        [
            linear_cache,
            LayerCache(key=key_cache, value=value_cache),
        ]
    )

    assert backend.num_kv_blocks == 17
    assert backend.page_size == 128


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("activation", ["identity", "silu"])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_kda_dense_conv_dispatches_fused_activation(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    activation: str,
    is_prefill: bool,
) -> None:
    backend = NpuPagedAttentionBackend.__new__(NpuPagedAttentionBackend)
    value = torch.arange(2 * 12 * width, dtype=torch.bfloat16).view(2, width, 12)
    weight = torch.ones(12, 1, 4, dtype=torch.float32)
    layer = SimpleNamespace(
        layer_id=0,
        activation=activation,
        conv_weight_t=weight.squeeze(1).t().to(value.dtype).contiguous(),
    )
    state = torch.zeros(2, 3, 12, dtype=torch.bfloat16)
    expected_conv = torch.linspace(-4, 4, value.numel()).to(torch.bfloat16).view(2, width, 12)

    def native_conv(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        dense_state: torch.Tensor,
        query_start_loc: list[int],
        activation_mode: int,
        run_mode: int,
    ) -> torch.Tensor:
        assert inputs.is_contiguous()
        assert weights is layer.conv_weight_t
        torch.testing.assert_close(weights, weight.squeeze(1).t().to(value.dtype))
        assert dense_state is state
        assert query_start_loc == []
        assert activation_mode == (1 if activation == "silu" else 0)
        assert run_mode == (0 if is_prefill else 1)
        torch.testing.assert_close(inputs, value)
        dense_state.fill_(7)
        return expected_conv.clone()

    monkeypatch.setattr(torch.ops.xllm_ops, "causal_conv1d", native_conv)
    output = backend._causal_conv1d(value, state, layer, is_prefill=is_prefill)
    torch.testing.assert_close(output, expected_conv, rtol=0, atol=0)
    assert output.dtype == torch.bfloat16
    assert torch.all(state == 7)


def test_graph_index_history_uses_full_table_and_masks_unused_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    # Cross the former 32K limit while keeping the index width small.
    capacity = 257 * 128
    index = torch.arange(capacity, dtype=torch.float32).reshape(257, 128, 1, 1)
    cache = torch.empty(257, 128, 1, 128)
    layer_cache = LayerCache(key=cache, value=cache, index=index)
    backend.bind_kv_caches([layer_cache])
    metadata = SimpleNamespace(
        block_table=torch.arange(257, dtype=torch.int32).repeat(2, 1),
        kv_seq_lens=torch.tensor([32769, 129], dtype=torch.int32),
    )
    backend._metadata = metadata
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[layer_cache],
        execution_state=AclGraphExecutionState(persistent_buffers={}),
    )

    def forbid_host_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("graph history must not read device values on host")

    with forward_context(context), monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbid_host_read)
        patch.setattr(torch.Tensor, "tolist", forbid_host_read)
        patch.setattr(torch.Tensor, "item", forbid_host_read)
        history = backend.gather_index_history(SimpleNamespace(layer_id=0), batch_size=2)
    assert history.shape == (2, capacity, 1)
    torch.testing.assert_close(history[0, :32769, 0], torch.arange(32769, dtype=torch.float32))
    torch.testing.assert_close(history[1, :129, 0], torch.arange(129, dtype=torch.float32))
    assert torch.count_nonzero(history[0, 32769:]) == 0
    assert torch.count_nonzero(history[1, 129:]) == 0


def test_xfia_prepare_preserves_live_device_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=2048,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        use_xfia_decode=True,
    )
    cache = torch.empty(20, 128, 1, 128, dtype=torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    lengths = torch.tensor([128, 129, 2303, 2304, 2305, 1, 1, 1], dtype=torch.int32)
    table = torch.zeros(8, 20, dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=False, is_chunked_prefill=False, block_table=table, kv_seq_lens=lengths, slot_mapping=torch.arange(8)
    )

    def forbid_host_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("XFIA decode metadata must not be read on host")

    buffers = {}

    def execution_buffer(key: tuple, factory: Callable[[], torch.Tensor]) -> torch.Tensor:
        if key not in buffers:
            buffers[key] = factory()
        return buffers[key]

    with monkeypatch.context() as patch:
        patch.setattr(
            sys.modules["xllm.python.attention.npu_paged_attention"], "get_execution_buffer", execution_buffer
        )
        patch.setattr(torch.Tensor, "cpu", forbid_host_read)
        patch.setattr(torch.Tensor, "tolist", forbid_host_read)
        patch.setattr(torch.Tensor, "item", forbid_host_read)
        backend.prepare(metadata, graph_mode=True)
        row_ends = backend._xfia_query_ends
        graph_lengths = backend._xfia_kv_lengths
        graph_starts = backend._xfia_kv_starts
        lengths.add_(1)
        backend.prepare(metadata, graph_mode=True)
    assert backend._xfia_query_ends is row_ends
    assert backend._xfia_kv_lengths is graph_lengths
    assert backend._xfia_kv_starts is graph_starts
    torch.testing.assert_close(graph_lengths, lengths)
    torch.testing.assert_close(backend._block_table_i32, table)
    assert row_ends.tolist() == list(range(1, 9))
    assert backend._graph_workspace is None
    assert backend._actual_seq_kv == []


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("proposal_width", [1, 8])
@pytest.mark.parametrize("sliding_window,expected_starts", [(0, [0, 0, 0]), (2048, [0, 7, 257])])
def test_xfia_single_token_decode_uses_actual_window(
    graph_mode: bool,
    proposal_width: int,
    sliding_window: int,
    expected_starts: list[int],
) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=sliding_window,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        use_xfia_decode=True,
        xfia_query_width=proposal_width,
    )
    cache = torch.empty(60, 128, 1, 128, dtype=torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    table = torch.arange(60, dtype=torch.int32).reshape(3, 20)
    lengths = torch.tensor([7, 2055, 2305], dtype=torch.int32)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        block_table=table,
        kv_seq_lens=lengths,
        slot_mapping=torch.arange(3),
    )
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[LayerCache(key=cache, value=cache)],
        execution_state=AclGraphExecutionState(persistent_buffers={}) if graph_mode else None,
    )
    with forward_context(context):
        backend.prepare(metadata, graph_mode=graph_mode)
    assert backend._xfia_query_ends.tolist() == [1, 2, 3]
    assert backend._xfia_kv_starts.tolist() == expected_starts
    torch.testing.assert_close(backend._xfia_kv_lengths, lengths)
    torch.testing.assert_close(backend._block_table_i32, table)


@pytest.mark.parametrize("sliding_window", [0, 2048])
@pytest.mark.parametrize("proposal_width", [8, 15])
@pytest.mark.parametrize("expanded_graph", [False, True])
def test_xfia_dflash2_chunked_query_keeps_band_window(
    monkeypatch: pytest.MonkeyPatch, sliding_window: int, proposal_width: int, expanded_graph: bool
) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=4,
        num_kv_heads=1,
        head_dim=128,
        scale=128**-0.5,
        sliding_window=sliding_window,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        use_xfia_decode=True,
        xfia_query_width=proposal_width,
        draft_query_width=proposal_width,
    )
    cache = torch.empty(40, 128, 1, 128, dtype=torch.bfloat16)
    backend.bind_kv_caches([LayerCache(key=cache, value=cache)])
    table = torch.arange(40, dtype=torch.int32).reshape(2, 20)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=expanded_graph,
        is_mixed=False,
        kv_split_size=1,
        use_dflash_proposal_attention=True,
        block_table=table,
        kv_seq_lens=torch.tensor([2055, 2305], dtype=torch.int32),
        slot_mapping=torch.arange(2 * proposal_width),
    )
    if expanded_graph:
        metadata.is_chunked_prefill = False
        metadata.block_table = table.repeat_interleave(proposal_width, dim=0)
        metadata.kv_seq_lens = metadata.kv_seq_lens.repeat_interleave(proposal_width)
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[LayerCache(key=cache, value=cache)],
        execution_state=AclGraphExecutionState(persistent_buffers={}) if expanded_graph else None,
    )
    monkeypatch.setattr(
        sys.modules[NpuPagedAttentionBackend.__module__].kernels,
        "reshape_paged_cache",
        lambda *args: None,
        raising=False,
    )
    monkeypatch.setattr(backend, "_xfia_decode", lambda query, *args: query.flatten(1))
    query = torch.randn(2 * proposal_width, 4, 128)
    key = torch.randn(2 * proposal_width, 1, 128)
    layer = SimpleNamespace(
        layer_id=0, causal=False, attention_window=(sliding_window - 1, proposal_width - 1) if sliding_window else None
    )
    with forward_context(context):
        backend.prepare(metadata, graph_mode=expanded_graph)
        torch.testing.assert_close(backend.execute(query, key, key, layer), query.flatten(1))
        if sliding_window:
            metadata.is_spec_verify = True
            metadata.use_dflash_proposal_attention = False
            with pytest.raises(ValueError, match="chunked proposal batch"):
                backend.execute(query, key, key, layer)
    assert backend._xfia_query_ends.tolist() == list(range(1, 2 * proposal_width + 1))
    assert backend._xfia_kv_lengths.tolist() == [2055] * proposal_width + [2305] * proposal_width
    expected_starts = [
        max(0, length - proposal_width + offset - sliding_window + 1) if sliding_window else 0
        for length in [2055, 2305]
        for offset in range(proposal_width)
    ]
    assert backend._xfia_kv_starts.tolist() == expected_starts
    torch.testing.assert_close(backend._block_table_i32, table.repeat_interleave(proposal_width, dim=0))


def test_target_spec_verify_chunk_does_not_use_xfia(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        scale=0.5,
        sliding_window=2048,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_xfia_decode=True,
        xfia_query_width=8,
    )
    cache = torch.empty(4, 128, 1, 4)
    backend.bind_kv_caches([LayerCache(key=cache, value=torch.empty_like(cache))])
    table = torch.arange(4, dtype=torch.int32).reshape(2, 2)
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=True,
        use_dflash_proposal_attention=False,
        block_table=table,
        kv_seq_lens=torch.full((2,), 136, dtype=torch.int32),
        kv_seq_lens_host_values=[136, 136],
        q_cu_seq_lens=torch.tensor([0, 8, 16], dtype=torch.int32),
        slot_mapping=torch.arange(16),
    )

    def forbid_xfia(*args: object, **kwargs: object) -> None:
        raise AssertionError("target spec-verify must not use XFIA decode")

    monkeypatch.setattr(backend, "_prepare_xfia_decode", forbid_xfia)
    backend.prepare(metadata)
    torch.testing.assert_close(backend._block_table_i32, table)


def test_jev_prefix_bidir_uses_full_window_and_keeps_decode_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = NpuPagedAttentionBackend(
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        scale=0.5,
        sliding_window=0,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    backend.enable_prefix_bidir()
    cache = torch.empty(2, 128, 1, 4)
    backend.bind_kv_caches([LayerCache(key=cache, value=torch.empty_like(cache))])
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        kv_seq_lens=None,
        kv_seq_lens_host_values=[2, 3],
        q_cu_seq_lens_host_values=[2, 5],
        q_cu_seq_lens=torch.tensor([2, 5], dtype=torch.int32),
        slot_mapping=torch.arange(5),
    )
    backend.prepare(metadata)
    module = sys.modules["xllm.python.attention.npu_paged_attention"]
    cache_write = []
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(cp_context=None))
    monkeypatch.setattr(module.kernels, "reshape_paged_cache", lambda *args: cache_write.append(args), raising=False)
    calls = []

    def fused_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object):
        calls.append((key, kwargs))
        return query.clone(), None

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", fused_attention, raising=False)
    query = torch.randn(5, 2, 4)
    key = torch.randn(5, 1, 4)
    bidir_layer = SimpleNamespace(layer_id=0, causal=False, attention_window=None)
    causal_layer = SimpleNamespace(layer_id=0, causal=True, attention_window=None)
    torch.testing.assert_close(backend.execute(query, key, key, bidir_layer), query.flatten(1))
    assert len(cache_write) == 1
    assert calls[0][0].shape == key.shape
    assert calls[0][1]["actual_seq_lengths"] == [2, 5]
    assert calls[0][1]["actual_seq_lengths_kv"] == [2, 5]
    assert calls[0][1]["atten_mask"] is None
    assert calls[0][1]["sparse_mode"] == 0
    assert calls[0][1]["pre_tokens"] == calls[0][1]["next_tokens"] == 2147483647
    assert "block_table" not in calls[0][1]

    backend.execute(query, key, key, causal_layer)
    assert calls[1][1]["atten_mask"] is backend._causal_mask
    assert calls[1][1]["sparse_mode"] == 3
    assert "block_table" in calls[1][1]

    metadata.is_prefill = False
    metadata.block_table = None
    backend.prepare(metadata)
    monkeypatch.setattr(backend, "_decode", lambda *args: query.flatten(1))
    torch.testing.assert_close(backend.execute(query, key, key, bidir_layer), query.flatten(1))


@pytest.mark.parametrize("bad_q,bad_kv", [([0, 2, 4], [2, 3]), ([1, 2, 5], [2, 3])])
def test_jev_prefix_bidir_rejects_partial_prefill_before_cache_write(bad_q: list[int], bad_kv: list[int]) -> None:
    backend = NpuPagedAttentionBackend(2, 1, 4, 0.5, 0, False, torch.device("cpu"), torch.float32)
    backend.enable_prefix_bidir()
    metadata = SimpleNamespace(
        is_prefill=True,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        q_cu_seq_lens_host_values=bad_q,
        kv_seq_lens_host_values=bad_kv,
    )
    with pytest.raises(ValueError, match="q_len == kv_len"):
        backend.prepare(metadata)
    assert backend._metadata is None
    metadata.q_cu_seq_lens_host_values = [0, 2, 5]
    metadata.is_chunked_prefill = True
    with pytest.raises(ValueError, match="pure"):
        backend.prepare(metadata)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("is_spec_verify", [False, None])
def test_xfia_matching_prompt_chunk_keeps_causal_prefill(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
    is_spec_verify: bool | None,
) -> None:
    width = 8
    backend = NpuPagedAttentionBackend(
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        scale=0.5,
        sliding_window=2048,
        is_mla=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_xfia_decode=True,
        xfia_query_width=width,
    )
    cache = torch.empty(batch_size * 2, 128, 1, 4)
    backend.bind_kv_caches([LayerCache(key=cache, value=torch.empty_like(cache))])
    table = torch.arange(batch_size * 2, dtype=torch.int32).reshape(batch_size, 2)
    query_ends = torch.arange(batch_size + 1, dtype=torch.int32) * width
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        block_table=table,
        kv_seq_lens=torch.full((batch_size,), 128 + width, dtype=torch.int32),
        kv_seq_lens_host_values=[128 + width] * batch_size,
        q_cu_seq_lens=query_ends,
        slot_mapping=torch.arange(batch_size * width),
    )
    if is_spec_verify is not None:
        metadata.is_spec_verify = is_spec_verify

    def forbid_xfia(*args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary prompt chunks must not use XFIA decode")

    monkeypatch.setattr(backend, "_prepare_xfia_decode", forbid_xfia)
    monkeypatch.setattr(backend, "_xfia_decode", forbid_xfia)
    backend.prepare(metadata)
    torch.testing.assert_close(backend._block_table_i32, table)

    module = sys.modules["xllm.python.attention.npu_paged_attention"]
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(cp_context=None))
    monkeypatch.setattr(module.kernels, "reshape_paged_cache", lambda *args: None, raising=False)

    def causal_prefill(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        # Preserve grouped prompt queries and the right-aligned causal mask,
        # including the existing 128-token KV prefix.
        assert kwargs["actual_seq_lengths"] == query_ends[1:].tolist()
        assert kwargs["actual_seq_lengths_kv"] == [128 + width] * batch_size
        assert kwargs["sparse_mode"] == 3
        assert kwargs["atten_mask"] is backend._causal_mask
        torch.testing.assert_close(kwargs["block_table"], table)
        return query.clone(), None

    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", causal_prefill, raising=False)
    query = torch.randn(batch_size * width, 2, 4)
    key = torch.randn(batch_size * width, 1, 4)
    layer = SimpleNamespace(layer_id=0, causal=True, attention_window=None)
    output = backend.execute(query, key, key, layer)
    torch.testing.assert_close(output, query.flatten(1))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("graph_mode,capacity", [(True, 0), (False, 0), (True, 4)])
def test_captured_draft_windows_move_with_each_query(
    monkeypatch: pytest.MonkeyPatch, causal: bool, graph_mode: bool, capacity: int
) -> None:
    def _scatter(
        slots: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> None:
        key_cache.flatten(0, 1).index_copy_(0, slots, key)
        value_cache.flatten(0, 1).index_copy_(0, slots, value)

    def _fia(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, None]:
        if kwargs["input_layout"] == "TND":
            table = kwargs["block_table"]
            lengths = kwargs.get("actual_seq_kvlen", kwargs.get("actual_seq_lengths_kv"))
            outputs = []
            for row, length in enumerate(lengths):
                positions = torch.arange(length)
                slots = table[row, positions // 4].long() * 4 + positions % 4
                q = query[row * 3 : (row + 1) * 3].transpose(0, 1)
                k = key.flatten(0, 1)[slots].view(-1, 1, 1).permute(1, 2, 0)
                v = value.flatten(0, 1)[slots].view(-1, 1, 1).transpose(0, 1)
                scores = q @ k
                if kwargs.get("sparse_mode") == 4:
                    query_positions = length - 3 + torch.arange(3)
                    visible = (positions[None, :] >= query_positions[:, None] - kwargs["pre_tokens"]) & (
                        positions[None, :] <= query_positions[:, None] + kwargs["next_tokens"]
                    )
                    if kwargs.get("atten_mask") is not None:
                        visible &= positions[None, :] <= query_positions[:, None]
                    scores.masked_fill_(~visible, -torch.inf)
                outputs.append((scores.softmax(-1) @ v).transpose(0, 1))
            return torch.cat(outputs), None
        assert kwargs["input_layout"] == "BSND"
        scores = query.transpose(1, 2) @ key.permute(0, 2, 3, 1)
        scores.masked_fill_(kwargs["atten_mask"], -torch.inf)
        return (scores.softmax(-1) @ value.transpose(1, 2)).transpose(1, 2).contiguous(), None

    monkeypatch.setattr(sys.modules["xllm.python.kernels"], "reshape_paged_cache", _scatter, raising=False)
    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score", _fia, raising=False)
    monkeypatch.setattr(torch.ops.npu, "npu_fused_infer_attention_score_v2", _fia, raising=False)
    table = torch.tensor([[2, 0, 1], [6, 7, 4]], dtype=torch.int32)
    cache = LayerCache(torch.full((8, 4, 1, 1), float("nan")), torch.full((8, 4, 1, 1), float("nan")))
    for row, anchor in enumerate((3, 8)):
        positions = torch.arange(anchor)
        physical = table[row, positions // 4].long() * 4 + positions % 4
        cache.key.flatten(0, 1).index_fill_(0, physical, 0)
        cache.value.flatten(0, 1)[physical] = (positions + 100 * row).float().view(-1, 1, 1)
    metadata = SimpleNamespace(
        block_table=table.repeat_interleave(3, dim=0),
        kv_seq_lens=torch.tensor([6] * 3 + [11] * 3, dtype=torch.int32),
        kv_seq_lens_host_values=[6] * 3 + [11] * 3,
        slot_mapping=torch.tensor([11, 0, 1, 16, 17, 18]),
        q_cu_seq_lens=None,
        q_cu_seq_lens_host_values=list(range(1, 7)),
        is_prefill=False,
        is_chunked_prefill=False,
        is_mixed=False,
        is_spec_verify=False,
        use_dflash_proposal_attention=True,
        kv_split_size=1,
        dsa_graph_token_capacity=8,
    )
    backend = NpuPagedAttentionBackend(1, 1, 1, 1.0, 4, False, torch.device("cpu"), torch.float32, draft_query_width=3)
    metadata.dsa_graph_token_capacity = capacity or backend.graph_token_capacity_ceiling
    if not graph_mode:
        metadata.block_table = table
        metadata.kv_seq_lens = torch.tensor([6, 11], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [6, 11]
        metadata.q_cu_seq_lens = torch.tensor([0, 3, 6], dtype=torch.int32)
        metadata.q_cu_seq_lens_host_values = [0, 3, 6]
        metadata.is_chunked_prefill = True
    # CPU reference operators need no NPU workspace.
    backend._draft_graph_workspaces[6] = torch.empty(0)
    backend.bind_kv_caches([cache])
    layer = SimpleNamespace(layer_id=0, causal=causal, attention_window=(3, 0 if causal else 2))
    context = ForwardContext(backend, torch.device("cpu"), metadata, [cache])
    query = key = torch.zeros(6, 1, 1)
    value = torch.tensor([3, 4, 5, 108, 109, 110], dtype=torch.float32).view(6, 1, 1)
    with forward_context(context):
        backend.prepare(metadata, graph_mode=graph_mode)
        output = backend.execute(query, key, value, layer)
        # A full layer shares this proposal's paging and execution context.
        full = backend.execute(query, key, value, SimpleNamespace(layer_id=0, causal=False, attention_window=None))
    expected = [1.5, 2.5, 3.5, 106.5, 107.5, 108.5] if causal else [2.5, 3, 3.5, 107.5, 108, 108.5]
    if capacity:
        expected = [2.5, 3, 3.5, 107.5, 108, 108.5] if causal else [3.5] * 3 + [108.5] * 3
    torch.testing.assert_close(output.flatten(), torch.tensor(expected))
    torch.testing.assert_close(full.flatten(), torch.tensor([2.5] * 3 + [105.0] * 3))
    with forward_context(context):
        if graph_mode:
            metadata.dsa_graph_token_capacity = 0
            backend.prepare(metadata, graph_mode=True)
            with pytest.raises(ValueError, match="positive token capacity"):
                backend.execute(query, key, value, layer)
        else:
            metadata.is_chunked_prefill = False
            backend.prepare(metadata)
            with pytest.raises(ValueError, match="chunked proposal batch"):
                backend.execute(query, key, value, layer)


def test_device_only_mode_keeps_graph_capacity_consistent() -> None:
    backend = NpuPagedAttentionBackend(
        2, 1, 128, 0.125, 8, False, torch.device("cpu"), torch.float32, draft_query_width=4
    )
    assert backend.graph_token_capacity_ceiling == 11
    backend.set_device_only_decode_inputs(True, 4)
    assert backend.graph_token_capacity_ceiling == 0
    assert not backend.requires_host_kv_lengths
    backend.set_device_only_decode_inputs(False, 4)
    assert backend.graph_token_capacity_ceiling == 11
    assert backend.requires_host_kv_lengths


def _offload_metadata(widths: Sequence[int], lengths: Sequence[int], slots: Sequence[int]) -> SimpleNamespace:
    cumulative = [0]
    for width in widths:
        cumulative.append(cumulative[-1] + width)
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=any(width > 1 for width in widths),
        is_mixed=False,
        is_spec_verify=any(width > 1 for width in widths),
        is_dummy=False,
        has_kv_shard=False,
        block_table=torch.arange(len(widths), dtype=torch.int32)[:, None],
        kv_seq_lens=torch.tensor(lengths, dtype=torch.int32),
        kv_seq_lens_host_values=list(lengths),
        q_seq_lens_host=torch.tensor(widths, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor(cumulative, dtype=torch.int32),
        slot_mapping=torch.tensor(slots, dtype=torch.int32),
    )


def _install_offload_cpu_oracle(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Run real MR808 Python views against a numerical CPU vendor substitute.

    This checks backend composition and alias/layout use, not NPU execution.
    Attention computes an independent softmax over the emitted paged mapping.
    """
    module = sys.modules["xllm.python.attention.npu_paged_attention"]
    calls: list[tuple[str, int, int]] = []
    for name in (
        "dsa_kv_offloading_update_nope_out",
        "dsa_kv_offloading_update_rope_with_plan_out",
        "dsa_kv_offloading_serve_rope_planned_out",
        "has_stable_resident_slots",
        "has_stable_rope_resident_slots",
    ):
        monkeypatch.setattr(module.kernels, name, getattr(sparse_attention, name), raising=False)
    monkeypatch.setattr(torch.ops.xllm_ops, "dsa_kv_offloading_selection_capacity", lambda: 2176, raising=False)

    def plan(
        descriptor: torch.Tensor,
        topk: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        table: torch.Tensor,
        resident: torch.Tensor,
        resident_slots: torch.Tensor,
        count: torch.Tensor,
        tokens: torch.Tensor,
        encoded: torch.Tensor,
        compact: torch.Tensor | None = None,
    ) -> int:
        tokens.fill_(-1)
        encoded.fill_(-1)
        indices.fill_(-1)
        if compact is not None:
            compact.fill_(-1)
        misses = 0
        for row, (_, window, visible, state_row) in enumerate(descriptor.tolist()):
            table[row].copy_(torch.arange(state_row * 17, (state_row + 1) * 17))
            positions = ((topk[row, 0] >= 0) & (topk[row, 0] < visible)).nonzero().flatten().tolist()
            lengths[row] = len(positions) if window else 0
            if not window:
                continue
            old = resident[state_row, 0, : len(positions)].tolist()
            hits = {
                pos: old.index(int(topk[row, 0, pos]))
                for pos in positions
                if int(topk[row, 0, pos]) < visible - window and int(topk[row, 0, pos]) in old
            }
            free = iter(slot for slot in range(len(positions)) if slot not in hits.values())
            resident[state_row].fill_(-1)
            resident_slots[state_row].fill_(-1)
            for ordinal, pos in enumerate(positions):
                token = int(topk[row, 0, pos])
                slot = hits[pos] if pos in hits else next(free)
                tokens[row, 0, pos] = token
                encoded[row, 0, pos] = slot if pos in hits else -slot - 2
                indices[row, 0, pos] = slot
                resident[state_row, 0, slot] = token
                resident_slots[state_row, 0, ordinal] = slot
                if compact is not None:
                    compact[row, 0, ordinal] = slot
                misses += int(pos not in hits)
            count[state_row] = len(positions)
        return misses

    def serve(
        host: torch.Tensor,
        host_rope: torch.Tensor | None,
        table: torch.Tensor,
        descriptor: torch.Tensor,
        row_map: torch.Tensor,
        tokens: torch.Tensor,
        encoded: torch.Tensor,
        append: torch.Tensor,
        current: torch.Tensor,
        current_rope: torch.Tensor | None,
        selected: torch.Tensor,
        selected_rope: torch.Tensor | None,
    ) -> None:
        assert row_map.shape == (tokens.shape[0], tokens.shape[0])
        saved = encoded.clone()
        for row, (request, window, visible, state_row) in enumerate(descriptor.tolist()):
            positions = (encoded[row, 0] < -1).nonzero().flatten().tolist()
            for pos in positions:
                token, slot = int(tokens[row, 0, pos]), -int(encoded[row, 0, pos]) - 2
                offset = token - (visible - window)
                payloads = [(host, current, selected)]
                if host_rope is not None:
                    payloads.append((host_rope, current_rope, selected_rope))
                for source, now, destination in payloads:
                    if offset >= 0:
                        assert offset < window
                        value = now[row_map[row, offset], 0]
                    else:
                        value = source[table[request, token // 128], token % 128, 0]
                    destination.view(-1, 2176, source.shape[-1])[state_row, slot].copy_(value)
            if append[row] >= 0:
                host.view(-1, 512)[append[row]].copy_(current[row, 0])
                if host_rope is not None:
                    host_rope.view(-1, 64)[append[row]].copy_(current_rope[row, 0])
        torch.testing.assert_close(encoded, saved, rtol=0, atol=0)

    def update_nope(*args: torch.Tensor) -> None:
        host, table, descriptor, row_map, topk, append, current, selected = args[:8]
        indices, lengths, selected_table, resident, slots, count, compact = args[8:]
        tokens, encoded = torch.empty_like(topk), torch.empty_like(topk)
        misses = plan(
            descriptor, topk, indices, lengths, selected_table, resident, slots, count, tokens, encoded, compact
        )
        serve(host, None, table, descriptor, row_map, tokens, encoded, append, current, None, selected, None)
        calls.append(("nope", misses, int((append >= 0).sum())))

    def update_rope(*args: torch.Tensor) -> None:
        host, table, descriptor, row_map, topk, append, current, selected = args[:8]
        indices, lengths, selected_table, resident, slots, count, rope, current_rope, selected_rope, tokens, encoded = (
            args[8:]
        )
        misses = plan(descriptor, topk, indices, lengths, selected_table, resident, slots, count, tokens, encoded)
        serve(
            host,
            rope,
            table,
            descriptor,
            row_map,
            tokens,
            encoded,
            append,
            current,
            current_rope,
            selected,
            selected_rope,
        )
        calls.append(("rope", misses, int((append >= 0).sum())))

    def shared_rope(*args: torch.Tensor) -> None:
        serve(*args)
        calls.append(("serve", int((args[6] < -1).sum()), int((args[7] >= 0).sum())))

    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
        table: torch.Tensor,
        query_ends: torch.Tensor,
        lengths: torch.Tensor,
        query_rope: torch.Tensor | None,
        key_rope: torch.Tensor | None,
        scale: float,
        *args: object,
    ) -> torch.Tensor:
        output = args[-1]
        assert isinstance(output, torch.Tensor)
        assert query_ends.tolist() == list(range(1, query.shape[0] + 1))
        for row in range(query.shape[0]):
            logical = indices[row, 0]
            logical = logical[(logical >= 0) & (logical < lengths[row])].long()
            if not logical.numel():
                output[row].zero_()
                continue
            pages, offsets = table[row, logical // 128].long(), logical % 128
            scores = query[row].float() @ key[pages, offsets, 0].float().t()
            if query_rope is not None:
                scores += query_rope[row].float() @ key_rope[pages, offsets, 0].float().t()
            output[row].copy_((scores * scale).softmax(-1) @ value[pages, offsets, 0].float())
        return output

    monkeypatch.setattr(torch.ops.xllm_ops, "dsa_kv_offloading_update_nope_out", update_nope, raising=False)
    monkeypatch.setattr(torch.ops.xllm_ops, "dsa_kv_offloading_update_rope_with_plan_out", update_rope, raising=False)
    monkeypatch.setattr(torch.ops.xllm_ops, "dsa_kv_offloading_serve_rope_planned_out", shared_rope, raising=False)
    monkeypatch.setattr(module.kernels, "sparse_flash_attention_out", attention, raising=False)
    return calls


def _offload_backend(rope: bool = False, capacity: int = 3) -> NpuPagedAttentionBackend:
    backend = NpuPagedAttentionBackend(
        2,
        1,
        512,
        512**-0.5,
        0,
        True,
        torch.device("cpu"),
        torch.bfloat16,
        offload_topk_width=3,
        max_offload_query_rows=capacity,
    )
    if rope:
        backend.configure_kv_offload_layers(
            [SimpleNamespace(layer_id=index, qk_rope_head_dim=64, is_shared=bool(index)) for index in range(2)]
        )
    caches = []
    for layer in range(2 if rope else 1):
        key = (torch.arange(256, dtype=torch.float32) + 10 * layer).view(2, 128, 1, 1).expand(-1, -1, -1, 512)
        rotary = (key[..., :64] + 20).contiguous().to(torch.bfloat16) if rope else None
        caches.append(
            LayerCache(
                key=key.contiguous().to(torch.bfloat16),
                value=rotary,
                index=torch.empty(2, 128, 1, 128) if not layer else None,
            )
        )
    backend.bind_kv_caches(caches)
    return backend


@pytest.mark.parametrize("rope", [False, True])
def test_kv_offload_request_reorder_departure_and_reused_instance_clear_only_affected_rows(rope: bool) -> None:
    state = DsaKvOffloadState(torch.device("cpu"), 4)
    state.enabled = True
    state.rope_dim = 64 if rope else 0
    state.plan_owners = (0, 0, 2)
    state.reset()
    owners = [state.ensure_layer(layer, 3) for layer in (0, 2)]
    shared = state.ensure_layer(1, 3) if rope else None
    payload_bytes = (
        state.capacity * sparse_attention.SELECTION_CAPACITY * (sparse_attention.RESIDENT_HEAD_DIM + state.rope_dim) * 2
    )
    for storage in [*owners, *([shared] if shared is not None else [])]:
        rope_bytes = storage.selection_rope.nbytes if storage.selection_rope is not None else 0
        assert storage.selection.nbytes + rope_bytes == payload_bytes
    addresses = [layer.selection.data_ptr() for layer in owners]
    state.prepare_requests(("a", "b"), (2, 1), (100, 200))
    for layer in owners:
        layer.resident[:3].fill_(11)
        layer.slots[:3].fill_(12)
        layer.valid[:3].fill_(1)
    state.prepare_requests(("b", "a"), (1, 2), (200, 100))
    descriptor = torch.zeros(3, 4, dtype=torch.int32)
    state.write_resident_rows(descriptor, 3)
    assert descriptor[:, 3].tolist() == [2, 0, 1]
    assert all(bool((layer.resident[:3] == 11).all()) for layer in owners)
    state.prepare_requests(("b",), (1,), (200,))
    for layer in owners:
        assert bool((layer.resident[:2] == -1).all()) and bool((layer.slots[:2] == -1).all())
        assert layer.valid[:3].tolist() == [0, 0, 1]
    old = state.resident_rows.owners
    state.prepare_requests((), (), ())
    assert state.resident_rows.owners == old and state.active_rows == ()
    state.prepare_requests(("b",), (1,), (201,))
    assert state.active_rows == (0,)
    assert all(bool((layer.resident[:3] == -1).all()) for layer in owners)
    old = state.resident_rows.owners
    with pytest.raises(ValueError, match="capacity"):
        state.prepare_requests(("b", "c"), (1, 4), (201, 300))
    assert state.resident_rows.owners == old
    assert [layer.selection.data_ptr() for layer in owners] == addresses
    if shared is not None:
        assert shared.resident is None and shared.plan_tokens is None
        assert state.ensure_layer(1, 1) is shared


@pytest.mark.parametrize("rope", [False, True])
def test_kv_offload_attention_composes_causal_current_kv_and_reuses_owner_shared_hits(
    monkeypatch: pytest.MonkeyPatch, rope: bool
) -> None:
    calls = _install_offload_cpu_oracle(monkeypatch)
    backend = _offload_backend(rope)
    bound_layers = dict(backend._offload_state.layers)
    assert set(bound_layers) == set(range(len(backend._kv_caches)))

    def expected_payload(values: Sequence[Sequence[float]], layer_id: int) -> torch.Tensor:
        outputs = []
        for tokens in values:
            latent = torch.tensor(tokens, dtype=torch.float32) + 10 * layer_id
            scores = (latent + 20) * backend.scale if rope else torch.zeros_like(latent)
            outputs.append((scores.softmax(0) * latent).sum())
        return torch.stack(outputs).to(torch.bfloat16)[:, None, None]

    metadata = _offload_metadata((2, 1), (4, 6), (2, 3, 133))
    backend.prepare_kv_offload_requests(("a", "b"), (2, 1), (100, 200))
    backend.prepare(metadata)
    descriptor, row_map, _ = backend._offload_state.step_descriptor
    assert descriptor.tolist() == [[0, 1, 3, 0], [0, 2, 4, 1], [1, 1, 6, 2]]
    assert row_map.tolist() == [[0, 1, 0], [0, 1, 0], [2, 2, 2]]
    query = torch.zeros(3, 2, 512, dtype=torch.bfloat16)
    query_rope = torch.full((3, 2, 64), 1 / 64, dtype=torch.bfloat16) if rope else None
    topk = torch.tensor([[[0, 2, 3]], [[1, 2, 3]], [[0, 5, 7]]], dtype=torch.int32)
    if rope:
        topk = topk.to(torch.int64)
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)):
        for layer_id, cache in enumerate(backend._kv_caches):
            cache.key.view(-1, 512)[metadata.slot_mapping.long()] = -999
            current = (torch.tensor([200, 201, 300]) + 10 * layer_id).to(torch.bfloat16)
            current = current[:, None, None].expand(-1, 1, 512).contiguous()
            rotary = (current[:, :, :64] + 20).contiguous() if rope else None
            layer = SimpleNamespace(layer_id=layer_id, qk_rope_head_dim=64 if rope else 0)
            result = backend.execute_mla(query, query_rope, current, rotary, layer, topk)
            expected = expected_payload(([0, 200], [1, 200, 201], [128, 300]), layer_id)
            torch.testing.assert_close(result, expected.expand_as(result))
            torch.testing.assert_close(cache.key.view(-1, 512)[metadata.slot_mapping.long()], current[:, 0])
            if rope:
                torch.testing.assert_close(cache.value.view(-1, 64)[metadata.slot_mapping.long()], rotary[:, 0])
    assert calls == ([("rope", 7, 3), ("serve", 7, 3)] if rope else [("nope", 7, 3)])
    assert backend._offload_state.layers == bound_layers
    # The Eager expanded layout supplies a causal KV length and table per query.
    metadata = _offload_metadata((2, 1), (4, 5, 7), (3, 4, 134))
    metadata.block_table = torch.tensor([[0], [0], [1]], dtype=torch.int32)
    backend.prepare_kv_offload_requests(("a", "b"), (2, 1), (100, 200))
    backend.prepare(metadata)
    assert backend._offload_state.step_descriptor[0][:, :3].tolist() == [[0, 1, 4], [1, 2, 5], [2, 1, 7]]
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)):
        for layer_id in range(len(backend._kv_caches)):
            layer = SimpleNamespace(layer_id=layer_id, qk_rope_head_dim=64 if rope else 0)
            current = (torch.tensor([200, 201, 300]) + 10 * layer_id).to(torch.bfloat16)
            current = current[:, None, None].expand(-1, 1, 512).contiguous()
            rotary = (current[:, :, :64] + 20).contiguous() if rope else None
            cache = backend._kv_caches[layer_id]
            cache.key.view(-1, 512)[metadata.slot_mapping.long()] = current[:, 0]
            if rope:
                cache.value.view(-1, 64)[metadata.slot_mapping.long()] = rotary[:, 0]
            host_key = cache.key.clone()
            host_rope = cache.value.clone() if rope else None
            result = backend.execute_mla(query, query_rope, current, rotary, layer, topk, cache_is_preprocessed=True)
            expected = expected_payload(([0, 200, 200], [1, 200, 200], [128, 300]), layer_id)
            torch.testing.assert_close(result, expected.expand_as(result))
            torch.testing.assert_close(cache.key, host_key, rtol=0, atol=0)
            if rope:
                torch.testing.assert_close(cache.value, host_rope, rtol=0, atol=0)
    assert all(appends == 0 for _, _, appends in calls[-(2 if rope else 1) :])
    metadata = _offload_metadata((1, 1), (5, 7), (4, 134))
    backend.prepare_kv_offload_requests(("a", "b"), (1, 1), (100, 200))
    backend.prepare(metadata)
    assert backend._offload_state.step_descriptor[0][:, 3].tolist() == [0, 2]
    query, query_rope = query[:2], query_rope[:2] if rope else None
    topk = torch.tensor([[[0, 4, -1]], [[0, 6, -1]]], dtype=torch.int32)
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)):
        for layer_id in range(len(backend._kv_caches)):
            layer = SimpleNamespace(layer_id=layer_id, qk_rope_head_dim=64 if rope else 0)
            current = torch.full((2, 1, 512), 400 + 10 * layer_id, dtype=torch.bfloat16)
            rotary = torch.full((2, 1, 64), 420 + 10 * layer_id, dtype=torch.bfloat16) if rope else None
            result = backend.execute_mla(query, query_rope, current, rotary, layer, topk)
            expected = expected_payload(([0, 400], [128, 400]), layer_id)
            torch.testing.assert_close(result, expected.expand_as(result))
    assert calls[-(2 if rope else 1) :] == ([("rope", 2, 2), ("serve", 2, 2)] if rope else [("nope", 2, 2)])
    backend.prepare_kv_offload_requests(("a", "b"), (1, 1), (101, 201))
    backend.prepare(metadata)
    host_topk = torch.tensor([[[3, -1, -1]], [[5, -1, -1]]], dtype=torch.int32)
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)):
        for layer_id in range(len(backend._kv_caches)):
            layer = SimpleNamespace(layer_id=layer_id, qk_rope_head_dim=64 if rope else 0)
            current = torch.full((2, 1, 512), 400 + 10 * layer_id, dtype=torch.bfloat16)
            rotary = torch.full((2, 1, 64), 420 + 10 * layer_id, dtype=torch.bfloat16) if rope else None
            result = backend.execute_mla(
                query, query_rope, current, rotary, layer, host_topk, cache_is_preprocessed=True
            )
            expected = expected_payload(([200], [300]), layer_id)
            torch.testing.assert_close(result, expected.expand_as(result))
    assert calls[-(2 if rope else 1) :] == ([("rope", 2, 0), ("serve", 2, 0)] if rope else [("nope", 2, 0)])
    # A synthetic empty step exposes no KV and leaves real resident ownership intact.
    residents = {
        layer_id: layer.resident.clone()
        for layer_id, layer in backend._offload_state.layers.items()
        if layer.resident is not None
    }
    host_values = [(cache.key.clone(), cache.value.clone() if rope else None) for cache in backend._kv_caches]
    owners = backend._offload_state.resident_rows.owners
    dummy = _offload_metadata((1,), (0,), (-1,))
    dummy.is_dummy = True
    backend.prepare_kv_offload_requests((), (), ())
    backend.prepare(dummy)
    with forward_context(ForwardContext(backend, torch.device("cpu"), dummy, backend._kv_caches)):
        for layer_id in range(len(backend._kv_caches)):
            result = backend.execute_mla(
                query[:1],
                query_rope[:1] if rope else None,
                current[:1],
                rotary[:1] if rope else None,
                SimpleNamespace(layer_id=layer_id, qk_rope_head_dim=64 if rope else 0),
                topk[:1],
            )
            assert torch.count_nonzero(result) == 0
    assert backend._offload_state.resident_rows.owners == owners
    for layer_id, saved in residents.items():
        torch.testing.assert_close(backend._offload_state.layers[layer_id].resident, saved, rtol=0, atol=0)
    for cache, (key, value) in zip(backend._kv_caches, host_values):
        torch.testing.assert_close(cache.key, key, rtol=0, atol=0)
        if rope:
            torch.testing.assert_close(cache.value, value, rtol=0, atol=0)
    if rope:
        backend.prepare_kv_offload_requests(("a", "b"), (1, 1), (100, 200))
        backend.prepare(metadata)
        with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)):
            shared = SimpleNamespace(layer_id=1, qk_rope_head_dim=64)
            with pytest.raises(ValueError, match="configured order"):
                backend.execute_mla(query, query_rope, current, rotary, shared, topk)
            backend.execute_mla(
                query, query_rope, current, rotary, SimpleNamespace(layer_id=0, qk_rope_head_dim=64), topk
            )
            with pytest.raises(ValueError, match="same-step"):
                backend.execute_mla(query, query_rope, current, rotary, shared, topk + 1)
            assert calls[-1][0] == "rope"
            backend.execute_mla(query, query_rope, current, rotary, shared, topk.to(torch.int64))
    rebound = [
        LayerCache(key=cache.key.clone(), value=cache.value.clone() if rope else None, index=cache.index)
        for cache in backend._kv_caches
    ]
    backend.bind_kv_caches(rebound)
    assert set(backend._offload_state.layers) == set(bound_layers)
    for layer_id, storage in backend._offload_state.layers.items():
        assert storage is not bound_layers[layer_id]
        if storage.resident is not None:
            assert bool((storage.resident == -1).all()) and bool((storage.slots == -1).all())
            assert torch.count_nonzero(storage.valid) == 0
    assert all(owner is None for owner in backend._offload_state.resident_rows.owners)


@pytest.mark.parametrize("configured,cache_device", [(False, "cpu"), (False, "meta"), (True, "meta")])
def test_kv_offload_opt_in_keeps_existing_host_and_device_attention_paths(
    monkeypatch: pytest.MonkeyPatch, configured: bool, cache_device: str
) -> None:
    backend = NpuPagedAttentionBackend(
        2,
        1,
        512,
        512**-0.5,
        0,
        True,
        torch.device("cpu"),
        torch.bfloat16,
        offload_topk_width=3 if configured else 0,
        max_offload_query_rows=3 if configured else 0,
    )
    if not configured and cache_device == "cpu":
        with pytest.raises(ValueError, match="offload-capable"):
            backend.configure_kv_offload_layers([SimpleNamespace(layer_id=0, qk_rope_head_dim=0)])
    key = torch.empty(1, 128, 1, 512, dtype=torch.bfloat16, device=cache_device)
    backend.bind_kv_caches([LayerCache(key=key, value=None, index=torch.empty(1, 128, 1, 128))])
    assert not backend.uses_host_main_kv
    assert backend._offload_state is None or (
        not backend._offload_state.layers and backend._offload_state.resident_rows is None
    )
    backend.prepare_kv_offload_requests((), (), ())
    metadata = _offload_metadata((1,), (5,), (4,))
    backend.prepare(metadata)
    query = torch.ones(1, 2, 512, dtype=torch.bfloat16)
    topk = torch.tensor([[[4, -1, 0]]], dtype=torch.int32)
    seen = []
    module = sys.modules["xllm.python.attention.npu_paged_attention"]
    monkeypatch.setattr(torch.ops.xllm_ops, "reshape_paged_cache", lambda *args: seen.append(args[3]), raising=False)
    monkeypatch.setattr(
        module.kernels, "sparse_flash_attention_out", lambda *args: seen.append(args[3]) or query, raising=False
    )
    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)):
        result = backend.execute_mla(
            query, None, query[:, :1].contiguous(), None, SimpleNamespace(layer_id=0, qk_rope_head_dim=0), topk
        )
    assert result is query and seen[0] is key and seen[1] is topk


def test_kv_offload_rejects_unsafe_metadata_and_graph_before_kv_or_selection_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_offload_cpu_oracle(monkeypatch)
    backend = _offload_backend()
    metadata = _offload_metadata((1,), (5,), (4,))
    original = backend._kv_caches[0].key.clone()
    original_state = backend._offload_state
    original_layers = dict(original_state.layers)
    with pytest.raises(ValueError, match="request identities"):
        backend.prepare(metadata)
    backend.prepare_kv_offload_requests(("a",), (1,), (100,))
    with pytest.raises(ValueError, match="Eager"):
        backend.prepare(metadata, graph_mode=True)
    metadata.kv_seq_lens = torch.tensor([2**32 + 5], dtype=torch.int64)
    with pytest.raises(ValueError, match="INT32"):
        backend.prepare(metadata)
    metadata.kv_seq_lens = torch.tensor([129], dtype=torch.int32)
    with pytest.raises(ValueError, match="current window"):
        backend.prepare(metadata)
    metadata.kv_seq_lens = torch.tensor([5], dtype=torch.int32)
    metadata.slot_mapping[0] = 128
    with pytest.raises(ValueError, match="causal query positions"):
        backend.prepare(metadata)
    metadata.slot_mapping[0] = 4
    duplicate = _offload_metadata((1, 1), (5, 5), (4, 4))
    duplicate.block_table[1, 0] = 2
    backend.prepare_kv_offload_requests(("a", "b"), (1, 1), (100, 200))
    with pytest.raises(ValueError, match="physical Host page"):
        backend.prepare(duplicate)
    duplicate.block_table[1, 0] = 1
    with pytest.raises(ValueError, match="same physical Host slot"):
        backend.prepare(duplicate)
    duplicate.slot_mapping[1] = 256
    with pytest.raises(ValueError, match="physical Host row"):
        backend.prepare(duplicate)
    assert original_state.step_metadata is None and original_state.requests_prepared
    duplicate.slot_mapping.fill_(-1)
    readbacks: list[tuple[int, ...]] = []
    original_cpu = torch.Tensor.cpu

    def read_validation(tensor: torch.Tensor) -> torch.Tensor:
        readbacks.append(tuple(tensor.shape))
        return original_cpu(tensor)

    def reject_scalar_or_dynamic_readback(*args: object, **kwargs: object) -> None:
        raise AssertionError("offload prepare must consolidate validation readback without dynamic unique")

    with monkeypatch.context() as validation_patch:
        validation_patch.setattr(torch.Tensor, "cpu", read_validation)
        validation_patch.setattr(torch.Tensor, "__bool__", reject_scalar_or_dynamic_readback)
        validation_patch.setattr(torch.Tensor, "item", reject_scalar_or_dynamic_readback)
        validation_patch.setattr(torch.Tensor, "unique", reject_scalar_or_dynamic_readback)
        backend._prepare_kv_offload_step(duplicate, graph_mode=False)
    assert readbacks == [(6,)]
    backend.prepare_kv_offload_requests(("a",), (1,), (100,))
    backend.prepare(metadata)
    with pytest.raises(ValueError, match="request identities"):
        backend.prepare(metadata)
    with (
        forward_context(ForwardContext(backend, torch.device("cpu"), metadata, backend._kv_caches)),
        pytest.raises(ValueError, match="complete current KV"),
    ):
        backend.execute_mla(
            torch.zeros(1, 2, 512, dtype=torch.bfloat16),
            None,
            None,
            None,
            SimpleNamespace(layer_id=0, qk_rope_head_dim=0),
            torch.zeros(1, 1, 3, dtype=torch.int32),
        )
    assert not calls and backend._offload_state.layers == original_layers
    torch.testing.assert_close(backend._kv_caches[0].key, original, rtol=0, atol=0)
    original_binding = backend._kv_caches
    with pytest.raises(ValueError, match="page128/latent512"):
        backend.bind_kv_caches([LayerCache(key=original.float(), value=None)])
    assert backend._kv_caches is original_binding

    def allocation_failure(*args: object, **kwargs: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("selection allocation failure")

    with monkeypatch.context() as allocation_patch:
        allocation_patch.setattr(torch, "empty", allocation_failure)
        with pytest.raises(torch.OutOfMemoryError, match="selection allocation failure"):
            backend.bind_kv_caches([LayerCache(key=original.clone(), value=None)])
    assert backend._kv_caches is original_binding and backend._offload_state is original_state
    assert backend._offload_state.layers == original_layers
