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

"""CPU tests for SFA DCP graph-prepare metadata."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadataBuilder,
    _lse_as_token_head,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    copy_into_execution_buffer,
    forward_context,
)


def _cpu_context(execution_state: AclGraphExecutionState | None) -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=execution_state,
    )


def _builder() -> AscendSFADCPMetadataBuilder:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    return AscendSFADCPMetadataBuilder(
        layout=layout,
        device=torch.device("cpu"),
        max_num_reqs=4,
    )


def test_builder_graph_decode_skips_prefill_count() -> None:
    builder = _builder()
    slots = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([8, 4], dtype=torch.int32)

    metadata = builder.build(
        slots,
        block_table,
        seq_lens,
        num_reqs=2,
        num_input_tokens=2,
        num_prefills=0,
    )

    assert metadata.num_prefills == 0
    assert metadata.dcp_context.kv_gather_block_ids is None
    assert metadata.dcp_context.kv_gather_block_table is None
    assert torch.equal(metadata.dcp_context.seq_lens, torch.tensor([4, 4], dtype=torch.int32))


def test_copy_into_execution_buffer_reuses_graph_storage() -> None:
    state = AclGraphExecutionState({})
    with forward_context(_cpu_context(state)):
        first = torch.tensor([1, 2, 3], dtype=torch.int32)
        buffer = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (3,)), first)
        pointer = buffer.data_ptr()
        second = torch.tensor([4, 5, 6], dtype=torch.int32)
        reused = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (3,)), second)

        assert reused.data_ptr() == pointer
        assert torch.equal(reused, second)


def test_copy_into_execution_buffer_eager_returns_source() -> None:
    source = torch.tensor([1, 2], dtype=torch.int32)
    with forward_context(_cpu_context(None)):
        out = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (2,)), source)
        assert out.data_ptr() == source.data_ptr()


def test_lse_as_token_head_squeezes_graph_leading_one() -> None:
    num_tokens = 8
    num_heads = 16
    lse = torch.randn(1, num_tokens, num_heads, dtype=torch.float32)
    out = _lse_as_token_head(lse, num_tokens, num_heads)
    assert tuple(out.shape) == (num_tokens, num_heads)
    torch.testing.assert_close(out, lse[0])
    assert out.data_ptr() == lse.data_ptr()


class _IdentityDcpGroup:
    """Two-rank descriptor; communication is deferred by the CPU fixture."""

    world_size = 2
    rank_in_group = 0
    device_group = None


@pytest.fixture(autouse=True)
def _deferred_dcp_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    def gather(
        output: torch.Tensor,
        value: torch.Tensor,
        *,
        group: object,
        async_op: bool,
    ) -> MagicMock:
        assert group is None and async_op
        assert output.shape == (value.shape[0] * 2, *value.shape[1:])
        handle = MagicMock()
        handle.wait.side_effect = lambda: output.copy_(torch.cat((value, value + 100), dim=0))
        return handle

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)


def _dcp_impl() -> AscendSFADCPImpl:
    return AscendSFADCPImpl(
        _IdentityDcpGroup(),
        scale=0.1,
        index_topk=8,
        layout=KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0),
    )


def test_query_gather_nope_returns_none_q_pe() -> None:
    impl = _dcp_impl()
    ql_nope = torch.randn(2, 4, 8)
    gather_context = impl._start_dcp_query_gather(ql_nope, None)
    assert gather_context.split_sizes == (8,)
    gathered_ql, gathered_q_pe = impl._finish_dcp_gather(gather_context)
    torch.testing.assert_close(gathered_ql, torch.cat((ql_nope, ql_nope + 100), dim=1))
    assert gathered_q_pe is None


def test_query_gather_with_rope_keeps_fused_split() -> None:
    impl = _dcp_impl()
    ql_nope = torch.randn(2, 4, 8)
    q_pe = torch.randn(2, 4, 4)
    gather_context = impl._start_dcp_query_gather(ql_nope, q_pe)
    assert gather_context.split_sizes == (8, 4)
    gathered_ql, gathered_q_pe = impl._finish_dcp_gather(gather_context)
    torch.testing.assert_close(gathered_ql, torch.cat((ql_nope, ql_nope + 100), dim=1))
    assert gathered_q_pe is not None
    torch.testing.assert_close(gathered_q_pe, torch.cat((q_pe, q_pe + 100), dim=1))


def test_decode_nope_query_gather_unpacks_none_before_sfa() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[1, 2], [3, 0]], dtype=torch.int32),
        torch.tensor([8, 8], dtype=torch.int32),
        num_reqs=2,
        num_input_tokens=2,
        num_prefills=0,
    )
    ql_nope = torch.randn(2, 4, 8)
    impl._record_query_gather_context(ql_nope, None, metadata)
    sfa_out = torch.randn(2, 8, 8)
    softmax_max = torch.zeros(2, 8)
    softmax_sum = torch.ones(2, 8)
    merged = torch.randn(2, 4, 8)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32)
    kv = torch.randn(4, 4, 1, 8)
    with (
        patch.object(impl, "_remap_sparse_indices", side_effect=lambda indices: indices),
        patch.object(
            impl,
            "_npu_sparse_flash_attention",
            return_value=(sfa_out, softmax_max, softmax_sum),
        ) as sparse_attn,
        patch.object(impl, "_merge_dcp_outputs", return_value=merged) as merge,
    ):
        out = impl._execute_sparse_flash_attention_process(
            ql_nope,
            None,
            (kv,),
            topk,
            metadata,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
        )
    assert out is merged
    assert sparse_attn.call_args.args[1] is None
    torch.testing.assert_close(sparse_attn.call_args.args[0], torch.cat((ql_nope, ql_nope + 100), dim=1))
    merge.assert_called_once()


def test_prefill_kv_gather_with_rope_still_fuses() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_input_tokens=1,
        num_prefills=1,
    )
    nope = torch.randn(4, 4, 1, 16)
    rope = torch.randn(4, 4, 1, 4)
    impl._record_dcp_kv_gather_context((nope, rope), metadata)
    gather_context = metadata.dcp_context.gather_context
    assert gather_context is not None
    assert gather_context.split_sizes == (16, 4)
    gathered_kv, gathered_rope = impl._finish_dcp_gather(gather_context)
    assert gathered_rope is not None
    torch.testing.assert_close(gathered_kv, torch.cat((nope[1:2], nope[1:2] + 100), dim=0))
    torch.testing.assert_close(gathered_rope, torch.cat((rope[1:2], rope[1:2] + 100), dim=0))


def test_prefill_kv_gather_without_rope_cache() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[1, 2], [3, 0]], dtype=torch.int32),
        torch.tensor([8, 4], dtype=torch.int32),
        num_reqs=2,
        num_input_tokens=2,
        num_prefills=2,
    )
    nope = torch.randn(8, 4, 1, 16)
    impl._record_dcp_kv_gather_context((nope,), metadata)
    gather_context = metadata.dcp_context.gather_context
    assert gather_context is not None
    assert gather_context.split_sizes == (16,)
    gathered_kv, gathered_rope = impl._finish_dcp_gather(gather_context)
    assert gathered_rope is None
    assert gathered_kv.shape[-1] == 16

    sfa_out = torch.randn(2, 4, 16)
    ql_nope = torch.randn(2, 4, 16)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32)
    metadata.dcp_context.gather_context = gather_context
    with patch.object(impl, "_npu_sparse_flash_attention", return_value=sfa_out) as sparse_attn:
        out = impl._execute_sparse_flash_attention_process(
            ql_nope,
            None,
            (nope, None),
            topk,
            metadata,
            torch.tensor([1, 2], dtype=torch.int32),
            torch.tensor([8, 4], dtype=torch.int32),
        )
    assert out is sfa_out
    kv_arg = sparse_attn.call_args.args[2]
    assert kv_arg[1] is None


def test_prefill_kv_gather_skips_zero_width_rope_cache() -> None:
    impl = _dcp_impl()
    builder = _builder()
    metadata = builder.build(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        num_reqs=1,
        num_input_tokens=1,
        num_prefills=1,
    )
    nope = torch.randn(4, 4, 1, 16)
    rope = torch.empty(4, 4, 1, 0)
    impl._record_dcp_kv_gather_context((nope, rope), metadata)
    gather_context = metadata.dcp_context.gather_context
    assert gather_context is not None
    assert gather_context.split_sizes == (16,)
    _, gathered_rope = impl._finish_dcp_gather(gather_context)
    assert gathered_rope is None


def test_npu_sparse_flash_attention_passes_none_rope() -> None:
    impl = _dcp_impl()
    query = torch.randn(2, 4, 8)
    key = torch.randn(4, 4, 1, 8)
    topk = torch.zeros(2, 1, 8, dtype=torch.int32)
    block_table = torch.zeros(1, 2, dtype=torch.int32)
    actual_q = torch.tensor([2], dtype=torch.int32)
    actual_kv = torch.tensor([8], dtype=torch.int32)
    attn_out = torch.randn(2, 4, 8)
    softmax_max = torch.zeros(1)
    softmax_sum = torch.ones(1)
    fake_ops = MagicMock()
    fake_ops.sparse_flash_attention_lse.return_value = (attn_out, softmax_max, softmax_sum)
    with patch.object(torch.ops, "xllm_ops", fake_ops, create=True):
        out = impl._npu_sparse_flash_attention(
            query,
            None,
            (key, None),
            topk,
            actual_q,
            actual_kv,
            block_table,
            sparse_mode=0,
            return_lse=False,
        )
    assert out is attn_out
    kwargs = fake_ops.sparse_flash_attention_lse.call_args.kwargs
    assert kwargs["query_rope"] is None
    assert kwargs["key_rope"] is None


def test_sfa_dcp_graph_prepare_never_allocates_num_heads_workspace() -> None:
    """MLA graph prepare must not request the dense-GQA native workspace."""
    torch_npu = pytest.importorskip("torch_npu", reason="SFA backend graph preparation requires torch_npu")
    from xllm.python.attention.sfa_dcp_backend import SfaDcpAttentionBackend

    dcp_group = MagicMock()
    dcp_group.size.return_value = 2
    dcp_group.rank.return_value = 0
    backend = SfaDcpAttentionBackend(
        # The GLOBAL head count, exactly what glm5_next's Attention base
        # reports on a tp=2 instance.
        num_heads=64,
        num_kv_heads=1,
        head_dim=512,
        scale=0.044,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=dcp_group,
        index_topk=2048,
        max_num_reqs=4,
    )
    assert backend.is_mla
    page_size = 128
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(16, page_size, 1, 512, dtype=torch.bfloat16),
                value=torch.empty(16, page_size, 1, 0, dtype=torch.bfloat16),
                index=torch.empty(32, page_size, 1, 128, dtype=torch.bfloat16),
            )
        ]
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([10], dtype=torch.int32),
        block_table=torch.tensor([[1]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([200], dtype=torch.int32),
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
        has_kv_shard=False,
        kv_split_size=1,
    )

    with (
        forward_context(_cpu_context(AclGraphExecutionState({}))),
        patch.object(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace") as workspace,
    ):
        backend.prepare(metadata, graph_mode=True)
    workspace.assert_not_called()
    assert backend._graph_workspace is None


def test_merge_dcp_outputs_derives_local_heads_from_output_shape() -> None:
    """The merged output's head count comes from the SFA output shape
    (``h_local = output_heads // dcp_size``, sfa_dcp.py), not from the
    backend's constructor num_heads.

    The gathered query's head count is whatever the model's per-rank
    projection produced times the DCP group size -- for the M11.4 shape
    D(cp1, tp2, ks2) that is the GLOBAL count -- so deriving per-rank heads
    from the tensor keeps the merge correct for every TP x kv_split
    combination regardless of the convention the Attention base used."""
    impl = AscendSFADCPImpl(
        _IdentityDcpGroup(),
        scale=0.1,
        index_topk=8,
        layout=KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0),
    )
    num_tokens = 3
    gathered_heads = 16
    head_dim = 8
    output = torch.randn(num_tokens, gathered_heads, head_dim)
    softmax_lse = torch.randn(num_tokens, gathered_heads)
    merged = torch.randn(num_tokens, gathered_heads // 2, head_dim)

    packed_ops = ModuleType("xllm.python.kernels_npu.triton.dcp_packed_a2a")
    pack = MagicMock()
    combine = MagicMock(return_value=merged)
    packed_ops.packed_send_shape = MagicMock(return_value=(2, 8, 3, 9))
    packed_ops.pack_dcp_output_lse = pack
    packed_ops.fused_dcp_lse_combine = combine
    with (
        patch.dict(sys.modules, {packed_ops.__name__: packed_ops}),
        forward_context(_cpu_context(None)),
        patch.object(torch.distributed, "all_to_all_single") as all_to_all,
    ):
        result = impl._merge_dcp_outputs(output, softmax_lse)

    assert result is merged
    packed_ops.packed_send_shape.assert_called_once_with(
        num_tokens, gathered_heads, head_dim, 2, scatter_dim=1, dtype=output.dtype
    )
    pack.assert_called_once()
    all_to_all.assert_called_once()
    # The combine consumes a merged buffer sized from the OUTPUT shape:
    # 16 gathered heads over a dcp group of 2 -> 8 per-rank heads.
    combine.assert_called_once()
    merged_arg = combine.call_args.kwargs["output"]
    assert tuple(merged_arg.shape) == (num_tokens, gathered_heads // 2, head_dim)
    # The pack stage saw the gathered (output-shaped) head count verbatim.
    assert pack.call_args.args[0] is output
