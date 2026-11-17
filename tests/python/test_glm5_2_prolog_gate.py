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

"""CPU wiring tests for the fused MLA prolog path.

These tests do not exercise the NPU operator itself. They pin the decisions that
guard it: which caches may use the fused path, how the paged slots reach the
operator, and how the prolog weight aliases are attached to the attention
module.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.kernels_npu import mla_cache
from xllm.python.models import glm5_2


@pytest.fixture(autouse=True)
def prolog_vendor(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Keep CPU wiring tests independent of the installed vendor runtime."""
    monkeypatch.setattr(glm5_2, "_MLA_PROLOG", True)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", ModuleType("cann_ops_transformer"))
    namespace = SimpleNamespace(mla_prolog=object())
    monkeypatch.setattr(torch.ops, "cann_ops_transformer", namespace)
    return namespace


@pytest.fixture
def npu_format(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Drive ``mla_cache`` through a fake format probe."""

    def fake_get_npu_format(_: torch.Tensor) -> int:
        return state.format

    state = SimpleNamespace(format=-1)
    torch_npu = ModuleType("torch_npu")
    torch_npu.get_npu_format = fake_get_npu_format
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    return state


def test_prolog_cache_check_accepts_bf16_paged_cache(npu_format: SimpleNamespace) -> None:
    kv_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    rope_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    assert mla_cache.mla_prolog_cache_check(kv_cache, rope_cache) is None


@pytest.mark.parametrize("quantized", ["kv", "rope"])
def test_prolog_cache_check_rejects_quantized_cache(npu_format: SimpleNamespace, quantized: str) -> None:
    kv_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    rope_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    if quantized == "kv":
        kv_cache = kv_cache.to(torch.int8)
    else:
        rope_cache = rope_cache.to(torch.float16)
    reason = mla_cache.mla_prolog_cache_check(kv_cache, rope_cache)
    assert reason is not None
    assert "kv=" in reason and "rope=" in reason


def test_prolog_cache_check_rejects_packed_layout(npu_format: SimpleNamespace) -> None:
    npu_format.format = mla_cache.FRACTAL_NZ_FORMAT
    kv_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    rope_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    reason = mla_cache.mla_prolog_cache_check(kv_cache, rope_cache)
    assert reason is not None
    assert "PA_NZ" in reason


@pytest.mark.parametrize("check", ["layout", "prolog"])
def test_cache_layout_requires_format_probe(monkeypatch: pytest.MonkeyPatch, check: str) -> None:
    """Every caller must fail rather than guess a layout when the probe is absent."""
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    kv_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    with pytest.raises(AttributeError, match="get_npu_format"):
        if check == "layout":
            mla_cache.mla_cache_mode(kv_cache)
        else:
            mla_cache.mla_prolog_cache_check(kv_cache, torch.zeros_like(kv_cache))


@pytest.mark.parametrize("check", ["layout", "prolog"])
def test_cache_layout_requires_npu_runtime(monkeypatch: pytest.MonkeyPatch, check: str) -> None:
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    kv_cache = torch.zeros(2, 8, dtype=torch.bfloat16)
    with pytest.raises(ModuleNotFoundError, match="torch_npu"):
        if check == "layout":
            mla_cache.mla_cache_mode(kv_cache)
        else:
            mla_cache.mla_prolog_cache_check(kv_cache, torch.zeros_like(kv_cache))


@pytest.mark.parametrize("packed", [False, True])
def test_cache_layout_uses_reported_format(npu_format: SimpleNamespace, packed: bool) -> None:
    npu_format.format = mla_cache.FRACTAL_NZ_FORMAT if packed else -1
    expected = mla_cache.PA_NZ_CACHE_MODE if packed else mla_cache.PA_CACHE_MODE
    assert mla_cache.mla_cache_mode(torch.zeros(2, 8, dtype=torch.bfloat16)) == expected


def _dynamic_projection(rows: int, cols: int) -> nn.Module:
    projection = nn.Module()
    projection._w8a8 = nn.Module()
    projection._w8a8.weight = nn.Parameter(torch.ones((rows, cols), dtype=torch.int8), requires_grad=False)
    projection._w8a8.weight_scale = nn.Parameter(torch.ones(rows, dtype=torch.float32))
    return projection


def _static_projection(rows: int, cols: int) -> nn.Module:
    projection = nn.Module()
    projection._w8a8 = nn.Module()
    projection._w8a8.weight = nn.Parameter(torch.ones((rows, cols), dtype=torch.int8), requires_grad=False)
    projection._w8a8.deq_scale = nn.Parameter(torch.ones(rows, dtype=torch.float32))
    return projection


def _prolog_attention(indexer: nn.Module | None = None) -> glm5_2.Glm52MLAAttention:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    nn.Module.__init__(attention)
    attention.cfg = SimpleNamespace(layerwise_split_size=1, layerwise_split_rank=0, tp_size=1)
    attention.q_a_proj = _dynamic_projection(4, 8)
    attention.kv_a_proj_with_mqa = _dynamic_projection(5, 8)
    attention.q_b_proj = _dynamic_projection(6, 8)
    attention.indexer = indexer
    attention.W_UK = torch.zeros(1, 1, 1)
    attention.q_a_layernorm = SimpleNamespace(weight=torch.zeros(4), eps=1e-6)
    attention.kv_a_layernorm = SimpleNamespace(weight=torch.zeros(5), eps=1e-6)
    return attention


def _dynamic_indexer() -> nn.Module:
    indexer = nn.Module()
    indexer.wq_b = _dynamic_projection(6, 4)
    return indexer


def test_prepare_prolog_weights_keeps_parameter_names() -> None:
    attention = _prolog_attention(indexer=_dynamic_indexer())
    names_before = set(dict(attention.named_parameters()))
    attention._prepare_prolog_weights()
    assert attention._prolog_ready is True
    # The aliases must not add parameters: a duplicated registration would
    # shadow the canonical ``..._proj._w8a8.weight`` names in the state dict.
    assert set(dict(attention.named_parameters())) == names_before
    assert attention._prolog_weight_dq is attention.q_a_proj._w8a8.weight
    assert attention._prolog_weight_dkv is attention.kv_a_proj_with_mqa._w8a8.weight
    assert attention._prolog_scale_uq.shape == (1, 6)


def test_prepare_prolog_weights_rejects_static_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    attention = _prolog_attention(indexer=_dynamic_indexer())
    attention.q_b_proj = _static_projection(6, 8)
    attention._prepare_prolog_weights()
    assert attention._prolog_ready is False


def test_prepare_prolog_weights_rejects_static_indexer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The indexer consumes an activation scale, so it gates the fused path too."""
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    indexer = nn.Module()
    indexer.wq_b = _static_projection(6, 4)
    attention = _prolog_attention(indexer=indexer)
    attention._prepare_prolog_weights()
    assert attention._prolog_ready is False


@pytest.mark.parametrize("disabled", ["switch", "layerwise"])
def test_prepare_prolog_weights_skips_vendor_when_disabled(monkeypatch: pytest.MonkeyPatch, disabled: str) -> None:
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    attention = _prolog_attention()
    if disabled == "switch":
        monkeypatch.setattr(glm5_2, "_MLA_PROLOG", False)
    else:
        attention.cfg.layerwise_split_size = 2
    attention._prepare_prolog_weights()
    assert attention._prolog_ready is True


def test_prepare_prolog_weights_skips_vendor_without_dynamic_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    attention = _prolog_attention()
    attention.q_a_proj._w8a8 = None
    attention._prepare_prolog_weights()
    assert attention._prolog_ready is False


def test_prepare_prolog_weights_fails_at_load_without_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", None)
    attention = _prolog_attention()
    with pytest.raises(ModuleNotFoundError, match="cann_ops_transformer"):
        attention._prepare_prolog_weights()
    assert attention._prolog_ready is False
    assert not hasattr(attention, "_prolog_weight_dq")


def test_prepare_prolog_weights_fails_at_load_without_registered_op(prolog_vendor: SimpleNamespace) -> None:
    """Import success alone is insufficient when an old or partial package is installed."""
    del prolog_vendor.mla_prolog
    attention = _prolog_attention()
    with pytest.raises(RuntimeError, match="ops-transformer runtime.*mla_prolog"):
        attention._prepare_prolog_weights()
    assert attention._prolog_ready is False
    assert not hasattr(attention, "_prolog_weight_dq")


def _preprocess_context(
    kv_dtype: torch.dtype = torch.bfloat16,
    slots: list[int] | None = None,
    cache_slots: list[int] | None = None,
) -> glm5_2.MlaPreprocessContext:
    return glm5_2.MlaPreprocessContext(
        kv_cache=torch.zeros(2, 8, dtype=kv_dtype),
        rope_cache=torch.zeros(2, 8, dtype=torch.bfloat16),
        slot_mapping=torch.tensor(slots if slots is not None else [0, 0, 0], dtype=torch.long),
        cache_slot_mapping=None if cache_slots is None else torch.tensor(cache_slots, dtype=torch.long),
    )


class _FakeBackend:
    def __init__(self, context: glm5_2.MlaPreprocessContext | None) -> None:
        self.context = context
        self.context_calls = 0

    def mla_preprocess_context(self, _layer: nn.Module) -> glm5_2.MlaPreprocessContext | None:
        self.context_calls += 1
        return self.context


def _prolog_context(
    attention: glm5_2.Glm52MLAAttention,
    context: glm5_2.MlaPreprocessContext,
    rejection: str | None = None,
    enabled: bool = True,
    ready: bool = True,
) -> tuple[glm5_2.MlaPreprocessContext | None, _FakeBackend]:
    attention._prolog_ready = ready
    backend = _FakeBackend(context)
    with (
        patch.object(glm5_2, "_MLA_PROLOG", enabled),
        patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=None)),
        patch.object(glm5_2.kernels, "mla_prolog_cache_check", return_value=rejection, create=True),
    ):
        return attention._prolog_context(backend), backend


def test_prolog_context_accepts_bf16_cache() -> None:
    attention = _prolog_attention()
    context = _preprocess_context()
    assert _prolog_context(attention, context)[0] is context


def test_prolog_context_declines_unsupported_cache() -> None:
    attention = _prolog_attention()
    reason = "unsupported cache dtypes: kv=torch.int8, rope=torch.bfloat16"
    assert _prolog_context(attention, _preprocess_context(), reason)[0] is None


def test_prolog_context_skips_the_backend_when_weights_are_not_ready() -> None:
    """A rejected path must not pay for the per-layer preprocess context."""
    attention = _prolog_attention()
    context, backend = _prolog_context(attention, _preprocess_context(), ready=False)
    assert context is None
    assert backend.context_calls == 0


def test_prolog_context_declines_when_disabled() -> None:
    attention = _prolog_attention()
    context, backend = _prolog_context(attention, _preprocess_context(), enabled=False)
    assert context is None
    assert backend.context_calls == 0


def test_prolog_context_declines_under_context_parallel() -> None:
    attention = _prolog_attention()
    attention._prolog_ready = True
    backend = _FakeBackend(_preprocess_context())
    with (
        patch.object(glm5_2, "_MLA_PROLOG", True),
        patch.object(glm5_2, "get_forward_context", return_value=SimpleNamespace(cp_context=object())),
    ):
        assert attention._prolog_context(backend) is None
    assert backend.context_calls == 0


def test_prolog_context_declines_under_layerwise_split() -> None:
    attention = _prolog_attention()
    attention.cfg = SimpleNamespace(layerwise_split_size=2, layerwise_split_rank=0, tp_size=1)
    context, backend = _prolog_context(attention, _preprocess_context())
    assert context is None
    assert backend.context_calls == 0


def _run_prolog_decode(
    attention: glm5_2.Glm52MLAAttention, context: glm5_2.MlaPreprocessContext
) -> dict[str, torch.Tensor]:
    attention._prepare_prolog_weights()
    recorded: dict[str, torch.Tensor] = {}
    # ``kernels.mla_prolog_v3`` returns (query, query_rope, q_c, q_c_scale).
    outputs = (
        torch.zeros(3, 6, dtype=torch.int8),
        torch.zeros(3, 2, dtype=torch.bfloat16),
        torch.zeros(3, 4, dtype=torch.int8),
        torch.zeros(3, dtype=torch.float32),
    )

    def fake_prolog(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # positional layout of ``kernels.mla_prolog_v3``: cache_index is the
        # 16th argument, after the rope angles and the two cache tensors.
        recorded["cache_index"] = args[15]
        return outputs

    with (
        patch.object(glm5_2.kernels, "mla_prolog_v3", side_effect=fake_prolog, create=True),
        patch.object(
            glm5_2.kernels,
            "dynamic_quant",
            lambda hidden: (torch.zeros(3, 8, dtype=torch.int8), torch.ones(3)),
            create=True,
        ),
        patch.object(
            glm5_2,
            "_gather_interleave_cos_sin",
            lambda cache, positions: (torch.zeros(3, 2), torch.zeros(3, 2)),
        ),
    ):
        attention._prolog_decode(torch.zeros(3, 8), torch.zeros(3, dtype=torch.long), torch.zeros(1), context)
    return recorded


def test_prolog_decode_folds_padding_slots() -> None:
    attention = _prolog_attention()
    context = _preprocess_context(slots=[5, -1, 7])
    cache_index = _run_prolog_decode(attention, context)["cache_index"]
    assert cache_index.dtype == torch.int64
    assert cache_index.tolist() == [5, 0, 7]


def test_prolog_decode_prefers_local_slots() -> None:
    attention = _prolog_attention()
    context = _preprocess_context(slots=[5, -1, 7], cache_slots=[9, -1, 0])
    cache_index = _run_prolog_decode(attention, context)["cache_index"]
    assert cache_index.tolist() == [9, 0, 0]


def test_project_query_reuses_quantized_cq() -> None:
    """The indexer must take prolog's quantized c^Q, not re-quantize it."""
    indexer = glm5_2.Glm52Indexer.__new__(glm5_2.Glm52Indexer)
    nn.Module.__init__(indexer)
    calls: list[tuple[torch.Tensor, ...]] = []

    def fake_forward_quantized(*args: torch.Tensor) -> torch.Tensor:
        calls.append(args)
        return torch.zeros(3, 6, dtype=torch.bfloat16)

    indexer.wq_b = nn.Module()
    indexer.wq_b._w8a8 = SimpleNamespace(forward_quantized=fake_forward_quantized)
    indexer.wq_b.forward = lambda _q: pytest.fail("wq_b must not re-quantize prolog's c^Q")

    quantized = (torch.zeros(3, 4, dtype=torch.int8), torch.ones(3))
    out = indexer._project_query(quantized)
    assert out.shape == (3, 6)
    assert len(calls) == 1
    assert calls[0][0] is quantized[0] and calls[0][1] is quantized[1]


@pytest.mark.parametrize("relative_path", ["kernels_npu/mla.py", "models/glm5_2.py"])
def test_prolog_modules_keep_the_vendor_import_lazy(relative_path: str) -> None:
    """The vendor package must stay a per-fusion dependency, not a startup one.

    ``kernels_npu._initialize_runtime`` imports every kernel module for every NPU
    model, so a module-scope ``import cann_ops_transformer`` would take the whole
    process down wherever the ops-transformer package is absent, even with the
    fused path switched off.
    """
    source = (Path(glm5_2.__file__).parents[1] / relative_path).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        assert not any(name.split(".")[0] == "cann_ops_transformer" for name in imported), (
            f"{relative_path}:{node.lineno} imports the vendor package at module scope"
        )
    # The fusion still needs the registration, so the import must not vanish.
    assert "import cann_ops_transformer" in source
