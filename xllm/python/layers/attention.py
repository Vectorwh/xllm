# Copyright 2025-2026 The xLLM Authors.
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

"""Attention layer that delegates runtime execution to the current backend.

The layer owns layer-local attention semantics. Wrapper state, plan, workspace
and KV cache reside in the FlashInferBackend, accessed via ForwardContext.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    wait_for_layer_load,
)


class Attention(nn.Module):
    """Thin attention layer that dispatches to the current backend."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        layer_id: int,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.layer_id = layer_id
        self.causal = causal
        # Tuple: (preceding, following) extents relative to each query.
        # Int: shared history width before the first query, with a fixed left
        # boundary; the query block follows self.causal. None: no window.
        self.attention_window: int | tuple[int, int] | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        wait_for_layer_load(self.layer_id)
        backend = get_forward_context().attention_backend
        if isinstance(self.attention_window, int) and not backend.supports_anchor_context_window:
            raise NotImplementedError(f"{type(backend).__name__} does not support anchor_context_window")
        return backend.execute(q, k, v, self)


def joint_diffusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Hardware-agnostic non-causal joint attention for diffusion models.

    Delegates to the active platform's fused kernel (``npu_fusion_attention``
    on NPU) and falls back to eager ``scaled_dot_product_attention``.
    """
    return kernels.joint_diffusion_attention(query, key, value, attn_mask, scale)
