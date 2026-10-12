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

"""NPU paged-attention support kernels.

Both operators are capturable: they read only their arguments, so a decode
graph can replay them without re-planning.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import List

import torch

# Full bidirectional window (INT32_MAX), matching C++ kSwaIntMax.
_FULL_ATTENTION_WINDOW = 2147483647

reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache
update_decode_graph_metadata = torch.ops.xllm_ops.update_decode_graph_metadata

_HAS_MINDIESD = find_spec("mindiesd") is not None
_TRANSPOSE_BATCHMATMUL = getattr(
    torch.ops.npu,
    "npu_transpose_batchmatmul",
    None,
)


def vision_fusion_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    actual_seq_qlen: list[int],
    actual_seq_kvlen: list[int],
    num_heads: int,
    scale: float,
    input_layout: str = "TND",
) -> torch.Tensor:
    """Run fused self-attention for ViT blocks via ``torch_npu.npu_fusion_attention``.

    Args:
        q, k, v: Query/key/value in the layout specified by ``input_layout``.
        actual_seq_qlen: Cumulative query sequence lengths.
        actual_seq_kvlen: Cumulative key/value sequence lengths.
        num_heads: Number of attention heads.
        scale: Attention scaling factor.
        input_layout: Tensor layout (default ``"TND"``).

    Returns:
        Attention output with same shape as ``q``.
    """
    import torch_npu

    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        head_num=num_heads,
        scale=scale,
        input_layout=input_layout,
    )[0]


def packed_infer_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    actual_seq_lengths: list[int],
    num_heads: int,
    num_key_value_heads: int,
    scale: float,
    input_layout: str = "TND",
) -> torch.Tensor:
    """Run bidirectional packed self-attention using fused infer attention.

    Cumulative sequence lengths keep media items independent. This uses the
    infer-attention operator, distinct from ``vision_fusion_attention``.
    """
    import torch_npu

    # No mask or paged KV: sparse_mode=0 selects dense attention, and the
    # full window states the bidirectional contract explicitly.
    return torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key.contiguous(),
        value=value.contiguous(),
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        scale=scale,
        input_layout=input_layout,
        sparse_mode=0,
        pre_tokens=_FULL_ATTENTION_WINDOW,
        next_tokens=_FULL_ATTENTION_WINDOW,
    )[0]


def batch_matmul_transpose(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project MLA input [T,H,D] with weight [H,D,O] into [T,H,O]."""
    return torch.ops.npu.npu_transpose_batchmatmul(
        x,
        weight,
        perm_x1=(1, 0, 2),
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )


def joint_diffusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Non-causal joint attention for diffusion, via the best NPU kernel.

    The DiT ``joint_attention`` passes a boolean keep-mask (``True`` = attend).
    ``npu_fusion_attention`` uses the opposite convention (``1`` = masked out),
    so the mask is inverted on the NPU fast path. Falls back to
    ``scaled_dot_product_attention`` for non-NPU devices or non-boolean masks.

    Args:
        query, key, value: ``[B, S, H, D]`` activations.
        attn_mask: ``[B, 1, S, S]`` boolean keep-mask, or ``None``.
        scale: Attention scaling factor; defaults to ``head_dim ** -0.5``.

    Returns:
        Attention output ``[B, S, H, D]``.
    """
    head_dim = query.size(-1)
    if scale is None:
        scale = head_dim**-0.5
    num_heads = query.size(2)
    mask_is_bool = attn_mask is None or attn_mask.dtype == torch.bool

    if query.device.type == "npu" and mask_is_bool:
        if _HAS_MINDIESD:
            try:
                from mindiesd import attention_forward

                # mindiesd's ``fused_attn_score`` (BSND) is the same
                # ``npu_fusion_attention`` call the direct path makes, and it
                # inverts the boolean keep-mask internally, so the raw
                # ``attn_mask`` is passed through untouched.
                return attention_forward(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    scale=scale,
                    fused=True,
                    head_first=False,
                    opt_mode="manual",
                    op_type="fused_attn_score",
                    layout="BSND",
                )
            except ImportError:
                pass

        import torch_npu

        # npu_fusion_attention accepts the native [B, S, H, D] BSND layout
        # directly (matching the C++ joyimage_joint_attention), so no
        # transpose/contiguous is needed on the way in or out.
        drop_mask = None if attn_mask is None else ~attn_mask
        out = torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=num_heads,
            input_layout="BSND",
            scale=scale,
            atten_mask=drop_mask,
        )[0]
        return out  # [B, S, H, D]

    import torch.nn.functional as F

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous()


__all__ = [
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "vision_fusion_attention",
    "packed_infer_attention",
    "joint_diffusion_attention",
    "batch_matmul_transpose",
]
