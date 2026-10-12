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

"""NPU normalization kernels."""

from __future__ import annotations

from importlib.util import find_spec

import torch
import torch_npu

rms_norm = torch.ops.xllm_ops.rms_norm
fused_add_rms_norm = torch.ops.xllm_ops.fused_add_rms_norm
fused_add_gemma_rms_norm = torch.ops.xllm_ops.fused_add_gemma_rms_norm
_FUSED_ADD_RMS_NORM_DYNAMIC_QUANT = torch_npu.npu_add_rms_norm_dynamic_quant

# DSV4.1 ``RmsNormCastVa`` (third_party/xllm_ops, ACLNN ``aclnnRmsNormCastVa``):
# ``rms_norm_cast_va(x, gamma, eps) -> (y, y_fp32)`` where ``y`` is the
# RMSNorm result cast back to the input dtype and ``y_fp32`` is the widened
# ROUNDED result (``y.float()``), i.e. ``RMSNorm`` + ``Tensor.float()``.
rms_norm_cast_va = torch.ops.xllm_ops.rms_norm_cast_va
_HAS_MINDIESD = find_spec("mindiesd") is not None

# ``mindiesd.layernorm_scale_shift`` needs an ``nn.LayerNorm`` descriptor. The
# DiT AdaLayerNorm has no affine parameters (weight/bias are ``None``), so a
# single cached, parameter-less ``nn.LayerNorm`` per (hidden, eps) key suffices;
# its forward is never called on the fused path (only eps/normalized_shape are
# read), so it stays on the CPU.
_ADA_LAYER_NORM_HOLDERS: dict[tuple[int, float], torch.nn.LayerNorm] = {}


def _ada_layer_norm_holder(hidden_size: int, eps: float) -> torch.nn.LayerNorm:
    key = (hidden_size, eps)
    holder = _ADA_LAYER_NORM_HOLDERS.get(key)
    if holder is None:
        holder = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        _ADA_LAYER_NORM_HOLDERS[key] = holder
    return holder


def gemma_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Gemma RMSNorm with the checkpoint's additive gamma."""
    output, _ = torch_npu.npu_gemma_rms_norm(value, weight, eps)
    return output


def rms_norm_fp32(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm over the last dim computed in float32 (Wan DiT per-head norm).

    ``torch_npu.npu_rms_norm`` accumulates in fp32 and, when ``weight`` is kept
    in fp32, is bit-exact with the eager fp32 reference used by the DiT
    ``RMSNorm`` layers. Casting ``weight`` to the activation dtype would round
    it and drift the result, so the weight is passed through untouched.

    Args:
        value: ``[..., D]`` tensor (2-8 dims).
        weight: ``[D]`` RMSNorm scale, typically fp32.
        eps: Epsilon added before the reciprocal square root.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    if value.device.type == "npu" and value.dim() >= 2:
        return torch_npu.npu_rms_norm(value.contiguous(), weight, epsilon=eps)[0]

    # Eager fp32 reference (matches models/dit/transformers/common.py RMSNorm).
    x_float = value.to(torch.float32)
    normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * weight.to(torch.float32)).to(value.dtype)


def ada_layer_norm(
    value: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused adaptive LayerNorm: ``LayerNorm(value) * (1 + scale) + shift``.

    Wraps the ``xllm_ops::ada_layer_norm`` custom op, which calls
    ``aclnnAdaLayerNorm`` under the hood. The modulation ``(1 + scale)`` is
    applied inside the kernel, so the raw ``scale`` is passed through untouched
    (matching the C++ ``AdaLayerNorm`` layer). ``value`` is ``[B, S, H]`` and
    ``scale``/``shift`` are ``[B, H]`` (broadcast over the sequence dim).

    Args:
        value: ``[B, S, H]`` input tensor.
        scale: ``[B, H]`` raw scale modulation factor.
        shift: ``[B, H]`` shift modulation factor.
        eps: Epsilon added before the reciprocal square root.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    if value.device.type == "npu" and _HAS_MINDIESD:
        try:
            from mindiesd import layernorm_scale_shift

            # mindiesd's ``ops.adaln`` is the same ``aclnnAdaLayerNorm`` the
            # ``xllm_ops::ada_layer_norm`` fallback wraps, so the result is
            # bit-identical; routing here keeps all Python DiT models on the
            # mindiesd operator layer.
            holder = _ada_layer_norm_holder(value.size(-1), eps)
            return layernorm_scale_shift(holder, value.contiguous(), scale, shift, fused=True)
        except ImportError:
            pass

    return torch.ops.xllm_ops.ada_layer_norm(value.contiguous(), scale, shift, None, None, eps)


def fused_add_rms_norm_dynamic_quant(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add residual, apply RMSNorm, and dynamically quantize the result."""
    outputs = _FUSED_ADD_RMS_NORM_DYNAMIC_QUANT(
        value,
        residual,
        weight,
        epsilon=eps,
        output_mask=[True, False],
    )
    return outputs[0], outputs[3], outputs[2]


rms_norm_dynamic_quant = torch.ops.xllm_ops.rms_norm_dynamic_quant


def l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the last dimension of ``value`` to unit L2 norm.

    Args:
        value: Tensor whose last dimension is normalized.
        eps: Added to the squared norm before the reciprocal square root.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    return torch.ops.xllm_ops.l2_norm(value, eps)


def rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm to ``value`` and gate the result with ``silu(gate)``.

    Args:
        value: Tensor to normalize.
        gate: Gate applied after normalization (SiLU is applied internally).
        weight: RMSNorm weight over the last dimension.
        eps: RMSNorm epsilon.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    return torch.ops.xllm_ops.rms_norm_gated(value, gate, weight, eps)


def rms_norm_sigmoid_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm to ``value`` and gate the result with ``sigmoid(gate)``.

    Args:
        value: Tensor to normalize.
        gate: Gate applied after normalization (sigmoid is applied internally).
        weight: RMSNorm weight over the last dimension.
        eps: RMSNorm epsilon.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    from .triton.rms_norm import (
        rms_norm_sigmoid_gated as triton_rms_norm_sigmoid_gated,
    )

    return triton_rms_norm_sigmoid_gated(value, gate, weight, eps)


def fused_eh_norm(
    embed: torch.Tensor,
    carried: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize embedding and carried states, then concatenate the results.

    Args:
        embed: Token embedding to normalize.
        carried: Carried hidden state to normalize.
        enorm_weight: RMSNorm weight for ``embed``.
        hnorm_weight: RMSNorm weight for ``carried``.
        eps: RMSNorm epsilon shared by both inputs.

    Returns:
        The normalized inputs concatenated along the final dimension.
    """
    from .triton.eh_norm import fused_eh_norm as triton_fused_eh_norm

    return triton_fused_eh_norm(
        embed,
        carried,
        enorm_weight,
        hnorm_weight,
        eps,
    )


__all__ = [
    "rms_norm",
    "gemma_rms_norm",
    "rms_norm_fp32",
    "ada_layer_norm",
    "fused_add_rms_norm",
    "fused_add_gemma_rms_norm",
    "fused_add_rms_norm_dynamic_quant",
    "rms_norm_dynamic_quant",
    "l2_norm",
    "rms_norm_gated",
    "rms_norm_sigmoid_gated",
    "fused_eh_norm",
    "rms_norm_cast_va",
]
