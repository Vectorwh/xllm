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

"""JoyImageEditPlus transformer (double-stream, Wan-style patchification).

Port of ``xllm/models/dit/transformers/transformer_joyimage_edit_plus.h``.
``forward`` wraps ``forward_impl`` with :class:`SequenceParallelMixin` exactly
like the C++ ``sequence_parallel_forward`` call; the attention resharding
(``all_to_all_4D``) pads/unpads against the SAME bookkeeping instance the
wrapper records, replacing the C++ ``inline static`` padding map.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from xllm.python.distributed import collectives
from xllm.python.models.dit.base import DiTComponentBase, prefixed, require_loaded
from xllm.python.models.dit.transformers.common import (
    FeedForward,
    RMSNorm,
    TimestepEmbedding,
    Timesteps,
    _kernels,
    fp32_layer_norm,
)
from xllm.python.parallel import SequenceParallelMixin


def apply_rotary_emb_batched(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """x[B,S,H,D] * cos + rotate_half(x) * sin with cos/sin [B,S,D] (or [1,S,D])."""
    kernels = _kernels()
    if kernels is not None:
        return kernels.wan_interleaved_rope(x, cos, sin)

    cos_b = (cos.unsqueeze(2) if cos.dim() == 3 else cos).to(torch.float32)  # [B|1, S, 1, D]
    sin_b = (sin.unsqueeze(2) if sin.dim() == 3 else sin).to(torch.float32)

    x_float = x.to(torch.float32)
    # rotate_half: view the last dim as pairs, produce [-x_imag, x_real].
    x_pairs = x_float.reshape(x_float.size(0), x_float.size(1), x_float.size(2), -1, 2)
    x_real = x_pairs.select(-1, 0)
    x_imag = x_pairs.select(-1, 1)
    rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)  # [B, S, H, D]
    return (x_float * cos_b + rotated * sin_b).to(x.dtype)


def joint_attention(
    query: torch.Tensor,  # [B, S, H, D]
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Joint attention over [B, S, H, D] with an optional bool keep-mask."""
    kernels = _kernels()
    if kernels is not None:
        return kernels.joint_diffusion_attention(query, key, value, attn_mask)

    q = query.transpose(1, 2)  # [B, H, S, D]
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    out_dtype = q.dtype
    output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,  # [B, 1, 1, S] bool broadcasts over heads/query
        dropout_p=0.0,
        is_causal=False,
    )
    if output.dtype != out_dtype:
        output = output.to(out_dtype)
    return output.transpose(1, 2).contiguous()  # [B, S, H, D]


class Modulate(nn.Module):
    """Wan-style learnable modulation table: (table + x) chunked ``factor``-ways."""

    def __init__(self, hidden_size: int, factor: int) -> None:
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(torch.zeros(1, factor, hidden_size), requires_grad=False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: [B, hidden] -> [B, 1, hidden]
        xin = x.unsqueeze(1) if x.dim() != 3 else x
        summed = self.modulate_table.to(xin.dtype) + xin  # [B, factor, hidden]
        return [chunk.squeeze(1) for chunk in summed.chunk(self.factor, dim=1)]

    def load_weights(self, state_dicts: list) -> None:
        loaded = False
        for state_dict in state_dicts:
            if state_dict.has("modulate_table"):
                self.modulate_table.data.copy_(state_dict.get_tensor("modulate_table"))
                loaded = True
        if not loaded:
            raise RuntimeError("weight is not loaded for modulate_table")


class TextProjection(nn.Module):
    """PixArt-style text projection: Linear -> gelu_tanh -> Linear."""

    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.nn.functional.gelu(self.linear_1(caption), approximate="tanh"))

    def load_weights(self, state_dicts: list) -> None:
        loaded = {
            "linear_1.weight": False,
            "linear_1.bias": False,
            "linear_2.weight": False,
            "linear_2.bias": False,
        }
        for state_dict in state_dicts:
            for name, param in (
                ("linear_1.weight", self.linear_1.weight),
                ("linear_1.bias", self.linear_1.bias),
                ("linear_2.weight", self.linear_2.weight),
                ("linear_2.bias", self.linear_2.bias),
            ):
                if not loaded[name] and state_dict.has(name):
                    param.data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        require_loaded(loaded, type(self).__name__)


class TimeTextEmbedding(nn.Module):
    """timesteps -> time_embedder -> (temb, time_proj); text_embedder(ehs)."""

    def __init__(self, hidden_size: int, time_freq_dim: int, time_proj_dim: int, text_embed_dim: int) -> None:
        super().__init__()
        self.timesteps_proj = Timesteps(
            time_freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=1.0,
        )
        self.time_embedder = TimestepEmbedding(time_freq_dim, hidden_size)
        self.time_proj = nn.Linear(hidden_size, time_proj_dim, bias=True)
        self.text_embedder = TextProjection(text_embed_dim, hidden_size)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.timesteps_proj(timestep)
        temb = self.time_embedder(t)  # [B, hidden], computed in FP32
        temb = temb.to(encoder_hidden_states.dtype)
        act = torch.nn.functional.silu(temb)
        timestep_proj = self.time_proj(act)  # [B, time_proj_dim]
        text = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, text

    def keep_fp32_modules(self) -> None:
        self.time_embedder.float()

    def load_weights(self, state_dicts: list) -> None:
        self.time_embedder.load_weights(prefixed(state_dicts, "time_embedder."))
        self.text_embedder.load_weights(prefixed(state_dicts, "text_embedder."))
        loaded = {"time_proj.weight": False, "time_proj.bias": False}
        for state_dict in state_dicts:
            for name, param in (
                ("time_proj.weight", self.time_proj.weight),
                ("time_proj.bias", self.time_proj.bias),
            ):
                if not loaded[name] and state_dict.has(name):
                    param.data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        require_loaded(loaded, type(self).__name__)


class JoyAttention(nn.Module, SequenceParallelMixin):
    """Double-stream joint attention with sequence-parallel head resharding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        sp_group: str | None,
        padding_lengths: dict[str, int],
        eps: float = 1e-6,
    ) -> None:
        nn.Module.__init__(self)
        SequenceParallelMixin.__init__(self, sp_group=sp_group, padding_lengths=padding_lengths)
        self.heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.img_attn_q = nn.Linear(dim, inner, bias=True)
        self.img_attn_k = nn.Linear(dim, inner, bias=True)
        self.img_attn_v = nn.Linear(dim, inner, bias=True)
        self.img_attn_q_norm = RMSNorm(head_dim, eps)
        self.img_attn_k_norm = RMSNorm(head_dim, eps)
        self.img_attn_proj = nn.Linear(inner, dim, bias=True)

        self.txt_attn_q = nn.Linear(dim, inner, bias=True)
        self.txt_attn_k = nn.Linear(dim, inner, bias=True)
        self.txt_attn_v = nn.Linear(dim, inner, bias=True)
        self.txt_attn_q_norm = RMSNorm(head_dim, eps)
        self.txt_attn_k_norm = RMSNorm(head_dim, eps)
        self.txt_attn_proj = nn.Linear(inner, dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reshape = (self.heads, -1)
        sp_group = self.sp_group
        sp_size = collectives.named_group_size(sp_group, hidden_states.device)

        # Communication-matmul overlap (CMO), mirroring the QwenImage
        # QwenDoubleStreamAttnProcessorCMO2_0 path: launch each resharding
        # collective right after its projection so the collective's latency
        # hides behind the *next* projection's matmul on the compute stream,
        # instead of waiting for all six projections to finish first. The
        # trailing V collective still hides behind the q/k norm+RoPE below.
        img_q = self.img_attn_q(hidden_states).unflatten(-1, reshape)
        img_q_work = collectives.all_to_all_4D_async(img_q, 2, 1, sp_group)
        img_k = self.img_attn_k(hidden_states).unflatten(-1, reshape)
        img_k_work = collectives.all_to_all_4D_async(img_k, 2, 1, sp_group)
        img_v = self.img_attn_v(hidden_states).unflatten(-1, reshape)
        img_v_work = collectives.all_to_all_4D_async(img_v, 2, 1, sp_group)
        txt_q = self.txt_attn_q(encoder_hidden_states).unflatten(-1, reshape)
        txt_q_work = collectives.all_to_all_4D_async(txt_q, 2, 1, sp_group)
        txt_k = self.txt_attn_k(encoder_hidden_states).unflatten(-1, reshape)
        txt_k_work = collectives.all_to_all_4D_async(txt_k, 2, 1, sp_group)
        txt_v = self.txt_attn_v(encoder_hidden_states).unflatten(-1, reshape)
        txt_v_work = collectives.all_to_all_4D_async(txt_v, 2, 1, sp_group)

        img_q = self.unpad_tensor(img_q_work(), "hidden_states", 1)
        img_k = self.unpad_tensor(img_k_work(), "hidden_states", 1)
        txt_q = self.unpad_tensor(txt_q_work(), "encoder_hidden_states", 1)
        txt_k = self.unpad_tensor(txt_k_work(), "encoder_hidden_states", 1)

        # Fuse the img-stream per-head Q/K RMSNorm with RoPE through the kernel
        # layer (vLLM-Omni fused_qk_norm_rope NPU fast path); the text stream is
        # RoPE-free and keeps its norm-only path.
        kernels = _kernels()
        if rope_cos is not None and kernels is not None:
            img_q, img_k = kernels.fused_qk_norm_rope_wan(
                img_q,
                img_k,
                self.img_attn_q_norm.weight,
                self.img_attn_k_norm.weight,
                rope_cos,
                rope_sin,
                self.img_attn_q_norm.eps,
            )
        else:
            img_q = self.img_attn_q_norm(img_q)
            img_k = self.img_attn_k_norm(img_k)
            if rope_cos is not None:
                img_q = apply_rotary_emb_batched(img_q, rope_cos, rope_sin)
                img_k = apply_rotary_emb_batched(img_k, rope_cos, rope_sin)
        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        img_v = self.unpad_tensor(img_v_work(), "hidden_states", 1)
        txt_v = self.unpad_tensor(txt_v_work(), "encoder_hidden_states", 1)

        joint_q = torch.cat([img_q, txt_q], dim=1)
        joint_k = torch.cat([img_k, txt_k], dim=1)
        joint_v = torch.cat([img_v, txt_v], dim=1)
        joint = joint_attention(joint_q, joint_k, joint_v, attn_mask)
        joint = joint.to(joint_q.dtype)

        image_sequence_length = img_q.size(1)
        img_out = joint[:, :image_sequence_length]
        txt_out = joint[:, image_sequence_length:]

        img_out = self.pad_tensor(img_out, "hidden_states", 1)
        txt_out = self.pad_tensor(txt_out, "encoder_hidden_states", 1)
        img_out_work = collectives.all_to_all_4D_async(img_out, 1, 2, sp_group)
        txt_out_work = collectives.all_to_all_4D_async(txt_out, 1, 2, sp_group)
        img_out = self.img_attn_proj(img_out_work().flatten(2, 3))
        txt_out = self.txt_attn_proj(txt_out_work().flatten(2, 3))
        return img_out, txt_out

    def load_weights(self, state_dicts: list) -> None:
        self._load_qkv_projections(state_dicts, "img_attn_qkv", self.img_attn_q, self.img_attn_k, self.img_attn_v)
        self._load_qkv_projections(state_dicts, "txt_attn_qkv", self.txt_attn_q, self.txt_attn_k, self.txt_attn_v)
        loaded = {
            "img_attn_q_norm.weight": False,
            "img_attn_k_norm.weight": False,
            "txt_attn_q_norm.weight": False,
            "txt_attn_k_norm.weight": False,
            "img_attn_proj.weight": False,
            "img_attn_proj.bias": False,
            "txt_attn_proj.weight": False,
            "txt_attn_proj.bias": False,
        }
        params = {
            "img_attn_q_norm.weight": self.img_attn_q_norm.weight,
            "img_attn_k_norm.weight": self.img_attn_k_norm.weight,
            "txt_attn_q_norm.weight": self.txt_attn_q_norm.weight,
            "txt_attn_k_norm.weight": self.txt_attn_k_norm.weight,
            "img_attn_proj.weight": self.img_attn_proj.weight,
            "img_attn_proj.bias": self.img_attn_proj.bias,
            "txt_attn_proj.weight": self.txt_attn_proj.weight,
            "txt_attn_proj.bias": self.txt_attn_proj.bias,
        }
        for state_dict in state_dicts:
            for name in loaded:
                if not loaded[name] and state_dict.has(name):
                    params[name].data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        require_loaded(loaded, type(self).__name__)

    def _load_qkv_projections(
        self,
        state_dicts: list,
        prefix: str,
        q_projection: nn.Linear,
        k_projection: nn.Linear,
        v_projection: nn.Linear,
    ) -> None:
        fused_weight = None
        fused_bias = None
        for state_dict in state_dicts:
            if state_dict.has(f"{prefix}.weight"):
                fused_weight = state_dict.get_tensor(f"{prefix}.weight")
            if state_dict.has(f"{prefix}.bias"):
                fused_bias = state_dict.get_tensor(f"{prefix}.bias")
        if fused_weight is None:
            raise RuntimeError(f"{prefix}.weight is missing")
        if fused_bias is None:
            raise RuntimeError(f"{prefix}.bias is missing")

        inner = self.heads * self.head_dim
        if fused_weight.dim() != 2 or fused_weight.size(0) != inner * 3:
            raise RuntimeError(f"{prefix}.weight output size mismatch")
        if fused_bias.dim() != 1 or fused_bias.size(0) != inner * 3:
            raise RuntimeError(f"{prefix}.bias output size mismatch")

        for index, projection in enumerate((q_projection, k_projection, v_projection)):
            start = index * inner
            projection.weight.data.copy_(fused_weight[start : start + inner])
            projection.bias.data.copy_(fused_bias[start : start + inner])


class TransformerBlock(nn.Module):
    """Double-stream transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        sp_group: str | None,
        padding_lengths: dict[str, int],
        mlp_width_ratio: float = 4.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.img_mod = Modulate(dim, 6)
        self.img_mlp = FeedForward(dim, dim, mult=mlp_width_ratio)
        self.txt_mod = Modulate(dim, 6)
        self.txt_mlp = FeedForward(dim, dim, mult=mlp_width_ratio)
        self.attn = JoyAttention(dim, num_heads, head_dim, sp_group, padding_lengths, eps)

    @staticmethod
    def _ada_layer_norm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
        """LayerNorm(x) * (1 + scale) + shift, fused on NPU, eager fp32 fallback."""
        kernels = _kernels()
        if kernels is not None:
            return kernels.ada_layer_norm(x, scale, shift, eps)
        return fp32_layer_norm(x, eps) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,  # [B, 6, hidden]
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod = self.img_mod(temb)  # 6 x [B, hidden]
        txt_mod = self.txt_mod(temb)
        img_s1, img_c1, img_g1, img_s2, img_c2, img_g2 = img_mod
        txt_s1, txt_c1, txt_g1, txt_s2, txt_c2, txt_g2 = txt_mod

        hs = hidden_states
        ehs = encoder_hidden_states

        # --- attention ---
        img_modulated = self._ada_layer_norm(hs, img_c1, img_s1, self.eps)
        txt_modulated = self._ada_layer_norm(ehs, txt_c1, txt_s1, self.eps)

        img_attn, txt_attn = self.attn(img_modulated, txt_modulated, rope_cos, rope_sin, attn_mask)

        img_attn.mul_(img_g1.unsqueeze(1))
        hs = img_attn.add_(hs)
        txt_attn.mul_(txt_g1.unsqueeze(1))
        ehs = txt_attn.add_(ehs)

        # --- FFN ---
        img_ffn_in = self._ada_layer_norm(hs, img_c2, img_s2, self.eps)
        txt_ffn_in = self._ada_layer_norm(ehs, txt_c2, txt_s2, self.eps)
        img_ffn = self.img_mlp(img_ffn_in)
        txt_ffn = self.txt_mlp(txt_ffn_in)
        hs = img_ffn.mul_(img_g2.unsqueeze(1)).add_(hs)
        ehs = txt_ffn.mul_(txt_g2.unsqueeze(1)).add_(ehs)

        return hs, ehs

    def load_weights(self, state_dicts: list) -> None:
        self.img_mod.load_weights(prefixed(state_dicts, "img_mod."))
        self.txt_mod.load_weights(prefixed(state_dicts, "txt_mod."))
        self.attn.load_weights(prefixed(state_dicts, "attn."))
        self._load_mlp(state_dicts, "img_mlp", self.img_mlp)
        self._load_mlp(state_dicts, "txt_mlp", self.txt_mlp)

    def _load_mlp(self, state_dicts: list, prefix: str, mlp: FeedForward) -> None:
        names = ("net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias")
        loaded = {name: False for name in names}
        params = {
            "net.0.proj.weight": mlp.linear1.weight,
            "net.0.proj.bias": mlp.linear1.bias,
            "net.2.weight": mlp.linear2.weight,
            "net.2.bias": mlp.linear2.bias,
        }
        for state_dict in state_dicts:
            for name in names:
                full = f"{prefix}.{name}"
                if not loaded[name] and state_dict.has(full):
                    params[name].data.copy_(state_dict.get_tensor(full))
                    loaded[name] = True
        require_loaded(loaded, f"{type(self).__name__}.{prefix}")


class JoyImageEditPlusTransformer3DModel(DiTComponentBase, SequenceParallelMixin):
    """Double-stream DiT over 6D patchified latents [B, N, C, pt, ph, pw]."""

    def __init__(
        self,
        config: dict,
        sp_group: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        DiTComponentBase.__init__(self)
        SequenceParallelMixin.__init__(self, sp_group=sp_group)
        self.hidden_size = int(config.get("hidden_size", 4096))
        self.num_heads = int(config.get("num_attention_heads", 32))
        num_layers = int(config.get("num_layers", 40))
        self.in_channels = int(config.get("in_channels", 16))
        out_channels = int(config.get("out_channels", 16))
        self.out_channels = out_channels if out_channels > 0 else self.in_channels
        patch_size = list(config.get("patch_size", [1, 2, 2]))
        mlp_width_ratio = float(config.get("mlp_width_ratio", 4.0))
        text_dim = int(config.get("text_dim", 4096))
        eps = float(config.get("eps", 1e-6))

        self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        # Conv3d patchifier: kernel == stride == patch_size.
        self.img_in = nn.Conv3d(
            self.in_channels,
            self.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.condition_embedder = TimeTextEmbedding(
            self.hidden_size,
            time_freq_dim=256,
            time_proj_dim=self.hidden_size * 6,
            text_embed_dim=text_dim,
        )

        self.double_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.hidden_size,
                    self.num_heads,
                    self.head_dim,
                    sp_group,
                    self._padding_lengths,
                    mlp_width_ratio,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        patch_prod = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_out = nn.Linear(self.hidden_size, self.out_channels * patch_prod, bias=True)
        self._loaded_flags = {"img_in.weight": False, "img_in.bias": False}

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, N, C, pt, ph, pw]
        timestep: torch.Tensor,  # [B]
        encoder_hidden_states: torch.Tensor,  # [B, L, text_dim]
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        use_cfg: bool = False,
        step_index: int = 1,
        cache_hooks: Any = None,
    ) -> torch.Tensor:
        outputs = self.sequence_parallel_forward(
            {
                "hidden_states": (hidden_states, 1),
                "encoder_hidden_states": (encoder_hidden_states, 1),
            },
            lambda local: {
                "hidden_states": (
                    self._forward_impl(
                        local["hidden_states"],
                        timestep,
                        local["encoder_hidden_states"],
                        rope_cos,
                        rope_sin,
                        attention_mask,
                        use_cfg,
                        step_index,
                        cache_hooks,
                    ),
                    1,
                )
            },
        )
        return outputs["hidden_states"][0]

    def _forward_impl(
        self,
        local_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        local_encoder_hidden_states: torch.Tensor,
        rope_cos: torch.Tensor | None,
        rope_sin: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        use_cfg: bool,
        step_index: int,
        cache_hooks: Any,
    ) -> torch.Tensor:
        batch_size = local_hidden_states.size(0)
        sequence_length = local_hidden_states.size(1)
        channels = local_hidden_states.size(2)
        pt = local_hidden_states.size(3)
        ph = local_hidden_states.size(4)
        pw = local_hidden_states.size(5)

        # 1. Condition embeddings.
        _, vec, txt = self.condition_embedder(timestep, local_encoder_hidden_states)
        # Blocks re-add modulate_table to a [B, 6, hidden] signal.
        temb6 = vec.unflatten(1, (6, -1))

        # 2. Patchify via Conv3d.
        x = local_hidden_states.reshape(batch_size * sequence_length, channels, pt, ph, pw)
        x = self.img_in(x)  # [B*N, hidden, 1, 1, 1]
        img = x.reshape(batch_size, sequence_length, self.hidden_size)

        # 3. Blocks with optional DiT cache hooks.
        original_img = img
        original_txt = txt
        use_step_cache = False
        if cache_hooks is not None:
            use_step_cache = cache_hooks.on_before_step(
                step_index,
                {"hidden_states": img, "encoder_hidden_states": txt},
                use_cfg,
            )

        if not use_step_cache:
            for block_index, block in enumerate(self.double_blocks):
                use_block_cache = False
                if cache_hooks is not None:
                    use_block_cache = cache_hooks.on_before_block(block_index, use_cfg)
                if not use_block_cache:
                    img, txt = block(img, txt, temb6, rope_cos, rope_sin, attention_mask)
                if cache_hooks is not None:
                    out = cache_hooks.on_after_block(
                        block_index,
                        {
                            "hidden_states": img,
                            "encoder_hidden_states": txt,
                            "original_hidden_states": original_img,
                            "original_encoder_hidden_states": original_txt,
                        },
                    )
                    img = out["hidden_states"]
                    txt = out["encoder_hidden_states"]

        if cache_hooks is not None:
            out = cache_hooks.on_after_step(
                {
                    "hidden_states": img,
                    "encoder_hidden_states": txt,
                    "original_hidden_states": original_img,
                    "original_encoder_hidden_states": original_txt,
                }
            )
            img = out["hidden_states"]

        # 4. Output projection + reshape to [B, N, C_out, pt, ph, pw].
        img = self.proj_out(fp32_layer_norm(img, 1e-6))
        return img.reshape(batch_size, sequence_length, pt, ph, pw, self.out_channels).permute(0, 1, 5, 2, 3, 4)

    def keep_fp32_modules(self) -> None:
        self.condition_embedder.keep_fp32_modules()

    def load_weights(self, state_dicts: list) -> None:
        for state_dict in state_dicts:
            # Conv3d img_in: load raw weight/bias.
            if state_dict.has("img_in.weight"):
                self.img_in.weight.data.copy_(state_dict.get_tensor("img_in.weight"))
                self._loaded_flags["img_in.weight"] = True
            if state_dict.has("img_in.bias"):
                self.img_in.bias.data.copy_(state_dict.get_tensor("img_in.bias"))
                self._loaded_flags["img_in.bias"] = True

        prefixed_embedder = prefixed(state_dicts, "condition_embedder.")
        self.condition_embedder.load_weights(prefixed_embedder)
        self._load_proj_out(state_dicts)
        for index, block in enumerate(self.double_blocks):
            block.load_weights(prefixed(state_dicts, f"double_blocks.{index}."))

        require_loaded(self._loaded_flags, type(self).__name__)

    def _load_proj_out(self, state_dicts: list) -> None:
        loaded = {"proj_out.weight": False, "proj_out.bias": False}
        params = {
            "proj_out.weight": self.proj_out.weight,
            "proj_out.bias": self.proj_out.bias,
        }
        for state_dict in state_dicts:
            for name in loaded:
                if not loaded[name] and state_dict.has(name):
                    params[name].data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        require_loaded(loaded, "proj_out")
