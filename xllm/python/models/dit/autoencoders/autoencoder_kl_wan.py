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

"""AutoencoderKLWan (Wan 2.x video VAE, causal-conv3d streaming variant).

Port of ``xllm/models/dit/autoencoders/autoencoder_kl_wan.h``. Encode walks
the video in 4-frame chunks and decode frame-by-frame, both threaded through
the per-conv feature cache (``feat_cache``/``feat_idx``). All spatial ops go
through :class:`VaeParallelMixin` so the W dimension can be sharded; the halo
exchange and variable-width merge live in the mixin, this module only places
the calls at the same points as the C++ original.

One deliberate deviation: the C++ ``WanResidualDownBlockImpl`` declares an
``avg_shortcut_`` that is never registered (the forward branch is dead), so
the Python block has no shortcut either.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.models.dit.base import DiTComponentBase, prefixed
from xllm.python.parallel import VaeParallelMixin

CACHE_T = 2


def _has_cache(t: torch.Tensor | None) -> bool:
    return t is not None and t.numel() > 0


class DiagonalGaussianDistribution:
    """Diagonal gaussian over channel-split (mean, logvar) parameters."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.deterministic = deterministic
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.std = self.std.fill_(0.0)
            self.var = self.var.fill_(0.0)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, seed: int = 42) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            dtype=self.mean.dtype,
            device="cpu",
        ).to(self.mean.device)
        return self.mean + self.std * noise


class AvgDown3D(nn.Module):
    """Channel-averaged downsample shortcut (no weights)."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("in_channels * factor must be divisible by out_channels")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.size(2) % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // self.factor_t, h // self.factor_s, w // self.factor_s)
        x = x.view(b, self.out_channels, self.group_size, t // self.factor_t, h // self.factor_s, w // self.factor_s)
        return x.mean(2)


class DupUp3D(nn.Module):
    """Channel/temporal/spatial duplication upsample shortcut (no weights)."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels:
            raise ValueError("out_channels * factor must be divisible by in_channels")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :]
        return x


class WanCausalConv3D(nn.Module, VaeParallelMixin):
    """Causal Conv3d: temporal front padding from a cache, halo exchange in W."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        vae_group: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        nn.Module.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)
        self.padding = list(padding)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            # Conv3d always gets 0 temporal padding — causal temporal padding
            # is handled via cache_x concatenation. padding layout:
            # {temporal, height, width}.
            padding=(0, padding[1], padding[2]),
            bias=True,
            dtype=dtype,
        )
        # 6-element F.pad format {W_left, W_right, H_left, H_right, T_front,
        # T_back}; only the temporal front (2 * pad_t) is non-zero.
        self._pad = (0, 0, 0, 0, 2 * padding[0], 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        pad = list(self._pad)
        # Halo exchange only matters for 3x3 spatial convolutions (kernel=3,
        # pad=1); 1x1 convs skip it.
        use_halo = self.vae_parallel_enabled(x.device) and self.padding[1] == 1 and self.padding[2] == 1

        # Temporal padding (causal) — must come BEFORE halo exchange.
        if cache_x is not None and pad[4] > 0:
            cache = cache_x.to(x.device)
            x = torch.cat([cache, x], dim=2)
            pad[4] -= cache.size(2)

        if use_halo:
            x = self.exchange(x, pad=True)

        x = F.pad(x, pad)
        out = self.conv(x)

        # Trim the halo columns the exchange added (kernel=3, pad=1 preserves
        # spatial size, so the output is 2 columns too wide in W).
        if use_halo:
            out = out[..., 1 : out.size(-1) - 1]
        return out

    def load_weights(self, state_dicts: list) -> None:
        loaded_weight = False
        loaded_bias = False
        for state_dict in state_dicts:
            if not loaded_weight and state_dict.has("weight"):
                self.conv.weight.data.copy_(state_dict.get_tensor("weight"))
                loaded_weight = True
            if not loaded_bias and state_dict.has("bias"):
                self.conv.bias.data.copy_(state_dict.get_tensor("bias"))
                loaded_bias = True
        if not loaded_weight:
            raise RuntimeError(f"weight is not loaded for {type(self).__name__}")
        if not loaded_bias:
            raise RuntimeError(f"bias is not loaded for {type(self).__name__}")


class WanRMSNorm(nn.Module):
    """Channel-first RMSNorm normalizing in fp32 (diffusers-compatible)."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False) -> None:
        super().__init__()
        self.channel_first = channel_first
        self.images = images
        self.bias_enabled = bias
        broadcastable = (1, 1) if images else (1, 1, 1)
        shape = (dim, *broadcastable) if channel_first else (dim,)
        self.register_parameter("gamma", nn.Parameter(torch.ones(shape), requires_grad=False))
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(shape), requires_grad=False))
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dim = 1 if self.channel_first else -1
        # Normalize fp16/bf16 inputs in fp32 for stability, then cast back.
        needs_fp32 = x.dtype in (torch.float16, torch.bfloat16)
        norm_input = x.to(torch.float32) if needs_fp32 else x
        normed = F.normalize(norm_input, p=2.0, dim=norm_dim, eps=1e-12).to(x.dtype)
        out = normed * self.scale * self.gamma
        if self.bias_enabled:
            out = out + self.bias
        return out

    def load_weights(self, state_dicts: list) -> None:
        loaded = False
        for state_dict in state_dicts:
            if not loaded and state_dict.has("gamma"):
                self.gamma.data.copy_(state_dict.get_tensor("gamma"))
                loaded = True
            if self.bias_enabled and state_dict.has("bias"):
                self.bias.data.copy_(state_dict.get_tensor("bias"))
        if not loaded:
            raise RuntimeError("gamma is not loaded for WanRMSNorm")


class WanUpsample(nn.Module):
    """Nearest-exact 2x2 spatial upsample performed in fp32."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.interpolate(x.to(torch.float32), scale_factor=(2.0, 2.0), mode="nearest-exact")
        return result.to(x.dtype)


class WanResample(nn.Module, VaeParallelMixin):
    """Spatial/temporal resample block with halo exchange around it."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int = -1,
        vae_group: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        nn.Module.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)
        self.mode = mode
        if upsample_out_dim == -1:
            upsample_out_dim = dim // 2

        layers: list[nn.Module] = []
        if mode in ("upsample2d", "upsample3d"):
            layers = [WanUpsample(), nn.Conv2d(dim, upsample_out_dim, kernel_size=3, padding=1, dtype=dtype)]
        elif mode in ("downsample2d", "downsample3d"):
            layers = [
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=(2, 2), dtype=dtype),
            ]
        else:
            layers = [nn.Identity()]
        self.resample = nn.Sequential(*layers)

        self.time_conv: WanCausalConv3D | None = None
        if mode == "upsample3d":
            self.time_conv = WanCausalConv3D(dim, dim * 2, (3, 1, 1), (1, 1, 1), (1, 0, 0), vae_group, dtype)
        elif mode == "downsample3d":
            self.time_conv = WanCausalConv3D(dim, dim, (3, 1, 1), (2, 1, 1), (0, 0, 0), vae_group, dtype)

        self.register_parameter("rep_tensor", nn.Parameter(torch.tensor([-999.0]), requires_grad=False))

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]

        b, c, t, h, w = x.shape
        is_upsample = self.mode in ("upsample2d", "upsample3d")
        is_downsample = self.mode in ("downsample2d", "downsample3d")

        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if not _has_cache(feat_cache[idx]):
                feat_cache[idx] = self.rep_tensor
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:].clone()
                if (
                    cache_x.size(2) < 2
                    and _has_cache(feat_cache[idx])
                    and not torch.equal(self.rep_tensor, feat_cache[idx])
                ):
                    cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
                if (
                    cache_x.size(2) < 2
                    and _has_cache(feat_cache[idx])
                    and torch.equal(self.rep_tensor, feat_cache[idx])
                ):
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                if torch.equal(self.rep_tensor, feat_cache[idx]):
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.view(b, 2, c, t, h, w)
                x = torch.stack([x[:, 0], x[:, 1]], dim=3)
                x = x.view(b, c, t * 2, h, w)
        t = x.size(2)

        # Upsample: exchange halo columns before spatial upsampling.
        if is_upsample:
            x = self.exchange(x, pad=False)
            w = x.size(-1)
        # Downsample: merge spatial slices before spatial downsampling.
        if is_downsample:
            x = self.merge(x)
            w = x.size(-1)
            h = x.size(-2)

        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        # Upsample: trim the excess columns introduced by halo exchange — the
        # interpolate doubles each halo column, so trim 2 per contributed side.
        if is_upsample:
            x = self.trim_halo(x, per_side=2)
        # Downsample: split the full result back to local slices.
        if is_downsample:
            x = self.split(x)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if not _has_cache(feat_cache[idx]):
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:], x], dim=2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x

    def load_weights(self, state_dicts: list) -> None:
        conv = self.resample[1] if not isinstance(self.resample[0], nn.Identity) else None
        if conv is not None:
            loaded_weight = False
            loaded_bias = False
            for state_dict in state_dicts:
                if not loaded_weight and state_dict.has("resample.1.weight"):
                    conv.weight.data.copy_(state_dict.get_tensor("resample.1.weight"))
                    loaded_weight = True
                if not loaded_bias and state_dict.has("resample.1.bias"):
                    conv.bias.data.copy_(state_dict.get_tensor("resample.1.bias"))
                    loaded_bias = True
            if not loaded_weight:
                raise RuntimeError("weight is not loaded for WanResample resample.1")
            if not loaded_bias:
                raise RuntimeError("bias is not loaded for WanResample resample.1")
        if self.time_conv is not None:
            self.time_conv.load_weights(prefixed(state_dicts, "time_conv."))


class WanResidualBlock(nn.Module):
    """RMSNorm -> causal conv -> RMSNorm -> causal conv with shortcut."""

    def __init__(
        self, in_dim: int, out_dim: int, vae_group: str | None, dtype: torch.dtype, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm1 = WanRMSNorm(in_dim, channel_first=True, images=False)
        self.conv1 = WanCausalConv3D(in_dim, out_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group, dtype)
        self.norm2 = WanRMSNorm(out_dim, channel_first=True, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3D(out_dim, out_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group, dtype)
        self.conv_shortcut: WanCausalConv3D | None = None
        if in_dim != out_dim:
            self.conv_shortcut = WanCausalConv3D(in_dim, out_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), vae_group, dtype)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]

        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x

        x = self.norm1(x)
        x = F.silu(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.size(2) < 2 and _has_cache(feat_cache[idx]):
                cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.size(2) < 2 and idx < len(feat_cache) and _has_cache(feat_cache[idx]):
                cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h

    def load_weights(self, state_dicts: list) -> None:
        self.norm1.load_weights(prefixed(state_dicts, "norm1."))
        self.conv1.load_weights(prefixed(state_dicts, "conv1."))
        self.norm2.load_weights(prefixed(state_dicts, "norm2."))
        self.conv2.load_weights(prefixed(state_dicts, "conv2."))
        if self.conv_shortcut is not None:
            self.conv_shortcut.load_weights(prefixed(state_dicts, "conv_shortcut."))


class WanAttentionBlock(nn.Module, VaeParallelMixin):
    """Single-head spatial self-attention with channel-first RMSNorm."""

    def __init__(self, dim: int, vae_group: str | None) -> None:
        nn.Module.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)
        self.norm = WanRMSNorm(dim, channel_first=True, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size, channels, time, height, width = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)

        if self.vae_parallel_enabled(x.device):
            # Gather K/V in the spatial domain before the BNSD reshape.
            spatial_chunks = qkv.chunk(3, dim=1)
            k_sp = self.merge(spatial_chunks[1].unsqueeze(2)).squeeze(2)
            v_sp = self.merge(spatial_chunks[2].unsqueeze(2)).squeeze(2)
            width_global = k_sp.size(-1)
            q = (
                spatial_chunks[0]
                .reshape(batch_size * time, 1, channels, height * width)
                .permute(0, 1, 3, 2)
                .contiguous()
            )
            k = k_sp.reshape(batch_size * time, 1, channels, height * width_global).permute(0, 1, 3, 2).contiguous()
            v = v_sp.reshape(batch_size * time, 1, channels, height * width_global).permute(0, 1, 3, 2).contiguous()
        else:
            qkv = qkv.reshape(batch_size * time, 1, channels * 3, height * width)
            qkv = qkv.permute(0, 1, 3, 2).contiguous()
            q, k, v = qkv.chunk(3, dim=-1)

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        attn_out = attn_output.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        attn_out = self.proj(attn_out)

        attn_out = attn_out.view(batch_size, time, channels, height, width).permute(0, 2, 1, 3, 4)
        return attn_out + identity

    def load_weights(self, state_dicts: list) -> None:
        self.norm.load_weights(prefixed(state_dicts, "norm."))
        loaded = {"to_qkv.weight": False, "to_qkv.bias": False, "proj.weight": False, "proj.bias": False}
        for state_dict in state_dicts:
            for name in loaded:
                if not loaded[name] and state_dict.has(name):
                    param = {
                        "to_qkv.weight": self.to_qkv.weight,
                        "to_qkv.bias": self.to_qkv.bias,
                        "proj.weight": self.proj.weight,
                        "proj.bias": self.proj.bias,
                    }[name]
                    param.data.copy_(state_dict.get_tensor(name))
                    loaded[name] = True
        for name, is_loaded in loaded.items():
            if not is_loaded:
                raise RuntimeError(f"{name} is not loaded for WanAttentionBlock")


class WanMidBlock(nn.Module):
    """resnet, then (attention, resnet) x num_layers."""

    def __init__(
        self, dim: int, vae_group: str | None, dtype: torch.dtype, dropout: float = 0.0, num_layers: int = 1
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([WanResidualBlock(dim, dim, vae_group, dtype, dropout)])
        self.attentions = nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(WanAttentionBlock(dim, vae_group))
            self.resnets.append(WanResidualBlock(dim, dim, vae_group, dtype, dropout))

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        x = self.resnets[0](x, feat_cache, feat_idx)
        for i, attn in enumerate(self.attentions):
            x = attn(x)
            x = self.resnets[i + 1](x, feat_cache, feat_idx)
        return x

    def load_weights(self, state_dicts: list) -> None:
        for i, resnet in enumerate(self.resnets):
            resnet.load_weights(prefixed(state_dicts, f"resnets.{i}."))
        for i, attn in enumerate(self.attentions):
            attn.load_weights(prefixed(state_dicts, f"attentions.{i}."))


class WanResidualDownBlock(nn.Module):
    """Residual-VAE down block: resnets followed by an optional downsampler."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        vae_group: str | None,
        dtype: torch.dtype,
        dropout: float,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_res_blocks):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, vae_group, dtype, dropout))
            current_dim = out_dim
        self.downsampler: WanResample | None = None
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode, vae_group=vae_group, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)
        return x

    def load_weights(self, state_dicts: list) -> None:
        for i, resnet in enumerate(self.resnets):
            resnet.load_weights(prefixed(state_dicts, f"resnets.{i}."))
        if self.downsampler is not None:
            self.downsampler.load_weights(prefixed(state_dicts, "downsampler."))


class WanVAEEncoder3D(nn.Module):
    """Wan VAE encoder (causal, chunked via the feature cache)."""

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        z_dim: int = 8,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
        is_residual: bool = False,
        vae_group: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        attn_scales = attn_scales or []
        temperal_downsample = temperal_downsample or [True, True, False]

        dims = [dim] + [dim * u for u in dim_mult]
        scale = 1.0
        self.conv_in = WanCausalConv3D(in_channels, dims[0], (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group, dtype)
        self.down_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        vae_group,
                        dtype,
                        dropout,
                        num_res_blocks,
                        temperal_downsample[i] if i != len(dim_mult) - 1 else False,
                        i != len(dim_mult) - 1,
                    )
                )
            else:
                current_dim = in_dim
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(current_dim, out_dim, vae_group, dtype, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim, vae_group))
                    current_dim = out_dim
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode, vae_group=vae_group, dtype=dtype))
                    scale /= 2.0
        self.mid_block = WanMidBlock(dims[-1], vae_group, dtype, dropout, 1)
        self.norm_out = WanRMSNorm(dims[-1], channel_first=True, images=False)
        self.conv_out = WanCausalConv3D(dims[-1], z_dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group, dtype)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.size(2) < 2 and _has_cache(feat_cache[idx]):
                cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for block in self.down_blocks:
            if isinstance(block, (WanResidualDownBlock, WanResidualBlock)):
                x = block(x, feat_cache, feat_idx)
            elif isinstance(block, WanAttentionBlock):
                x = block(x)
            elif isinstance(block, WanResample):
                x = block(x, feat_cache, feat_idx)

        x = self.mid_block(x, feat_cache, feat_idx)
        x = self.norm_out(x)
        x = F.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.size(2) < 2 and _has_cache(feat_cache[idx]):
                cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x

    def load_weights(self, state_dicts: list) -> None:
        self.conv_in.load_weights(prefixed(state_dicts, "conv_in."))
        for i, block in enumerate(self.down_blocks):
            block.load_weights(prefixed(state_dicts, f"down_blocks.{i}."))
        self.mid_block.load_weights(prefixed(state_dicts, "mid_block."))
        self.norm_out.load_weights(prefixed(state_dicts, "norm_out."))
        self.conv_out.load_weights(prefixed(state_dicts, "conv_out."))


class WanResidualUpBlock(nn.Module):
    """Residual-VAE up block: resnets, upsampler and a DupUp3D shortcut."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        vae_group: str | None,
        dtype: torch.dtype,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ) -> None:
        super().__init__()
        self.avg_shortcut: DupUp3D | None = None
        if up_flag:
            factor_t = 2 if temperal_upsample else 1
            factor_s = 2
            self.avg_shortcut = DupUp3D(in_dim, out_dim, factor_t, factor_s)
        self.resnets = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, vae_group, dtype, dropout))
            current_dim = out_dim
        self.upsampler: WanResample | None = None
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(
                out_dim, upsample_mode, upsample_out_dim=out_dim, vae_group=vae_group, dtype=dtype
            )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
        first_chunk: bool,
    ) -> torch.Tensor:
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk)
        return x

    def load_weights(self, state_dicts: list) -> None:
        for i, resnet in enumerate(self.resnets):
            resnet.load_weights(prefixed(state_dicts, f"resnets.{i}."))
        if self.upsampler is not None:
            self.upsampler.load_weights(prefixed(state_dicts, "upsampler."))


class WanUpBlock(nn.Module):
    """Plain up block: resnets and an optional upsampler."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        vae_group: str | None,
        dtype: torch.dtype,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(WanResidualBlock(current_dim, out_dim, vae_group, dtype, dropout))
            current_dim = out_dim
        self.upsamplers = nn.ModuleList()
        if upsample_mode is not None:
            self.upsamplers.append(WanResample(out_dim, upsample_mode, vae_group=vae_group, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        for upsampler in self.upsamplers:
            x = upsampler(x, feat_cache, feat_idx)
        return x

    def load_weights(self, state_dicts: list) -> None:
        for i, resnet in enumerate(self.resnets):
            resnet.load_weights(prefixed(state_dicts, f"resnets.{i}."))
        for i, upsampler in enumerate(self.upsamplers):
            upsampler.load_weights(prefixed(state_dicts, f"upsamplers.{i}."))


class WanVAEDecoder3D(nn.Module):
    """Wan VAE decoder (frame-by-frame streaming via the feature cache)."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
        out_channels: int = 3,
        is_residual: bool = False,
        vae_group: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_upsample = temperal_upsample or [False, True, True]

        dims = [dim * dim_mult[-1]] + [dim * u for u in reversed(dim_mult)]
        self.conv_in = WanCausalConv3D(z_dim, dims[0], (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group, dtype)
        self.mid_block = WanMidBlock(dims[0], vae_group, dtype, dropout, 1)
        self.up_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            if i > 0 and not is_residual:
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            upsample_mode = None
            if up_flag:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
            if is_residual:
                self.up_blocks.append(
                    WanResidualUpBlock(
                        in_dim,
                        out_dim,
                        num_res_blocks,
                        vae_group,
                        dtype,
                        dropout,
                        temperal_upsample[i] if up_flag else False,
                        up_flag,
                    )
                )
            else:
                self.up_blocks.append(
                    WanUpBlock(
                        in_dim, out_dim, num_res_blocks, vae_group, dtype, dropout, upsample_mode if up_flag else None
                    )
                )
        self.norm_out = WanRMSNorm(dims[-1], channel_first=True, images=False)
        self.conv_out = WanCausalConv3D(dims[-1], out_channels, (3, 3, 3), (1, 1, 1), (1, 1, 1), vae_group, dtype)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor] | None,
        feat_idx: list[int] | None,
        first_chunk: bool,
    ) -> torch.Tensor:
        if feat_idx is None:
            feat_idx = [0]

        # conv_in
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.size(2) < 2 and _has_cache(feat_cache[idx]):
                cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        for block in self.up_blocks:
            if isinstance(block, WanResidualUpBlock):
                x = block(x, feat_cache, feat_idx, first_chunk)
            elif isinstance(block, WanUpBlock):
                x = block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = F.silu(x)

        # conv_out
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:].clone()
            if cache_x.size(2) < 2 and _has_cache(feat_cache[idx]):
                cache_x = torch.cat([feat_cache[idx][:, :, -1:].to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x

    def load_weights(self, state_dicts: list) -> None:
        self.conv_in.load_weights(prefixed(state_dicts, "conv_in."))
        self.mid_block.load_weights(prefixed(state_dicts, "mid_block."))
        for i, block in enumerate(self.up_blocks):
            block.load_weights(prefixed(state_dicts, f"up_blocks.{i}."))
        self.norm_out.load_weights(prefixed(state_dicts, "norm_out."))
        self.conv_out.load_weights(prefixed(state_dicts, "conv_out."))


class AutoencoderKLWan(DiTComponentBase, VaeParallelMixin):
    """Wan KL VAE with chunked causal encode and frame-streamed decode."""

    def __init__(
        self,
        config: dict,
        vae_group: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        DiTComponentBase.__init__(self)
        VaeParallelMixin.__init__(self, vae_group)
        self.z_dim = int(config.get("z_dim", 16))
        base_dim = int(config.get("base_dim", 96))
        num_res_blocks = int(config.get("num_res_blocks", 2))
        temperal_downsample = list(config.get("temperal_downsample", [True, True, False]))
        attn_scales = list(config.get("attn_scales", []))
        dim_mult = list(config.get("dim_mult", [1, 2, 4, 4]))
        dropout = float(config.get("dropout", 0.0))
        in_channels = int(config.get("in_channels", 3))
        out_channels = int(config.get("out_channels", 3))
        is_residual = bool(config.get("is_residual", False))
        self.vae_scale_factor_temporal = int(config.get("scale_factor_temporal", 4))
        self.vae_scale_factor_spatial = int(config.get("scale_factor_spatial", 8))
        self.latents_mean = list(config.get("latents_mean", []))
        self.latents_std = list(config.get("latents_std", []))
        self._dtype = dtype if dtype is not None else torch.float32

        self.encoder = WanVAEEncoder3D(
            in_channels,
            base_dim,
            self.z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            temperal_downsample,
            dropout,
            is_residual,
            vae_group,
            self._dtype,
        )
        decoder_temporal = list(reversed(temperal_downsample))
        self.decoder = WanVAEDecoder3D(
            base_dim,
            self.z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            decoder_temporal,
            dropout,
            out_channels,
            is_residual,
            vae_group,
            self._dtype,
        )
        self.quant_conv = WanCausalConv3D(
            2 * self.z_dim, 2 * self.z_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), vae_group, self._dtype
        )
        self.post_quant_conv = WanCausalConv3D(
            self.z_dim, self.z_dim, (1, 1, 1), (1, 1, 1), (0, 0, 0), vae_group, self._dtype
        )

        self._conv_counts = {
            "decoder": self._count_causal_convs(self.decoder),
            "encoder": self._count_causal_convs(self.encoder),
        }
        self._init_cache()

    @staticmethod
    def _count_causal_convs(module: nn.Module) -> int:
        return sum(1 for m in module.modules() if isinstance(m, WanCausalConv3D))

    def _init_cache(self) -> None:
        self.feat_map: list[torch.Tensor | None] = [None] * self._conv_counts["decoder"]
        self.conv_idx = [0]
        self.enc_feat_map: list[torch.Tensor | None] = [None] * self._conv_counts["encoder"]
        self.enc_conv_idx = [0]

    def clear_cache(self) -> None:
        self._init_cache()

    # ---- encode -------------------------------------------------------------

    def encode_(self, videos: torch.Tensor) -> torch.Tensor:
        dtype = self.quant_conv.conv.weight.dtype
        device = self.quant_conv.conv.weight.device
        x = videos.to(device, dtype)
        x = self.split(x)

        num_frame = x.size(2)
        iter_count = 1 + (num_frame - 1) // 4
        self.clear_cache()

        enc_outputs: list[torch.Tensor] = []
        for i in range(iter_count):
            self.enc_conv_idx = [0]
            if i == 0:
                x_slice = x[:, :, 0:1]
            else:
                start = 1 + 4 * (i - 1)
                end = min(1 + 4 * i, num_frame)
                x_slice = x[:, :, start:end]
            enc_outputs.append(self.encoder(x_slice, self.enc_feat_map, self.enc_conv_idx))
        out = torch.cat(enc_outputs, dim=2)
        out = self.quant_conv(out)
        # Merge the latent back to global W before the DiT transformer.
        out = self.merge(out)
        self.clear_cache()
        return out

    def encode(self, videos: torch.Tensor) -> DiagonalGaussianDistribution:
        return DiagonalGaussianDistribution(self.encode_(videos))

    # ---- decode -------------------------------------------------------------

    def decode_(self, latents: torch.Tensor) -> torch.Tensor:
        dtype = self.post_quant_conv.conv.weight.dtype
        device = self.post_quant_conv.conv.weight.device
        processed = latents.to(device, dtype)
        processed = self.split(processed)

        num_frame = processed.size(2)
        self.clear_cache()
        processed = self.post_quant_conv(processed)
        dec_outputs: list[torch.Tensor] = []
        for i in range(num_frame):
            self.conv_idx = [0]
            x_slice = processed[:, :, i : i + 1]
            dec_outputs.append(self.decoder(x_slice, self.feat_map, self.conv_idx, first_chunk=(i == 0)))
        out = torch.cat(dec_outputs, dim=2)
        out = self.merge(out)
        out = out.clamp(-1.0, 1.0)
        self.clear_cache()
        return out

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode_(latents)

    def forward(self, sample: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        posterior = self.encode(sample)
        x = posterior.sample(42) if sample_posterior else posterior.mode()
        return self.decode(x)

    # ---- weights ------------------------------------------------------------

    def load_weights(self, state_dicts: list) -> None:
        self.encoder.load_weights(prefixed(state_dicts, "encoder."))
        self.decoder.load_weights(prefixed(state_dicts, "decoder."))
        self.quant_conv.load_weights(prefixed(state_dicts, "quant_conv."))
        self.post_quant_conv.load_weights(prefixed(state_dicts, "post_quant_conv."))
