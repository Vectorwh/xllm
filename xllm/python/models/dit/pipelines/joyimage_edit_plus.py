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

"""JoyImageEditPlus pipeline.

Port of ``xllm/models/dit/pipelines/pipeline_joyimage_edit_plus.h``. The
denoising step is wrapped with :class:`CFGParallelMixin` exactly like the C++
``exec_with_cfg``; the text encoder is reached through the encoder contract —
a form-B C++ proxy exposes ``encode_prompts`` and keeps tokenization on the
C++ side, so a request crosses the language boundary once for encoding and
once for the pipeline call itself.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Any

import torch

from xllm.python.models.dit.pipelines.base import DiffusionPipeline
from xllm.python.parallel import CFGParallelMixin

# Nearest 1024-base aspect buckets (h, w).
_BUCKETS: tuple[tuple[int, int], ...] = (
    (512, 1792),
    (512, 1856),
    (512, 1920),
    (512, 1984),
    (512, 2048),
    (576, 1600),
    (576, 1664),
    (576, 1728),
    (576, 1792),
    (640, 1472),
    (640, 1536),
    (640, 1600),
    (704, 1344),
    (704, 1408),
    (704, 1472),
    (768, 1216),
    (768, 1280),
    (768, 1344),
    (832, 1152),
    (832, 1216),
    (896, 1088),
    (896, 1152),
    (960, 1024),
    (960, 1088),
    (1024, 960),
    (1024, 1024),
    (1088, 896),
    (1088, 960),
    (1152, 832),
    (1152, 896),
    (1216, 768),
    (1216, 832),
    (1280, 768),
    (1344, 704),
    (1344, 768),
    (1408, 704),
    (1472, 640),
    (1472, 704),
    (1536, 640),
    (1600, 576),
    (1600, 640),
    (1664, 576),
    (1728, 576),
    (1792, 512),
    (1792, 576),
    (1856, 512),
    (1920, 512),
    (1984, 512),
    (2048, 512),
)


def joyimage_bucket(height: int, width: int) -> tuple[int, int]:
    target = height / width
    best = (1024, 1024)
    best_diff = math.inf
    for h, w in _BUCKETS:
        diff = abs(h / w - target)
        if diff < best_diff:
            best_diff = diff
            best = (h, w)
    return best


class JoyImageEditPlusPipeline(DiffusionPipeline, CFGParallelMixin):
    """Image-edit pipeline: Qwen3-VL text encoding + Wan VAE + double-stream DiT."""

    def __init__(self, components: dict, config: dict) -> None:
        DiffusionPipeline.__init__(self)
        self.register_modules(**components)

        required = ("transformer", "vae", "scheduler", "vae_image_processor")
        missing = [name for name in required if name not in self.components]
        if missing:
            raise KeyError(f"JoyImageEditPlusPipeline missing components: {missing}")
        self.text_encoder = self.components.get("text_encoder")

        CFGParallelMixin.__init__(self, cfg_group=config.get("dit_cfg_group"))

        vae_config = config["vae_config"]
        self.latent_channels = int(vae_config.get("z_dim", 16))
        self.vae_scale_factor_spatial = int(vae_config.get("scale_factor_spatial", 8) or 8)
        self.latents_mean = list(vae_config.get("latents_mean", []))
        self.latents_std = list(vae_config.get("latents_std", []))

        transformer_config = config["transformer_config"]
        self.in_channels = int(transformer_config.get("in_channels", 16))
        self.patch_t, self.patch_h, self.patch_w = (int(v) for v in transformer_config.get("patch_size", [1, 2, 2]))
        hidden_size = int(transformer_config.get("hidden_size", 4096))
        num_heads = int(transformer_config.get("num_attention_heads", 32))
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = hidden_size // num_heads
        self.rope_dim_list = list(transformer_config.get("rope_dim_list", []))
        self.rope_theta = float(transformer_config.get("theta", 10000))

        # nn.Module already exposes a `device` property, so the configured
        # target device/dtype live under underscore names. The executor has
        # already resolved `dtype` into a torch.dtype.
        self._device = torch.device(config["device"])
        self._dtype = config["dtype"]

        # Text-encoder result cache. encode_prompts (Qwen3-VL vision tower +
        # LLM prefill) dominates end-to-end latency; the same (prompt, images)
        # pair yields identical prompt embeddings, so cache the result keyed by
        # prompt text + image bytes. The values live on device, so the cache is
        # a bounded LRU: without eviction, online traffic grows NPU memory
        # monotonically. Capacity is configurable via
        # ``text_embed_cache_size`` (0 disables caching).
        self._text_embed_cache: OrderedDict[bytes, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._text_embed_cache_maxsize = int(config.get("text_embed_cache_size", 64))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # ---- entry point ---------------------------------------------------------

    def __call__(self, request: dict, cache_hooks: Any = None) -> list[torch.Tensor]:
        gp = request["generation_params"]
        num_inference_steps = int(gp["num_inference_steps"])
        guidance_scale = float(gp["true_cfg_scale"]) if float(gp["true_cfg_scale"]) > 0 else float(gp["guidance_scale"])
        seed = int(gp["seed"]) if int(gp["seed"]) >= 0 else 42

        raw_images = self._collect_reference_images(request)
        batch_size = self._resolve_batch_size(request, raw_images)

        height, width = self._resolve_output_resolution(gp, raw_images)
        height = (height // self.vae_scale_factor_spatial) * self.vae_scale_factor_spatial
        width = (width // self.vae_scale_factor_spatial) * self.vae_scale_factor_spatial

        vae_refs = self._preprocess_vae_references(raw_images, batch_size)

        do_cfg = guidance_scale > 1.0
        prompt_embeds, prompt_embeds_mask = self._resolve_prompt_embeds(request, raw_images, batch_size)
        if do_cfg:
            neg_embeds, neg_embeds_mask = self._resolve_negative_embeds(request, raw_images, batch_size)
            max_l = max(prompt_embeds.size(1), neg_embeds.size(1))
            prompt_embeds = self._pad_seq(prompt_embeds, max_l)
            neg_embeds = self._pad_seq(neg_embeds, max_l)
            prompt_embeds_mask = self._pad_seq(prompt_embeds_mask, max_l)
            neg_embeds_mask = self._pad_seq(neg_embeds_mask, max_l)

        latents, shape_list = self._prepare_latents(
            batch_size,
            self.in_channels,
            height,
            width,
            seed,
            vae_refs,
            request.get("latents"),
        )
        clean_backup = latents.clone()
        target_shape = shape_list[0][0]
        target_patch_count = target_shape[0] * target_shape[1] * target_shape[2]

        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps, self.device)
        scheduler.set_begin_index(0)
        timesteps = scheduler.timesteps

        if cache_hooks is not None:
            cache_hooks.set_context(num_inference_steps, len(self.transformer.double_blocks))

        prompt_context = self._prepare_transformer_context(
            batch_size, latents.size(1), self.device, prompt_embeds_mask, shape_list
        )
        negative_context = None
        if do_cfg:
            negative_context = self._prepare_transformer_context(
                batch_size, latents.size(1), self.device, neg_embeds_mask, shape_list
            )

        for step_index, t in enumerate(timesteps.tolist()):
            t_expand = torch.full((batch_size,), float(t), dtype=timesteps.dtype, device=timesteps.device)
            if do_cfg:
                # Default args bind the per-step loop values at definition time.
                cond, uncond = self.exec_with_cfg(
                    lambda is_positive, latents=latents, t_expand=t_expand, step_index=step_index: self.transformer(
                        latents,
                        t_expand,
                        prompt_embeds if is_positive else neg_embeds,
                        prompt_context["rope_cos"] if is_positive else negative_context["rope_cos"],
                        prompt_context["rope_sin"] if is_positive else negative_context["rope_sin"],
                        prompt_context["attention_mask"] if is_positive else negative_context["attention_mask"],
                        use_cfg=not is_positive,
                        step_index=step_index + 1,
                        cache_hooks=cache_hooks,
                    )
                )
                comb = uncond + guidance_scale * (cond - uncond)
                # Norm-rescale (diffusers) over the channel dim (2) of the 6D
                # [B, N, C, pt, ph, pw] prediction.
                cond_norm = torch.linalg.vector_norm(cond, ord=2, dim=2, keepdim=True)
                comb_norm = torch.linalg.vector_norm(comb, ord=2, dim=2, keepdim=True)
                noise_pred = comb * (cond_norm / comb_norm.clamp_min(1e-6))
            else:
                noise_pred = self.transformer(
                    latents,
                    t_expand,
                    prompt_embeds,
                    prompt_context["rope_cos"],
                    prompt_context["rope_sin"],
                    prompt_context["attention_mask"],
                    use_cfg=False,
                    step_index=step_index + 1,
                    cache_hooks=cache_hooks,
                )

            latents = scheduler.step(noise_pred, timesteps[step_index], latents).to(latents.dtype)
            # Reference patches stay frozen across steps.
            latents[:, target_patch_count:] = clean_backup[:, target_patch_count:]

        decoded = self._decode(latents, shape_list, batch_size)
        return decoded

    # ---- text encoding -------------------------------------------------------

    def _resolve_prompt_embeds(
        self,
        request: dict,
        raw_images: list[torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds = request.get("prompt_embeds")
        if prompt_embeds is not None:
            embeds = prompt_embeds.to(self.device, self.dtype)
            mask = torch.ones(embeds.size(0), embeds.size(1), dtype=torch.long, device=self.device)
            return embeds, mask
        prompts = request.get("prompts") or []
        if not prompts:
            raise ValueError("JoyImageEditPlus requires `prompts` or `prompt_embeds`")
        return self._encode_prompts(prompts, raw_images, batch_size)

    def _resolve_negative_embeds(
        self,
        request: dict,
        raw_images: list[torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        negative_prompt_embeds = request.get("negative_prompt_embeds")
        if negative_prompt_embeds is not None:
            embeds = negative_prompt_embeds.to(self.device, self.dtype)
            mask = torch.ones(embeds.size(0), embeds.size(1), dtype=torch.long, device=self.device)
            return embeds, mask
        negative_prompts = list(request.get("negative_prompts") or [])
        if not negative_prompts:
            negative_prompts = [""] * batch_size
        return self._encode_prompts(negative_prompts, raw_images, batch_size)

    def _encode_prompts(
        self,
        prompts: list[str],
        raw_images: list[torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(prompts) != batch_size:
            raise ValueError("JoyImageEditPlus prompt batch size mismatch")
        if self.text_encoder is None:
            raise RuntimeError("Qwen3-VL text encoder is not loaded")

        # Cache key: prompt text + the bytes of every reference image. The
        # VLM forward (vision tower + LLM prefill) is the dominant latency
        # cost and is fully deterministic for a given (prompt, images) pair.
        key = self._text_cache_key(prompts, raw_images)
        cached = self._text_embed_cache.get(key)
        if cached is not None:
            self._text_embed_cache.move_to_end(key)
            return cached

        encode = getattr(self.text_encoder, "encode_prompts", None)
        if encode is None:
            raise NotImplementedError(
                "Only form-B (C++ proxy) text encoders expose encode_prompts; "
                "native Python encoders are not wired for JoyImageEditPlus yet"
            )
        # One cross-language call per request: the proxy owns tokenization,
        # multimodal preprocessing and the VLM forward.
        result = encode(prompts, raw_images)
        if self._text_embed_cache_maxsize > 0:
            self._text_embed_cache[key] = result
            self._text_embed_cache.move_to_end(key)
            while len(self._text_embed_cache) > self._text_embed_cache_maxsize:
                # Evicting drops the last reference to the device tensors.
                self._text_embed_cache.popitem(last=False)
        return result

    def _text_cache_key(self, prompts: list[str], raw_images: list[torch.Tensor]) -> tuple:
        hasher = hashlib.sha256()
        for prompt in prompts:
            hasher.update(prompt.encode("utf-8"))
            hasher.update(b"\x00")
        for img in raw_images:
            # Hashing is done on the CPU bytes of the decoded reference image,
            # normalized to fp32 so the key is stable regardless of the source
            # dtype/device of the incoming tensor.
            hasher.update(img.detach().float().cpu().contiguous().numpy().tobytes())
            hasher.update(b"\x00")
        return hasher.digest()

    # ---- latents -------------------------------------------------------------

    def _latents_mean(self, ref: torch.Tensor) -> torch.Tensor:
        return (
            torch.tensor(self.latents_mean, dtype=torch.float32)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(ref.device, ref.dtype)
        )

    def _latents_std(self, ref: torch.Tensor) -> torch.Tensor:
        return (
            torch.tensor(self.latents_std, dtype=torch.float32)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(ref.device, ref.dtype)
        )

    def _patchify(self, item: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """[C, T, H, W] -> [num_patches, C, pt, ph, pw] plus (lt, lh, lw)."""
        c, t, h, w = item.shape
        lt, lh, lw = t // self.patch_t, h // self.patch_h, w // self.patch_w
        p = item.reshape(c, lt, self.patch_t, lh, self.patch_h, lw, self.patch_w)
        p = p.permute(1, 3, 5, 0, 2, 4, 6).reshape(-1, c, self.patch_t, self.patch_h, self.patch_w)
        return p, (lt, lh, lw)

    def _prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        seed: int,
        reference_images: list[list[torch.Tensor]],
        provided_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[list[tuple[int, int, int]]]]:
        """Build the 6D padded latent: target noise + VAE-encoded references.

        Returns (padded_latents[B, N, C, pt, ph, pw], shape_list with
        per-sample (t, h, w) patch grids per item). The target (noise) item is
        always the first per sample, so denoising freezes every patch after
        the first item's patch count.
        """
        all_patches: list[torch.Tensor] = []
        all_shapes: list[list[tuple[int, int, int]]] = []
        max_patches = 0

        h_target = height // self.vae_scale_factor_spatial
        w_target = width // self.vae_scale_factor_spatial

        for b in range(batch_size):
            items: list[torch.Tensor] = []
            if provided_latents is not None:
                noise = provided_latents[b].to(self.device, self.dtype)
            else:
                noise = self._randn_tensor((num_channels_latents, 1, h_target, w_target), seed + b)
            items.append(noise)

            if b < len(reference_images):
                for ref_img in reference_images[b]:
                    ref = ref_img.to(self.device, self.dtype)
                    if ref.dim() == 4:
                        ref = ref.unsqueeze(2)  # [B,C,H,W] -> [B,C,1,H,W]
                    lat = self.vae.encode(ref).mode()
                    lat = (lat - self._latents_mean(lat)) / self._latents_std(lat)
                    items.append(lat.squeeze(0))  # [C, 1, h', w']

            sample_patches: list[torch.Tensor] = []
            sample_shapes: list[tuple[int, int, int]] = []
            for item in items:
                patches, shape = self._patchify(item)
                sample_shapes.append(shape)
                sample_patches.append(patches)
            combined = torch.cat(sample_patches, dim=0)
            all_patches.append(combined)
            all_shapes.append(sample_shapes)
            max_patches = max(max_patches, combined.size(0))

        padded = torch.zeros(
            (batch_size, max_patches, num_channels_latents, self.patch_t, self.patch_h, self.patch_w),
            device=self.device,
            dtype=self.dtype,
        )
        for b in range(batch_size):
            n = all_patches[b].size(0)
            padded[b, :n] = all_patches[b]
        return padded, all_shapes

    def _randn_tensor(self, shape: tuple[int, ...], seed: int) -> torch.Tensor:
        """CPU-generator randn in the pipeline dtype (matches the C++ helper)."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=self.dtype, device="cpu").to(self.device)

    # ---- transformer context -------------------------------------------------

    def _prepare_transformer_context(
        self,
        batch_size: int,
        image_sequence_length: int,
        device: torch.device,
        encoder_hidden_states_mask: torch.Tensor | None,
        shape_list: list[list[tuple[int, int, int]]],
    ) -> dict:
        if len(shape_list) != batch_size:
            raise ValueError("shape_list batch size must match transformer batch size")

        cos_list: list[torch.Tensor] = []
        sin_list: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            cos_parts: list[torch.Tensor] = []
            sin_parts: list[torch.Tensor] = []
            temporal_offset = 0
            for shape in shape_list[batch_index]:
                start = (temporal_offset, 0, 0)
                stop = (temporal_offset + shape[0], shape[1], shape[2])
                rope = self._rope_for_range(start, stop, device)
                cos_parts.append(rope[0])
                sin_parts.append(rope[1])
                temporal_offset += shape[0]

            cos = torch.cat(cos_parts, dim=0)
            sin = torch.cat(sin_parts, dim=0)
            actual_sequence_length = cos.size(0)
            if actual_sequence_length < image_sequence_length:
                cos = torch.nn.functional.pad(cos, (0, 0, 0, image_sequence_length - actual_sequence_length), value=1.0)
                sin = torch.nn.functional.pad(sin, (0, 0, 0, image_sequence_length - actual_sequence_length), value=0.0)
            cos_list.append(cos)
            sin_list.append(sin)

        rope_cos = torch.stack(cos_list, dim=0)
        rope_sin = torch.stack(sin_list, dim=0)

        attention_mask = None
        if encoder_hidden_states_mask is not None:
            image_mask = torch.zeros((batch_size, image_sequence_length), dtype=torch.bool, device=device)
            for batch_index in range(batch_size):
                actual_sequence_length = 0
                for shape in shape_list[batch_index]:
                    actual_sequence_length += shape[0] * shape[1] * shape[2]
                image_mask[batch_index, :actual_sequence_length] = True
            full_mask = torch.cat([image_mask, encoder_hidden_states_mask.to(torch.bool)], dim=1)
            # NPU flash attention rejects a key-only [B, 1, 1, S] keep-mask
            # (see aclnnFlashAttentionScore's AnalyzeOptionalInput); it only
            # accepts [B, N, Sq, Skv] / [B, 1, Sq, Skv] / [1, 1, Sq, Skv] /
            # [Sq, Skv]. Expand the key mask over the query axis to [B, 1, S, S]
            # so mask[b, 0, i, j] = full_mask[b, j] (identical semantics to the
            # broadcasting form SDPA would otherwise infer on CPU/CUDA).
            seq_len = full_mask.size(1)
            attention_mask = full_mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, seq_len, -1).contiguous()

        return {"rope_cos": rope_cos, "rope_sin": rope_sin, "attention_mask": attention_mask}

    def _rope_for_range(
        self,
        start: tuple[int, int, int],
        stop: tuple[int, int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dimensions = self.rope_dim_list
        if not dimensions:
            dimension = self.head_dim // 3
            dimensions = [dimension, dimension, dimension]

        float_options = {"dtype": torch.float32, "device": device}
        grids = [torch.arange(start[i], stop[i], **float_options) for i in range(3)]
        mesh = torch.meshgrid(grids[0], grids[1], grids[2], indexing="ij")

        cos_parts: list[torch.Tensor] = []
        sin_parts: list[torch.Tensor] = []
        for dimension_index in range(3):
            position = mesh[dimension_index].reshape(-1)
            dim = dimensions[dimension_index]
            index = torch.arange(0, dim, 2, **float_options)[: dim // 2]
            frequencies = 1.0 / (self.rope_theta ** (index / dim))
            angles = torch.outer(position, frequencies)
            cos_parts.append(angles.cos().repeat_interleave(2, dim=1))
            sin_parts.append(angles.sin().repeat_interleave(2, dim=1))
        return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)

    # ---- decoding ------------------------------------------------------------

    def _decode(
        self,
        latents: torch.Tensor,
        shape_list: list[list[tuple[int, int, int]]],
        batch_size: int,
    ) -> list[torch.Tensor]:
        images: list[torch.Tensor] = []
        for b in range(batch_size):
            lt, lh, lw = shape_list[b][0]
            target_len = lt * lh * lw
            patches = latents[b][:target_len]  # [len, C, pt, ph, pw]
            c = patches.size(1)
            vid = patches.reshape(lt, lh, lw, c, self.patch_t, self.patch_h, self.patch_w)
            vid = vid.permute(3, 0, 4, 1, 5, 2, 6).reshape(
                1, c, lt * self.patch_t, lh * self.patch_h, lw * self.patch_w
            )
            vid = vid * self._latents_std(vid) + self._latents_mean(vid)
            img = self.vae.decode(vid)  # [1, C, 1, H, W]
            img = img.to(torch.float32).squeeze(0).squeeze(1)  # [C, H, W]
            images.append(img.unsqueeze(0))
        image = torch.cat(images, dim=0)
        image = self.vae_image_processor.postprocess(image)
        return list(torch.chunk(image, batch_size, dim=0))

    # ---- request helpers -----------------------------------------------------

    @staticmethod
    def _collect_reference_images(request: dict) -> list[torch.Tensor]:
        images_list = [img for img in (request.get("images_list") or []) if img is not None]
        if images_list:
            return images_list
        images = request.get("images")
        if images is not None:
            return [images]
        raise ValueError("JoyImageEditPlus requires reference images")

    @staticmethod
    def _resolve_batch_size(request: dict, raw_images: list[torch.Tensor]) -> int:
        batch_size = int(request.get("batch_size") or 0)
        if batch_size > 0:
            return batch_size
        prompt_embeds = request.get("prompt_embeds")
        if prompt_embeds is not None:
            return prompt_embeds.size(0)
        prompts = request.get("prompts") or []
        if prompts:
            return len(prompts)
        return raw_images[0].size(0) if raw_images[0].dim() == 4 else 1

    def _resolve_output_resolution(self, gp: dict, raw_images: list[torch.Tensor]) -> tuple[int, int]:
        height = int(gp.get("height") or 0)
        width = int(gp.get("width") or 0)
        if height <= 0 or width <= 0:
            last = raw_images[-1]
            image_h, image_w = last.size(-2), last.size(-1)
            height, width = joyimage_bucket(image_h, image_w)
        return height, width

    def _preprocess_vae_references(
        self,
        raw_images: list[torch.Tensor],
        batch_size: int,
    ) -> list[list[torch.Tensor]]:
        vae_refs: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            for imgs in raw_images:
                img = imgs[b] if imgs.dim() == 4 else imgs  # [C, H, W]
                image_h, image_w = img.size(1), img.size(2)
                bucket_h, bucket_w = joyimage_bucket(image_h, image_w)
                # The processor resizes on CPU (PIL); prepare_latents moves the
                # encoded result to the device, so skip the round trip here.
                img4 = img.unsqueeze(0)
                vae_refs[b].append(self.vae_image_processor.preprocess(img4, bucket_h, bucket_w, resize_mode="lanczos"))
        return vae_refs

    @staticmethod
    def _pad_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
        cur = x.size(1)
        if cur >= target_len:
            return x[:, cur - target_len :]
        pad = target_len - cur
        zeros = torch.zeros(x.size(0), pad, *x.shape[2:], dtype=x.dtype, device=x.device)
        return torch.cat([x, zeros], dim=1)
