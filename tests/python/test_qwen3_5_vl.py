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

"""Backend-neutral tests for the Qwen3.5 multimodal model wiring."""

from __future__ import annotations

import pytest
import torch

from xllm.python.models.multimodal import MultimodalInputs
from xllm.python.models.qwen3_5_vl import (
    Qwen3_5ForConditionalGeneration,
    _vision_config_from_dict,
)
from xllm.python.registry import get_model_class


def test_registry_routes_vlm_types_to_the_multimodal_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The VLM model types must not fall back to the text-only class."""
    from xllm.python import registry

    # MODEL_PLATFORM_SUPPORT has no "cpu" entry for the VLM, so the suite's
    # host platform would otherwise raise NotImplementedError.
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    expected = {
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
        "qwen3_5_moe": "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration": "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration": "Qwen3_5ForConditionalGeneration",
        "qwen3_5_text": "Qwen3_5ForCausalLM",
        "qwen3_5_moe_text": "Qwen3_5ForCausalLM",
    }
    for name, class_name in expected.items():
        model_class = get_model_class(name)
        assert model_class.__name__ == class_name, f"{name} resolved to {model_class.__name__}"
    assert get_model_class("qwen3_5").__module__ == "xllm.python.models.qwen3_5_vl"


def _vision_config(**overrides: object) -> dict:
    """Flat PyCausalLM vision args for a small, fast tower."""
    config = {
        "mm_hidden_size": 32,
        "mm_num_hidden_layers": 2,
        "mm_num_attention_heads": 4,
        "mm_intermediate_size": 64,
        "mm_projection_dim": 48,
        "mm_num_position_embeddings": 64,
        "mm_patch_size": 2,
        "mm_temporal_patch_size": 1,
        "mm_spatial_merge_size": 2,
        "mm_deepstack_visual_indexes": [],
    }
    config.update(overrides)
    return config


def test_vision_config_reads_flat_mm_keys() -> None:
    cfg = _vision_config_from_dict(
        {
            "hidden_size": 5120,
            "mm_hidden_size": 96,
            "mm_num_hidden_layers": 3,
            "mm_num_attention_heads": 6,
            "mm_intermediate_size": 192,
            "mm_projection_dim": 80,
            "mm_num_position_embeddings": 48,
            "mm_patch_size": 4,
            "mm_temporal_patch_size": 2,
            "mm_spatial_merge_size": 4,
            "mm_deepstack_visual_indexes": [],
        }
    )
    assert (cfg.hidden_size, cfg.depth, cfg.num_heads) == (96, 3, 6)
    assert cfg.intermediate_size == 192
    assert (cfg.out_hidden_size, cfg.num_position_embeddings) == (80, 48)
    assert (cfg.patch_size, cfg.temporal_patch_size, cfg.spatial_merge_size) == (4, 2, 4)
    assert cfg.deepstack_visual_indexes == []


def test_vision_config_rejects_deepstack() -> None:
    with pytest.raises(ValueError, match="DeepStack"):
        _vision_config_from_dict(_vision_config(mm_deepstack_visual_indexes=[8, 16, 24]))


def test_vision_config_treats_an_absent_deepstack_key_as_empty() -> None:
    """A checkpoint with no DeepStack field must load, like the native loader.

    Qwen3VLVisionConfig.from_dict would otherwise substitute Qwen3-VL's
    [8, 16, 24] and reject the tower.
    """
    for config in ({"mm_hidden_size": 1152}, {"deepstack_visual_indexes": None}):
        assert _vision_config_from_dict(config).deepstack_visual_indexes == []
    # An explicit non-empty list is still rejected.
    with pytest.raises(ValueError, match="DeepStack"):
        _vision_config_from_dict({"mm_hidden_size": 1152, "mm_deepstack_visual_indexes": [8]})


def _cpu_vision_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU stand-ins for the two NPU vision ops the tower calls."""
    from xllm.python import kernels
    from xllm.python.models import qwen3_vl

    def vision_rotary_mul(value: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor) -> torch.Tensor:
        cos = cos_full.squeeze(0).squeeze(1).unsqueeze(1).to(value.dtype)
        sin = sin_full.squeeze(0).squeeze(1).unsqueeze(1).to(value.dtype)
        half = value.shape[-1] // 2
        first, second = value[..., :half], value[..., half:]
        rotated = torch.cat([-second, first], dim=-1)
        return value * cos + rotated * sin

    def vision_fusion_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        actual_seq_qlen: list[int],
        actual_seq_kvlen: list[int],
        num_heads: int,
        scale: float,
        input_layout: str,
    ) -> torch.Tensor:
        del actual_seq_kvlen, input_layout
        outputs = []
        start = 0
        for end in actual_seq_qlen:
            qs, ks, vs = q[start:end], k[start:end], v[start:end]
            scores = torch.einsum("thd,shd->hts", qs, ks) * scale
            weights = torch.softmax(scores, dim=-1)
            outputs.append(torch.einsum("hts,shd->thd", weights, vs))
            start = end
        return torch.cat(outputs, dim=0)

    monkeypatch.setattr(qwen3_vl.kernels, "vision_rotary_mul", vision_rotary_mul, raising=False)
    monkeypatch.setattr(qwen3_vl.kernels, "vision_fusion_attention", vision_fusion_attention, raising=False)
    monkeypatch.setattr(kernels, "vision_rotary_mul", vision_rotary_mul, raising=False)
    monkeypatch.setattr(kernels, "vision_fusion_attention", vision_fusion_attention, raising=False)


def test_vision_tower_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from xllm.python.models.qwen3_vl import Qwen3VLVisionTransformer

    _cpu_vision_kernels(monkeypatch)
    cfg = _vision_config_from_dict(_vision_config())
    tower = Qwen3VLVisionTransformer(cfg, dtype=torch.float32, device=torch.device("cpu"))

    grid_thw = torch.tensor([[1, 4, 4], [2, 2, 2]], dtype=torch.int32)
    num_patches = int((grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0]).sum())
    pixel_values = torch.randn(num_patches, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2)

    embeds = tower(pixel_values, grid_thw)

    # No deepstack: one projection per merged token, merge area = 2 * 2.
    num_tokens = int((grid_thw.prod(-1) / cfg.spatial_merge_size**2).sum())
    assert embeds.shape == (num_tokens, cfg.out_hidden_size)


def _bare_model() -> Qwen3_5ForConditionalGeneration:
    import torch.nn as nn

    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(16, 4)
    return model


def test_prepare_multimodal_inputs_scatters_image_rows() -> None:
    model = _bare_model()
    # Deterministic text rows so the scatter is unambiguous: any position still
    # holding a text embedding is zero, any scattered row is 2.0.
    with torch.no_grad():
        model.model.embed_tokens.weight.zero_()
    input_ids = torch.tensor([1, 5, 5, 7])
    mask = torch.tensor([False, True, True, False])
    image_embeds = torch.full((2, 4), 2.0)
    data = {"image|embedding": image_embeds, "image|mask": mask}

    inputs = model.prepare_multimodal_inputs(input_ids, data)

    assert isinstance(inputs, MultimodalInputs)
    torch.testing.assert_close(inputs.inputs_embeds[1], torch.full((4,), 2.0))
    torch.testing.assert_close(inputs.inputs_embeds[2], torch.full((4,), 2.0))
    torch.testing.assert_close(inputs.inputs_embeds[0], model.model.embed_tokens.weight[torch.tensor(1)])
    torch.testing.assert_close(inputs.inputs_embeds[3], torch.zeros(4))
    assert inputs.deepstack_input_embeds is None


def test_prepare_multimodal_inputs_without_modalities_returns_plain_embeds() -> None:
    model = _bare_model()
    input_ids = torch.tensor([1, 2])

    inputs = model.prepare_multimodal_inputs(input_ids, {})

    torch.testing.assert_close(inputs.inputs_embeds, model.model.embed_tokens.weight[input_ids])
    assert inputs.deepstack_input_embeds is None


def test_prepare_multimodal_inputs_rejects_a_mask_without_embeddings() -> None:
    """A dropped encoder output must not degrade into text-only generation."""
    model = _bare_model()

    with pytest.raises(ValueError, match=r"masks without embeddings \['image'\]"):
        model.prepare_multimodal_inputs(torch.tensor([1, 2]), {"image|mask": torch.tensor([True, False])})


def test_prepare_multimodal_inputs_rejects_embeddings_without_a_mask() -> None:
    """A dropped mask must not surface as a bare KeyError inside merge."""
    model = _bare_model()

    with pytest.raises(ValueError, match=r"embeddings without masks \['image'\]"):
        model.prepare_multimodal_inputs(torch.tensor([1, 2]), {"image|embedding": torch.zeros(2, 4)})


def test_prepare_multimodal_inputs_rejects_mask_row_mismatch() -> None:
    model = _bare_model()
    input_ids = torch.tensor([1, 5, 5, 7])
    # Two scheduled rows but three feature rows.
    data = {
        "image|embedding": torch.zeros(3, 4),
        "image|mask": torch.tensor([False, True, True, False]),
    }

    with pytest.raises(ValueError, match="embedding rows do not match the scheduled mask"):
        model.prepare_multimodal_inputs(input_ids, data)


def test_encode_multimodal_skips_modalities_without_counts() -> None:
    model = _bare_model()

    assert model.encode_multimodal({}, {"image": []}) == {}


def test_encode_multimodal_splits_image_features(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _bare_model()

    def encode(pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        del pixel_values, grid_thw
        return torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)

    monkeypatch.setattr(model, "encode", encode, raising=False)

    outputs = model.encode_multimodal(
        {"pixel_values": torch.zeros(6, 8), "image_grid_thw": torch.ones(1, 3, dtype=torch.int32)},
        {"image": [2, 4]},
    )

    parts = outputs["image|embedding"]
    expected = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    torch.testing.assert_close(parts[0], expected[:2])
    torch.testing.assert_close(parts[1], expected[2:])


def test_encode_batched_vision_applies_transform_before_splitting() -> None:
    from xllm.python.models.multimodal import encode_batched_vision

    features = torch.tensor([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [8.0, 80.0]])

    def encode(pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        del pixel_values, grid_thw
        return features

    outputs = encode_batched_vision(
        encode,
        {"pixel_values": torch.zeros(4, 8), "image_grid_thw": torch.ones(1, 3, dtype=torch.int32)},
        {"image": [1, 3]},
        transform=lambda values: values - values.mean(dim=0, keepdim=True),
    )

    parts = outputs["image|embedding"]
    expected = features - features.mean(dim=0, keepdim=True)
    torch.testing.assert_close(parts[0], expected[:1])
    torch.testing.assert_close(parts[1], expected[1:])


def test_encode_multimodal_requires_tensor_inputs() -> None:
    model = _bare_model()

    with pytest.raises(TypeError, match="must be tensors"):
        model.encode_multimodal({"pixel_values": [], "image_grid_thw": torch.ones(1, 3)}, {"image": [1]})


class _RecordingLoader:
    """Minimal ScopedWeightLoader stand-in that reports which names it holds."""

    def __init__(self, present: set[str]) -> None:
        self.present = present
        self.loaded: list[str] = []

    def has(self, name: str) -> bool:
        return name in self.present

    def load_tensor(self, param: torch.Tensor, local_name: str) -> None:
        self.loaded.append(local_name)


def _vision_model(depth: int = 1) -> Qwen3_5ForConditionalGeneration:
    import torch.nn as nn

    from xllm.python.models.qwen3_vl import Qwen3VLVisionTransformer

    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    cfg = _vision_config_from_dict(_vision_config(mm_num_hidden_layers=depth))
    model.vision_cfg = cfg
    model.vision_model = Qwen3VLVisionTransformer(cfg, dtype=torch.float32, device=torch.device("cpu"))
    return model


def _checkpoint_names(model: Qwen3_5ForConditionalGeneration) -> set[str]:
    """Checkpoint-side names for a tower, with the Conv3d ".proj" alias applied."""
    names = set(dict(model.vision_model.named_parameters()))
    names -= {"patch_embed.weight", "patch_embed.bias"}
    return names | {"patch_embed.proj.weight", "patch_embed.proj.bias"}


@pytest.mark.parametrize(
    ("missing_tensor", "parameter_name"),
    [
        ("blocks.0.mlp.linear_fc2.bias", "blocks.0.mlp.linear_fc2.bias"),
        ("patch_embed.proj.bias", "patch_embed.bias"),
    ],
)
def test_load_vision_weights_names_the_missing_checkpoint_tensor(missing_tensor: str, parameter_name: str) -> None:
    model = _vision_model()
    loader = _RecordingLoader(_checkpoint_names(model) - {missing_tensor})

    with pytest.raises(KeyError) as error:
        model._load_vision_weights(loader)

    assert error.value.args[0] == (
        f"checkpoint is missing vision tensor {missing_tensor!r} required by parameter {parameter_name!r}"
    )


def test_load_vision_weights_loads_every_tower_tensor() -> None:
    model = _vision_model(depth=2)
    expected = _checkpoint_names(model)
    loader = _RecordingLoader(expected)

    model._load_vision_weights(loader)

    assert sorted(loader.loaded) == sorted(expected)


def test_text_config_follows_the_shared_execution_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The VLM must resolve SP through the text host's owner function.

    The metadata builders read `sequence_parallel_enabled` back out of the same
    config dict, so bypassing prepare_execution_config would silently diverge
    from qwen3_5_text.
    """
    from xllm.python.models import qwen3_5, qwen3_5_vl

    class _DecoderLayer(torch.nn.Module):
        supports_sequence_parallel = True

    # qwen3_5.py binds the factory symbol at import time.
    monkeypatch.setattr(qwen3_5, "get_qwen3_5_decoder_layer_class", lambda device: _DecoderLayer)
    # The tower and the LLM are irrelevant here; only the config contract is.
    monkeypatch.setattr(qwen3_5_vl, "Qwen3_5Model", lambda *args, **kwargs: torch.nn.Module())
    monkeypatch.setattr(qwen3_5_vl, "Qwen3VLVisionTransformer", lambda *args, **kwargs: torch.nn.Module())

    seen: dict[str, object] = {}
    original = qwen3_5.Qwen3_5Config.prepare_execution_config

    def spy(cls, config: dict, device: torch.device, **kwargs: object):
        cfg = original.__func__(cls, config, device, **kwargs)
        seen["device"] = device
        seen["resolved"] = config.get("sequence_parallel_enabled")
        return cfg

    monkeypatch.setattr(qwen3_5.Qwen3_5Config, "prepare_execution_config", classmethod(spy))

    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    config = _vision_config() | {
        "hidden_size": 32,
        "n_layers": 4,
        "n_heads": 4,
        "n_kv_heads": 2,
        "vocab_size": 64,
        "device": "cpu",
        # DECODE + tp>1 is what switches sequence parallelism on.
        "instance_role": "DECODE",
        "tp_size": 2,
        "world_size": 2,
    }
    Qwen3_5ForConditionalGeneration.__init__(model, config)

    assert seen["device"] == torch.device("cpu")
    assert seen["resolved"] is True
    # Written back into the shared dict, which is what the builders consume.
    assert config["sequence_parallel_enabled"] is True
    assert model.text_cfg.sequence_parallel_enabled is True


def test_load_weights_uses_the_config_driven_parallel_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """EP/MoE-TP topology must come from the config, not the dataclass defaults."""
    from xllm.python.model_loader import ParallelLoadContext
    from xllm.python.models import qwen3_5_vl

    captured: dict[str, object] = {}

    def fake_load(
        model: object, lm_head: object, weights: object, context: ParallelLoadContext, **kwargs: object
    ) -> None:
        captured["context"] = context
        captured["kwargs"] = kwargs

    monkeypatch.setattr(qwen3_5_vl, "load_causal_lm_weights", fake_load)

    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()  # type: ignore[assignment]
    model._load_vision_weights = lambda weights: None  # type: ignore[method-assign]
    model.text_cfg = type(
        "Cfg",
        (),
        {
            "tp_size": 2,
            "dp_rank": 0,
            "dp_size": 1,
            "moe_tp_rank": 0,
            "moe_tp_size": 2,
            "ep_rank": 1,
            "ep_size": 2,
            "tie_word_embeddings": False,
        },
    )()
    model.lm_head = type("Head", (), {"weight": torch.zeros(1)})()

    model.load_weights([], tp_rank=0, tp_size=2)

    context = captured["context"]
    assert isinstance(context, ParallelLoadContext)
    assert (context.tp_rank, context.tp_size) == (0, 2)
    assert (context.moe_tp_size, context.ep_size) == (2, 2)
    assert (context.ep_rank, context.moe_tp_rank) == (1, 0)
    # An untied checkpoint without lm_head.weight falls back to embed_tokens,
    # matching the qwen3_5 text host; without this the load raises KeyError.
    assert captured["kwargs"]["embed_fallback"] is True


def _load_weights_model(text_tp_size: int) -> Qwen3_5ForConditionalGeneration:
    """A host whose parameters were shaped from ``text_tp_size``."""
    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.text_cfg = type("Cfg", (), {"tp_size": text_tp_size})()
    model._load_vision_weights = lambda weights: None  # type: ignore[method-assign]
    return model


def test_load_weights_rejects_a_tp_size_that_disagrees_with_the_config() -> None:
    """The caller's tp_size shards the weights, so it must match the built shapes."""
    model = _load_weights_model(text_tp_size=2)

    with pytest.raises(ValueError, match=r"tp_size 4 disagrees with the model's tp_size 2"):
        model.load_weights([], tp_rank=0, tp_size=4)


def test_load_weights_rejects_a_tp_rank_outside_the_world() -> None:
    model = _load_weights_model(text_tp_size=2)

    with pytest.raises(ValueError, match=r"tp_rank must be in \[0, 2\), got 2"):
        model.load_weights([], tp_rank=2, tp_size=2)
