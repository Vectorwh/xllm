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

"""Optional GLM MTP checkpoint parameters, using the real loader on CPU."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xllm.python.models import glm5_2_mtp
from xllm.python.models.glm5_2 import Glm52Config, Glm52ForCausalLM


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


@pytest.mark.parametrize("has_rot", [False, True])
def test_optional_rot_is_created_before_loader_snapshot(monkeypatch: pytest.MonkeyPatch, has_rot: bool) -> None:
    monkeypatch.setattr(glm5_2_mtp, "glm52_eplv2_comm_policy", lambda cfg: None)
    monkeypatch.setattr(glm5_2_mtp, "Glm52DecoderLayer", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(glm5_2_mtp, "Glm52YarnRotaryEmbedding", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(Glm52ForCausalLM, "load_weights", lambda *args, **kwargs: None)
    cfg = Glm52Config.from_dict(
        {"hidden_size": 8, "n_layers": 1, "first_k_dense_replace": 0, "indexer_types": ["full"]}
    )
    model = glm5_2_mtp.Glm52MtpForCausalLM.__new__(glm5_2_mtp.Glm52MtpForCausalLM)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.dtype = torch.float32
    model.device = torch.device("cpu")
    model.model = glm5_2_mtp.Glm52MtpModel(cfg, model.dtype, model.device)
    assert model.model.rot is None

    tensors = {
        "model.eh_proj.weight": torch.ones_like(model.model.eh_proj.weight),
        "model.enorm.weight": torch.ones_like(model.model.enorm.weight),
        "model.hnorm.weight": torch.ones_like(model.model.hnorm.weight),
    }
    if has_rot:
        tensors["model.rot.weight"] = torch.full((8, 8), 2.0)
    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    assert model.model.enable_rot is has_rot
    parameters = dict(model.named_parameters())
    if has_rot:
        assert model.model.rot is not None
        torch.testing.assert_close(parameters["model.rot.weight"], tensors["model.rot.weight"])
    else:
        assert model.model.rot is None
        assert "model.rot.weight" not in parameters
