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

"""Keep the aux-capture eager gate limited to the DeepSeek V4.1 family."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PY_CAUSAL_LM = _REPO_ROOT / "xllm/models/llm/py_causal_lm.cpp"
_V41_FAMILY = frozenset({"deepseek_v41", "deepseek_v41_dspark"})


def _requires_eager(
    *,
    model_requires_eager: bool,
    layers_to_capture: bool,
    allow_graph_aux_hidden_states: bool,
    model_type: str,
) -> bool:
    return model_requires_eager or (
        layers_to_capture and model_type in _V41_FAMILY and not allow_graph_aux_hidden_states
    )


@pytest.mark.parametrize(
    "model_type,model_requires_eager,layers_to_capture,allow_graph_aux_hidden_states,expected",
    [
        pytest.param("deepseek_v41", False, True, False, True, id="v41-flag-off"),
        pytest.param("deepseek_v41_dspark", False, True, False, True, id="dspark-flag-off"),
        pytest.param("glm5_next", False, True, False, False, id="glm5-flag-off"),
        pytest.param("deepseek_v41", False, True, True, False, id="v41-flag-on"),
        pytest.param("deepseek_v41_dspark", False, True, True, False, id="dspark-flag-on"),
        pytest.param("glm5_next", False, True, True, False, id="glm5-flag-on"),
        pytest.param("deepseek_v41", True, False, True, True, id="model-eager-v41"),
        pytest.param("glm5_next", True, True, True, True, id="model-eager-glm5"),
    ],
)
def test_aux_capture_eager_policy_truth_table(
    model_type: str,
    model_requires_eager: bool,
    layers_to_capture: bool,
    allow_graph_aux_hidden_states: bool,
    expected: bool,
) -> None:
    assert (
        _requires_eager(
            model_requires_eager=model_requires_eager,
            layers_to_capture=layers_to_capture,
            allow_graph_aux_hidden_states=allow_graph_aux_hidden_states,
            model_type=model_type,
        )
        is expected
    )


def test_aux_capture_eager_gate_stays_scoped_to_the_v41_family() -> None:
    source = _PY_CAUSAL_LM.read_text()
    match = re.search(r"\bbool requires_eager_execution\s*=\s*(.*?);", source, re.DOTALL)
    assert match is not None
    predicate = re.sub(r"\s+", " ", match.group(1)).strip()

    # The model's own eager policy remains authoritative. Aux capture only
    # forces eager when layers are requested, the model is in the v41 family,
    # and graph aux capture has not been explicitly enabled.
    assert predicate == (
        "model_args_.requires_eager_execution() || "
        "(!model_args_.layers_to_capture().empty() && "
        "util::is_deepseek_v41_family_model_type(model_args_.model_type()) && "
        "!allow_graph_aux_hidden_states)"
    )
