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

"""Contracts for the supported Python Joyo pipeline entry point."""

from __future__ import annotations

import pytest

from xllm.python.model_executor.dit_executor import DiTPipelineExecutor


@pytest.mark.parametrize("policy", ["FBCache", "TaylorSeer", "FBCacheTaylorSeer", "ResidualCache"])
def test_rejects_unsupported_cache_policy_before_component_loading(policy: str) -> None:
    with pytest.raises(ValueError, match="does not support dit_cache_policy"):
        DiTPipelineExecutor({"dit_cache_policy": policy})


def test_execute_calls_pipeline_with_request_only() -> None:
    request = {"prompts": ["a test video"]}
    expected = {"tensors": []}
    seen = []

    def pipeline(value: dict) -> dict:
        seen.append(value)
        return expected

    executor = DiTPipelineExecutor.__new__(DiTPipelineExecutor)
    executor.pipeline = pipeline
    assert executor.execute(request) is expected
    assert seen == [request]
