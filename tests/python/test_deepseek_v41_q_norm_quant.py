# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""V4.1 indexer query consumes the normed q-lora latent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import xllm.python.models.deepseek_v41 as v41


def test_indexer_query_projects_the_normed_latent() -> None:
    attention = object.__new__(v41.DeepseekV41Attention)
    torch.nn.Module.__init__(attention)
    projected = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    projection = Mock(return_value=projected)
    attention.indexer = SimpleNamespace(wq_b=projection, n_heads=2, head_dim=2)
    qr = torch.randn(5, 8, dtype=torch.bfloat16)

    actual = attention.indexer_query(qr)

    projection.assert_called_once_with(qr)
    torch.testing.assert_close(actual, projected.reshape(5, 2, 2))
