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

"""NPU-owned gated MLP: NZ weight packing for the dense projections."""

from __future__ import annotations

import torch

from xllm.python import kernels
from xllm.python.layers.gated_mlp import GatedMLP


class NpuGatedMLP(GatedMLP):
    """Gated MLP that owns its NPU NZ weight preparation.

    The target-model load path gates on NPU device and BF16 dtype before
    calling :meth:`prepare_weights_nz`; the packing itself is unconditional.
    """

    def prepare_weights_nz(self) -> None:
        """Pack both dense projections into the fractal-NZ layout.

        ``down_proj`` is already transposed to ``[K, N]`` by
        ``process_weights_after_loading`` before this runs.
        """
        self.gate_up_proj.weight.data = kernels.prepare_weights_nz(self.gate_up_proj.weight.data)
        self.down_proj.weight.data = kernels.prepare_weights_nz(self.down_proj.weight.data)
