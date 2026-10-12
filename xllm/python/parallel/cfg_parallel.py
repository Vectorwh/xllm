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

"""Classifier-free guidance (CFG) parallelism.

Port of ``dit::CFGParallelMixin``. With a CFG group of size 2, one rank runs
the positive conditional and the other the negative; results are gathered so
both ranks hold {positive, negative}. Group size 1 runs both branches serially.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from xllm.python.distributed import collectives


class CFGParallelMixin:
    """Mixin providing classifier-free guidance parallelism.

    ``cfg_group`` is the registered group name (e.g. ``"dit_cfg"``); ``None``
    or an unregistered name degrades to serial execution.
    """

    def __init__(self, cfg_group: str | None = None) -> None:
        self.cfg_group = cfg_group

    def exec_with_cfg(
        self,
        forward_fn: Callable[[bool], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``forward_fn(is_positive)`` and return (positive, negative).

        The caller captures prompt embeddings and transformer context in the
        closure, mirroring the C++ lambda usage.
        """
        cfg_size = self._cfg_size()

        # Serial execution: evaluate positive and negative conditionals one by
        # one (also the path when no CFG group is registered).
        if cfg_size == 1:
            return forward_fn(True), forward_fn(False)

        if cfg_size != 2:
            raise ValueError(f"CFG parallel expects group size 1 or 2, got {cfg_size}")
        rank = collectives.named_group_rank(self.cfg_group, self._resolve_device())
        noise_pred = forward_fn(rank == 0)
        gathered = collectives.gather(noise_pred, 0, cfg_size, self.cfg_group)
        positive, negative = gathered.chunk(2, 0)
        return positive, negative

    def _cfg_size(self) -> int:
        return collectives.named_group_size(self.cfg_group, self._resolve_device())

    def _resolve_device(self) -> torch.device:
        device = getattr(self, "device", None)
        if isinstance(device, torch.device):
            return device
        device_attr = getattr(self, "_device", None)
        if isinstance(device_attr, torch.device):
            return device_attr
        raise RuntimeError("CFGParallelMixin requires a `device` attribute (torch.device) to resolve its process group")
