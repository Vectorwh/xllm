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

"""DiT cache strategies.

Cache hooks are acceleration policies attached at block/step boundaries —
replaceable strategy plugins, not model structure (mirroring mindiesd's
standalone ``cache_agent``). A consuming transformer must call the four hook
points below explicitly. The Python Joyo V2 pipeline does not integrate these
hooks and rejects non-None policies instead of silently ignoring them.

The strategy plugins themselves (FBCache, TaylorSeer, FBCacheTaylorSeer,
ResidualCache) live as sibling modules and register themselves through
:func:`register_cache_policy`. They mirror the C++ ``DitCacheImpl`` classes in
``core/framework/dit_cache/``; a :class:`DiTCache` wrapper splits state into a
conditional and an unconditional half exactly like the C++ ``DiTCache``
singleton, since the transformer's ``on_after_*`` hooks don't carry ``use_cfg``.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.distributed as dist

from xllm.python.distributed import collectives


class DitCacheHooks(Protocol):
    """Four hook points matching the C++ ``DiTCache`` call sites.

    ``on_before_*`` return True to SKIP the wrapped computation (cache hit);
    ``on_after_*`` receive the tensor map (possibly carrying the originals)
    and return the tensor map to continue with.
    """

    def on_before_step(self, step_index: int, tensors: dict[str, torch.Tensor], use_cfg: bool) -> bool: ...

    def on_after_step(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def on_before_block(self, block_index: int, use_cfg: bool) -> bool: ...

    def on_after_block(self, block_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def set_context(self, infer_steps: int, num_blocks: int) -> None: ...


class NoopCacheHooks:
    """Policy ``None``: run every block of every step."""

    def on_before_step(self, step_index: int, tensors: dict[str, torch.Tensor], use_cfg: bool) -> bool:
        del step_index, tensors, use_cfg
        return False

    def on_after_step(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return tensors

    def on_before_block(self, block_index: int, use_cfg: bool) -> bool:
        del block_index, use_cfg
        return False

    def on_after_block(self, block_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        del block_index
        return tensors

    def set_context(self, infer_steps: int, num_blocks: int) -> None:
        del infer_steps, num_blocks


class DitCachePolicy:
    """Stateful cache strategy; the Python mirror of the C++ ``DitCacheImpl``.

    Subclasses implement the four hook methods (without ``use_cfg`` — the
    :class:`DiTCache` wrapper splits cond/uncond state). ``sp_group`` is the
    registered process-group name used to all-reduce the similarity metric so
    every sequence-parallel rank takes the identical reuse decision.
    """

    def __init__(self, sp_group: str | None, options: dict[str, Any] | None) -> None:
        options = options or {}
        self.sp_group = sp_group
        self.warmup_steps = int(options.get("warmup_steps", 0))
        self.infer_steps = 0
        self.num_blocks = 0

    def set_context(self, infer_steps: int, num_blocks: int) -> None:
        self.infer_steps = int(infer_steps)
        self.num_blocks = int(num_blocks)

    # ---- helpers shared by FBCache / FBCacheTaylorSeer ----------------------

    def is_similar(self, lhs: torch.Tensor | None, rhs: torch.Tensor | None, threshold: float) -> bool:
        """Use a shared validity/threshold decision before comparing SP shards."""
        device = collectives.named_group_device(self.sp_group) if self.sp_group is not None else None
        if device is None:
            device = lhs.device if lhs is not None else rhs.device if rhs is not None else torch.device("cpu")
        valid = (
            lhs is not None
            and rhs is not None
            and lhs.shape == rhs.shape
            and lhs.device == rhs.device
            and lhs.dtype == rhs.dtype
        )
        group = collectives.named_group(self.sp_group, device)
        distributed = group is not None and group.size() > 1
        # Even a rank with no tensor joins this fixed-size control reduction.
        # MIN makes any invalid shard reject reuse; differing thresholds use
        # the strictest value and select the same comparison on every rank.
        control = torch.tensor([float(valid), threshold], dtype=torch.float32, device=device)
        if distributed:
            dist.all_reduce(control, op=dist.ReduceOp.MIN, group=group)
        if not bool(control[0].item()):
            return False
        threshold = float(control[1].item())
        if threshold <= 0.0:
            close = torch.tensor(float(torch.allclose(lhs, rhs)), dtype=torch.float32, device=device)
            if distributed:
                dist.all_reduce(close, op=dist.ReduceOp.MIN, group=group)
            return bool(close.item())

        # SP shards may have different sizes: reduce sums and count, not means.
        sum_abs_diff = (lhs - rhs).abs().to(torch.float32).sum()
        sum_abs_lhs = lhs.abs().to(torch.float32).sum()
        count = torch.full((), float(lhs.numel()), dtype=torch.float32, device=lhs.device)
        stats = torch.stack([sum_abs_diff, sum_abs_lhs, count]).to(device)
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
        sum_abs_diff, sum_abs_lhs, count = stats.unbind()
        if count.item() == 0:
            return False
        rel = (sum_abs_diff / count) / (sum_abs_lhs / count + 1e-6)
        return bool((rel < threshold).item())

    # ---- hook methods (subclasses override) ---------------------------------

    def on_before_step(self, step_index: int, tensors: dict[str, torch.Tensor]) -> bool:
        raise NotImplementedError

    def on_after_step(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def on_before_block(self, block_index: int) -> bool:
        raise NotImplementedError

    def on_after_block(self, block_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class DiTCache:
    """Cond/uncond split wrapper; mirrors the C++ ``DiTCache`` singleton.

    Holds two independent policy instances and routes each hook to the one
    selected by the most recent ``use_cfg`` (captured on ``on_before_*``), since
    the Python transformer does not pass ``use_cfg`` through ``on_after_*``.
    """

    def __init__(self, policy_cls: type, sp_group: str | None, options: dict[str, Any] | None) -> None:
        options = dict(options or {})
        self._uncond = policy_cls(sp_group, options)  # use_cfg == False
        self._cond = policy_cls(sp_group, options)  # use_cfg == True
        self._current_cfg = False

    def _active(self) -> DitCachePolicy:
        return self._cond if self._current_cfg else self._uncond

    def on_before_step(self, step_index: int, tensors: dict[str, torch.Tensor], use_cfg: bool) -> bool:
        self._current_cfg = use_cfg
        return self._active().on_before_step(step_index, tensors)

    def on_after_step(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._active().on_after_step(tensors)

    def on_before_block(self, block_index: int, use_cfg: bool) -> bool:
        self._current_cfg = use_cfg
        return self._active().on_before_block(block_index)

    def on_after_block(self, block_index: int, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._active().on_after_block(block_index, tensors)

    def set_context(self, infer_steps: int, num_blocks: int) -> None:
        self._cond.set_context(infer_steps, num_blocks)
        self._uncond.set_context(infer_steps, num_blocks)


_POLICIES: dict[str, tuple[str, str]] = {}


def register_cache_policy(name: str, module_name: str, class_name: str) -> None:
    _POLICIES[name] = (module_name, class_name)


def create_cache_hooks(policy: str | None, sp_group: str | None = None, options: dict[str, Any] | None = None) -> Any:
    """Build the hooks object for ``policy``; None/'None' means run everything.

    Strategy plugins (FBCache, TaylorSeer, ...) register themselves here and
    live as sibling modules of this file.
    """
    if not policy or policy == "None":
        return NoopCacheHooks()
    if policy not in _POLICIES:
        raise KeyError(f"DiT cache policy '{policy}' not registered; available: {sorted(_POLICIES)}")
    from importlib import import_module

    module_name, class_name = _POLICIES[policy]
    policy_cls = getattr(import_module(module_name), class_name)
    return DiTCache(policy_cls, sp_group, options)
