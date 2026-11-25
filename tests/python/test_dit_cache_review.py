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

"""Regression coverage for rank-local cache metadata diverging across SP."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .conftest import load_source_module


def _cache_sp_worker(rank: int, rendezvous: str) -> None:
    root = Path(__file__).parents[2]
    collectives = load_source_module(root / "xllm/python/distributed/collectives.py", "_xllm_collectives_under_test")
    cache = load_source_module(root / "xllm/python/model_executor/dit_cache/base.py", "_xllm_dit_cache_under_test")
    cache.collectives = collectives
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    key = ("cache_review_sp", "cpu")
    collectives._groups[key] = dist.group.WORLD
    try:
        policy = cache.DitCachePolicy(key[0], None)
        value = torch.ones(2)
        # Both ranks must reject even if just one has missing/mismatched data.
        assert not policy.is_similar(None if rank == 0 else value, value, 0.1)
        assert not policy.is_similar(value, torch.ones(3) if rank == 0 else value, 0.1)
        # Exact comparison and differing thresholds also require one decision.
        assert not policy.is_similar(value, value + rank, 0.0)
        assert policy.is_similar(value, value, 0.0 if rank == 0 else 0.1)
        # Unequal local shard sizes must use global sums/count, not rank means.
        lhs = torch.ones(1 if rank == 0 else 3)
        rhs = lhs + (0.2 if rank == 0 else 0.0)
        assert policy.is_similar(lhs, rhs, 0.06)
    finally:
        collectives._groups.pop(key, None)
        dist.destroy_process_group()


def test_cache_similarity_rank_asymmetry(tmp_path: Path) -> None:
    mp.spawn(_cache_sp_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=2, join=True)
