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

"""DiT cache strategies (execution-layer acceleration plugins)."""

from __future__ import annotations

# Importing the policy modules registers them with ``register_cache_policy``.
from . import fbcache, fbcache_taylorseer, residual_cache, taylorseer  # noqa: F401
from .base import DitCacheHooks, NoopCacheHooks, create_cache_hooks, register_cache_policy

__all__ = [
    "DitCacheHooks",
    "NoopCacheHooks",
    "create_cache_hooks",
    "register_cache_policy",
]
