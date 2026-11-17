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

"""Cache-layout helpers for the fused MLA operators.

Kept free of operator imports so the layout decisions can be unit tested
without the NPU kernels.
"""

from __future__ import annotations

import torch

from .constants import FRACTAL_NZ_FORMAT

PA_CACHE_MODE = "PA"
PA_NZ_CACHE_MODE = "PA_NZ"


def mla_cache_mode(kv_cache: torch.Tensor) -> str:
    """Return the cache layout name expected by the fused KV operator."""
    import torch_npu

    # A missing runtime or format probe is an environment error, not a layout.
    if torch_npu.get_npu_format(kv_cache) == FRACTAL_NZ_FORMAT:
        return PA_NZ_CACHE_MODE
    return PA_CACHE_MODE


def mla_prolog_cache_check(
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
) -> str | None:
    """Return why ``mla_prolog_v3`` cannot write into these caches, if it cannot.

    The fused op is pinned to BF16 caches in the plain paged layout and passes
    ``kv_cache_quant_mode=0``, so a quantized or internally packed cache must be
    served by the split path, which selects the cache mode and quant mode itself.
    """
    if kv_cache.dtype != torch.bfloat16 or rope_cache.dtype != torch.bfloat16:
        return f"unsupported cache dtypes: kv={kv_cache.dtype}, rope={rope_cache.dtype}"
    layout = mla_cache_mode(kv_cache)
    if layout != PA_CACHE_MODE:
        return f"unsupported kv cache layout: {layout}"
    return None
