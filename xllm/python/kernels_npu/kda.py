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

"""NPU kernel bindings for KDA speculative verification."""

from __future__ import annotations

from typing import NamedTuple

import torch


class KdaSafeGateParameters(NamedTuple):
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float


def recurrent_kda_verify(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    *,
    safe_gate: KdaSafeGateParameters | None = None,
) -> torch.Tensor:
    """Read the accepted checkpoint and write the new checkpoints in place.

    The kernel consumes packed TND tensors and a request-by-checkpoint slot
    table. Each request reads checkpoint column num_accepted_tokens - 1
    and writes one checkpoint per verify row. With safe_gate, gate contains
    raw projections.
    """
    from fla_npu.ops.ascendc import recurrent_kda

    if safe_gate is None:
        gate_kwargs = dict(use_gate_in_kernel=False)
    else:
        gate_kwargs = dict(
            A_log=safe_gate.a_log,
            dt_bias=safe_gate.dt_bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=safe_gate.lower_bound,
        )
    result = recurrent_kda(
        query,
        key,
        value,
        gate,
        beta,
        initial_state=ssm_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        layout="TND",
        scale=1.0 / (query.shape[-1] ** 0.5),
        output_final_state=True,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        **gate_kwargs,
    )
    return result[0] if isinstance(result, tuple) else result
