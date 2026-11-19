# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""GLM-5.2/5.3 MTP draft model for ``model_type=glm_moe_dsa_mtp``.

The speculative worker (``MTPWorkerImpl``) and the MTP scheduling stay in C++;
this module only describes the draft computation, mirroring
``deepseek_v32_mtp.py``:

- The draft is the checkpoint's appended layer
  ``model.layers.<num_hidden_layers>`` exported by ``tools/export_mtp.py``,
  renamed to ``model.layers.0.*`` / ``model.{embed_tokens,shared_head,enorm,
  hnorm,eh_proj,rot}``.
- ``eh_proj`` fuses ``[enorm(token_embed) ‖ hnorm(carried_hidden)]``.
  ``input_embedding`` is the hidden state produced by the target model (or by
  the previous draft step); when None the draft runs off its own token
  embedding.
- ``embed_tokens`` and ``lm_head`` are left as None: the C++ side wires the
  target model's copies in (``share_python_model_weights`` in
  ``py_causal_lm.cpp``), so the draft does not need its own 2x1.9 GB copies.

With ``index_share_for_mtp_iteration``, draft step 0 computes its own top-k.
The Python executor compacts the last query row of each request and owns this
selection for steps 1+: only scoring is skipped, while the indexer K cache
remains current. C++ supplies scheduling metadata, not top-k tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.model_executor.model_output import ModelOutput
from xllm.python.models.glm5_2 import (
    Glm52Config,
    Glm52DecoderLayer,
    Glm52ForCausalLM,
    Glm52YarnRotaryEmbedding,
    glm52_eplv2_comm_policy,
)
from xllm.python.models.weight_utils import W8A8WeightLoader

# The exporter renames the appended layer's shared head to
# ``model.shared_head.*``; the draft body keeps the usual ``model.norm.weight``
# name, so accept every spelling the checkpoints use.
_MTP_NORM_ALIASES: dict[str, tuple[str, ...]] = {
    "model.norm.weight": (
        "model.norm.weight",
        "model.final_norm.weight",
        "model.shared_head.norm.weight",
    ),
}


class Glm52MtpModel(nn.Module):
    """MTP draft body: eh_proj fusion + one decoder layer + head norm."""

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        if cfg.hidden_size % tp != 0:
            raise ValueError(f"hidden_size {cfg.hidden_size} not divisible by tp_size {tp}")
        if cfg.n_layers != 1:
            raise ValueError(f"glm_moe_dsa_mtp draft expects exactly 1 layer, got {cfg.n_layers}")
        if cfg.cp_size > 1:
            raise ValueError("glm_moe_dsa_mtp draft does not support context parallelism yet")
        if cfg.mlp_layer_types[0] != "sparse":
            raise ValueError(
                "glm_moe_dsa_mtp draft layer 0 must be MoE (mlp_layer_types[0]='sparse'); "
                f"got {cfg.mlp_layer_types[0]!r} with first_k_dense_replace={cfg.first_k_dense_replace}"
            )
        if cfg.indexer_types[0] == "shared":
            raise ValueError(
                "glm_moe_dsa_mtp draft layer 0 must carry a full indexer "
                "(indexer_types[0] != 'shared'), otherwise the draft cannot seed its own top-k"
            )

        self.cfg = cfg
        self.supports_mtp_topk_reuse = cfg.index_share_for_mtp_iteration
        # Wired by the C++ side to the target model's modules.
        self.embed_tokens: nn.Module | None = None
        self.eh_proj = ColumnParallelLinear(
            2 * cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        # Optional linear applied to the carried hidden state (the checkpoint
        # ships model.rot.weight; when absent the hidden state is used as-is).
        self.rot: ColumnParallelLinear | None = None
        self.enable_rot = False
        self.enorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.hnorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self._comm_policy = glm52_eplv2_comm_policy(cfg)
        self.layers = nn.ModuleList([Glm52DecoderLayer(cfg, 0, dtype, device, comm_policy=self._comm_policy)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = Glm52YarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor | ModelOutput:
        if self.embed_tokens is None:
            raise RuntimeError(
                "glm_moe_dsa_mtp draft has no embed_tokens: the C++ worker must share the target model's embedding"
            )
        token_hidden = self.embed_tokens(input_ids)
        carried = token_hidden if input_embedding is None else input_embedding
        if self.enable_rot:
            carried = self.rot(carried)
        hidden = self.eh_proj(torch.cat((self.enorm(token_hidden), self.hnorm(carried)), dim=-1))

        positions = positions.to(torch.int64).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        residual: torch.Tensor | None = None
        prev_topk: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual, prev_topk = layer(hidden, residual, positions, cos_sin_cache, prev_topk)
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        if self.supports_mtp_topk_reuse:
            return ModelOutput(hidden_states=hidden, mtp_topk_indices=prev_topk)
        return hidden


class Glm52MtpForCausalLM(Glm52ForCausalLM):
    """GLM-5.2/5.3 MTP draft calculator; scheduling stays in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        # The exported draft config still carries the *target* metadata
        # (``indexer_types`` / ``mlp_layer_types`` can be target-length, and
        # ``index_topk_pattern`` is rewritten to "S" for the native reuse plan).
        # The draft is a single MoE layer with a full indexer, so pin the layout
        # instead of inheriting the target's.
        self.cfg.indexer_types = ["full"] * self.cfg.n_layers
        self.cfg.mlp_layer_types = ["sparse"] * self.cfg.n_layers
        self.cfg.first_k_dense_replace = 0
        self.model = Glm52MtpModel(self.cfg, self.dtype, self.device)

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        # Optional parameters must exist before the loader snapshots their names.
        self.model.enable_rot = W8A8WeightLoader.state_dict_has(state_dicts, "rot.weight", src_prefixes=("", "model."))
        if self.model.enable_rot:
            self.model.rot = ColumnParallelLinear(
                self.cfg.hidden_size,
                self.cfg.hidden_size // self.cfg.tp_size,
                self.cfg.tp_size,
                gather_output=True,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.model.rot = None
        loader = W8A8WeightLoader(
            self,
            state_dicts,
            self.cfg.tp_size,
            self.cfg.tp_rank,
            src_prefixes=("", "model."),
            name_aliases=_MTP_NORM_ALIASES,
        )
        super().load_weights(
            state_dicts,
            tp_rank,
            tp_size,
            load_lm_head=False,
            load_embedding=False,
            loader=loader,
        )

        def copy_if_present(module_name: str, required: bool = False) -> bool:
            key = module_name + ".weight"
            if not loader.has(key):
                if required:
                    raise KeyError(f"missing required MTP weight: {key}")
                return False
            tensor = loader.load_tensor(key)
            parameter = self.get_parameter("model." + key)
            if tensor.shape != parameter.shape and tensor.dim() == 2:
                tensor = loader.shard(tensor, dim=0)
            loader.copy_in("model." + key, tensor)
            return True

        copy_if_present("eh_proj", required=True)
        copy_if_present("enorm", required=True)
        copy_if_present("hnorm", required=True)
        if self.model.enable_rot:
            copy_if_present("rot", required=True)


# Registration is centralised in xllm.python.registry; import there.
