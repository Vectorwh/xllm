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

"""DiT text encoders: form-A registry and the shared protocol."""

from __future__ import annotations

from .base import (
    TextEncoderOutput,
    TextEncoderProtocol,
    create_text_encoder,
    register_dit_encoder,
)

__all__ = [
    "TextEncoderOutput",
    "TextEncoderProtocol",
    "create_text_encoder",
    "register_dit_encoder",
]
