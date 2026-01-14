# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Callable

import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class OLMo2ModelProvider(GPTModelProvider):
    """Configuration class for OLMo2 models.

    OLMo2 is a decoder-only transformer model from Allen Institute for AI (AllenAI)
    that uses RMSNorm, SwiGLU activation, and RoPE positional embeddings.
    """

    # Common configs across OLMo2 model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    layernorm_epsilon: float = 1e-5
    init_method_std: float = 0.02
    rotary_base: float = 10000.0

    # Fusions
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
