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

import torch
import torch.nn.functional as F

from megatron.bridge.models.olmo2.olmo2_provider import OLMo2ModelProvider


class TestOLMo2ModelProvider:
    """Test cases for OLMo2ModelProvider class."""

    def test_olmo2_model_provider_initialization(self):
        """Test OLMo2ModelProvider can be initialized with default values."""
        provider = OLMo2ModelProvider()

        # Check OLMo2-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.seq_length == 4096
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is False
        assert provider.layernorm_epsilon == 1e-5
        assert provider.init_method_std == 0.02
        assert provider.rotary_base == 10000.0

        # Check position embedding
        assert provider.position_embedding_type == "rope"

        # Check fusion settings
        assert provider.bias_activation_fusion is True
        assert provider.masked_softmax_fusion is True
        assert provider.persist_layer_norm is True
        assert provider.bias_dropout_fusion is True
        assert provider.apply_rope_fusion is True

    def test_olmo2_model_provider_custom_initialization(self):
        """Test OLMo2ModelProvider can be initialized with custom values."""
        provider = OLMo2ModelProvider(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            ffn_hidden_size=11008,
            seq_length=8192,
            rotary_base=500000.0,
        )

        assert provider.num_layers == 32
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.ffn_hidden_size == 11008
        assert provider.seq_length == 8192
        assert provider.rotary_base == 500000.0

    def test_olmo2_model_provider_normalization(self):
        """Test normalization configuration."""
        provider = OLMo2ModelProvider()
        assert provider.normalization == "RMSNorm"
        assert provider.layernorm_epsilon == 1e-5

    def test_olmo2_model_provider_activation(self):
        """Test activation function configuration."""
        provider = OLMo2ModelProvider()
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True

    def test_olmo2_model_provider_rope_config(self):
        """Test RoPE configuration."""
        provider = OLMo2ModelProvider()
        assert provider.position_embedding_type == "rope"
        assert provider.rotary_base == 10000.0
        assert provider.apply_rope_fusion is True

    def test_olmo2_model_provider_bias_config(self):
        """Test bias configuration."""
        provider = OLMo2ModelProvider()
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False

    def test_olmo2_model_provider_dropout_config(self):
        """Test dropout configuration."""
        provider = OLMo2ModelProvider()
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0

    def test_olmo2_model_provider_embeddings_config(self):
        """Test embeddings configuration."""
        provider = OLMo2ModelProvider()
        assert provider.share_embeddings_and_output_weights is False
