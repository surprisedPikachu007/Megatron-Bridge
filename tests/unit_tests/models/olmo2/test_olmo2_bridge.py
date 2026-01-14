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

from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.olmo2.olmo2_bridge import OLMo2Bridge
from megatron.bridge.models.olmo2.olmo2_provider import OLMo2ModelProvider


class TestMegatronOLMo2Bridge:
    """Test cases for MegatronOLMo2Bridge class."""

    @pytest.fixture
    def olmo2_7b_config_dict(self):
        """Create a sample OLMo2-7B configuration."""
        return {
            "architectures": ["Olmo2ForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "model_type": "olmo2",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.57.1",
            "use_cache": True,
            "vocab_size": 50280,
        }

    @pytest.fixture
    def olmo2_13b_config_dict(self):
        """Create a sample OLMo2-13B configuration."""
        return {
            "architectures": ["Olmo2ForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 5120,
            "initializer_range": 0.02,
            "intermediate_size": 13824,
            "max_position_embeddings": 4096,
            "model_type": "olmo2",
            "num_attention_heads": 40,
            "num_hidden_layers": 40,
            "num_key_value_heads": 40,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 50280,
        }

    @pytest.fixture
    def olmo2_7b_config(self, olmo2_7b_config_dict):
        """Create an OLMo2 config instance for 7B model."""
        config = Mock()
        for key, value in olmo2_7b_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def olmo2_13b_config(self, olmo2_13b_config_dict):
        """Create an OLMo2 config instance for 13B model."""
        config = Mock()
        for key, value in olmo2_13b_config_dict.items():
            setattr(config, key, value)
        return config

    @pytest.fixture
    def mock_olmo2_7b_model(self, olmo2_7b_config):
        """Create a mock Olmo2ForCausalLM 7B model."""
        try:
            from transformers import Olmo2ForCausalLM

            mock_model = Mock(spec=Olmo2ForCausalLM)
        except ImportError:
            mock_model = Mock()
        mock_model.config = olmo2_7b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_olmo2_13b_model(self, olmo2_13b_config):
        """Create a mock Olmo2ForCausalLM 13B model."""
        try:
            from transformers import Olmo2ForCausalLM

            mock_model = Mock(spec=Olmo2ForCausalLM)
        except ImportError:
            mock_model = Mock()
        mock_model.config = olmo2_13b_config
        mock_model.dtype = torch.bfloat16
        return mock_model

    @pytest.fixture
    def mock_pretrained_olmo2_7b(self, olmo2_7b_config):
        """Create a mock PreTrainedCausalLM with OLMo2 7B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmo2_7b_config
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import Olmo2ForCausalLM

            mock_pretrained.model = Mock(spec=Olmo2ForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    @pytest.fixture
    def mock_pretrained_olmo2_13b(self, olmo2_13b_config):
        """Create a mock PreTrainedCausalLM with OLMo2 13B model."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmo2_13b_config
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import Olmo2ForCausalLM

            mock_pretrained.model = Mock(spec=Olmo2ForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()
        mock_pretrained.model.dtype = torch.bfloat16
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that MegatronOLMo2Bridge is properly registered."""
        assert issubclass(OLMo2Bridge, MegatronModelBridge)

    def test_provider_bridge_basic_7b(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        """Test basic provider_bridge functionality for OLMo2 7B."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert isinstance(result, OLMo2ModelProvider)
        assert result.num_layers == olmo2_7b_config.num_hidden_layers
        assert result.hidden_size == olmo2_7b_config.hidden_size
        assert result.num_attention_heads == olmo2_7b_config.num_attention_heads
        assert result.seq_length == olmo2_7b_config.max_position_embeddings
        assert result.rotary_base == olmo2_7b_config.rope_theta

    def test_provider_bridge_basic_13b(self, mock_pretrained_olmo2_13b, olmo2_13b_config):
        """Test basic provider_bridge functionality for OLMo2 13B."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_13b)

        assert isinstance(result, OLMo2ModelProvider)
        assert result.num_layers == olmo2_13b_config.num_hidden_layers
        assert result.hidden_size == olmo2_13b_config.hidden_size
        assert result.num_attention_heads == olmo2_13b_config.num_attention_heads

    def test_provider_bridge_vocabulary(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        """Test vocabulary size mapping."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert result.vocab_size == olmo2_7b_config.vocab_size
        assert result.share_embeddings_and_output_weights == olmo2_7b_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        """Test attention configuration mapping."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert result.num_attention_heads == olmo2_7b_config.num_attention_heads
        assert result.num_query_groups == olmo2_7b_config.num_key_value_heads

    def test_provider_bridge_mlp_config(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        """Test MLP configuration mapping."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert result.ffn_hidden_size == olmo2_7b_config.intermediate_size
        assert result.gated_linear_unit is True

    def test_provider_bridge_normalization(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        """Test normalization configuration."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert result.layernorm_epsilon == olmo2_7b_config.rms_norm_eps

    def test_provider_bridge_position_embedding(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        """Test position embedding configuration."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert result.rotary_base == olmo2_7b_config.rope_theta
        assert result.position_embedding_type == "rope"

    def test_provider_bridge_dtype_handling_bfloat16(self, olmo2_7b_config):
        """Test bfloat16 dtype handling in provider_bridge."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmo2_7b_config
        mock_pretrained.config.torch_dtype = torch.bfloat16
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import Olmo2ForCausalLM

            mock_pretrained.model = Mock(spec=Olmo2ForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()

        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.params_dtype == torch.bfloat16
        assert result.bf16 is True
        assert result.fp16 is False

    def test_provider_bridge_dtype_handling_fp16(self, olmo2_7b_config):
        """Test FP16 dtype handling in provider_bridge."""
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = olmo2_7b_config
        mock_pretrained.config.torch_dtype = torch.float16
        mock_pretrained.generation_config = Mock()
        try:
            from transformers import Olmo2ForCausalLM

            mock_pretrained.model = Mock(spec=Olmo2ForCausalLM)
        except ImportError:
            mock_pretrained.model = Mock()

        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained)

        assert result.params_dtype == torch.float16
        assert result.fp16 is True
        assert result.bf16 is False

    def test_provider_bridge_init_method_std(self, mock_pretrained_olmo2_7b, olmo2_7b_config):
        """Test initializer range mapping."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert result.init_method_std == olmo2_7b_config.initializer_range

    def test_provider_bridge_generation_config(self, mock_pretrained_olmo2_7b):
        """Test that generation config is passed through."""
        bridge = OLMo2Bridge()
        result = bridge.provider_bridge(mock_pretrained_olmo2_7b)

        assert result.generation_config == mock_pretrained_olmo2_7b.generation_config

    def test_mapping_registry_implementation(self, mock_pretrained_olmo2_7b):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = OLMo2Bridge()
        mapping_registry = bridge.mapping_registry()

        assert mapping_registry is not None


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with OLMo2 models."""

    @pytest.fixture
    def olmo2_configs(self):
        """Different OLMo2 model configurations for testing."""
        return {
            "olmo2-7b": {
                "architectures": ["Olmo2ForCausalLM"],
                "model_type": "olmo2",
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "intermediate_size": 11008,
                "vocab_size": 50280,
                "max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-5,
                "torch_dtype": "bfloat16",
                "tie_word_embeddings": False,
                "initializer_range": 0.02,
            },
        }

    def test_supports_olmo2_architectures(self, olmo2_configs):
        """Test that AutoBridge.supports correctly identifies OLMo2 models."""
        for model_name, config_dict in olmo2_configs.items():
            config = Mock()
            for key, value in config_dict.items():
                setattr(config, key, value)
            assert AutoBridge.supports(config) is True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["Olmo2Model"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) is False
