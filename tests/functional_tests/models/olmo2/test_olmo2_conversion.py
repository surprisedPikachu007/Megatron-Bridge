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

import json
from pathlib import Path

import pytest
import torch


HF_OLMO2_TOY_MODEL_CONFIG = {
    "architectures": ["Olmo2ForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 1024,  # Smaller than real model for faster testing
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": 4096,
    "model_type": "olmo2",
    "num_attention_heads": 16,
    "num_hidden_layers": 2,  # Much smaller for testing
    "num_key_value_heads": 16,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.57.1",
    "use_cache": True,
    "vocab_size": 50280,
}


class TestOLMo2Conversion:
    """
    Test OLMo2 model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def olmo2_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace OLMo2 toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("olmo2_toy_model")
        model_dir = temp_dir / "olmo2_toy"

        # Create OLMo2 config from the toy model config
        try:
            from transformers import Olmo2Config, Olmo2ForCausalLM

            config = Olmo2Config(**HF_OLMO2_TOY_MODEL_CONFIG)
        except ImportError:
            # If Olmo2Config is not available, use AutoConfig
            from transformers import AutoConfig

            config = AutoConfig.for_model(model_type="olmo2", **HF_OLMO2_TOY_MODEL_CONFIG)

        config.torch_dtype = torch.bfloat16

        # Create model with random weights and convert to bfloat16
        try:
            from transformers import Olmo2ForCausalLM

            model = Olmo2ForCausalLM(config)
        except ImportError:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_config(config)

        model = model.bfloat16()

        # Debug: Check model dtype before saving
        for name, param in model.named_parameters():
            print(f"Before save - {name}: {param.dtype}")
            break  # Just check the first parameter

        # Save the model
        model.save_pretrained(model_dir)

        # Save tokenizer config
        tokenizer_config = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "model_max_length": config.max_position_embeddings,
            "tokenizer_class": "PreTrainedTokenizerFast",
        }
        with open(model_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Create a minimal tokenizer.json file
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "model": {"type": "BPE", "vocab": {}, "merges": []},
        }
        with open(model_dir / "tokenizer.json", "w") as f:
            json.dump(tokenizer_json, f, indent=2)

        # Verify config was saved
        config_path = model_dir / "config.json"
        assert config_path.exists(), f"Config file not created at {config_path}"

        # Debug: Read and check the saved config
        with open(config_path, "r") as f:
            saved_config = json.load(f)
            print(f"Saved config torch_dtype: {saved_config.get('torch_dtype')}")

        return str(model_dir)

    @pytest.mark.parametrize(
        "tp,pp",
        [
            (1, 1),
            (2, 1),
            (1, 2),
        ],
    )
    def test_olmo2_conversion_with_parallelism(self, olmo2_toy_model_path, tp, pp):
        """
        Test OLMo2 model conversion with different tensor and pipeline parallelism configurations.

        Args:
            olmo2_toy_model_path: Path to the OLMo2 toy model directory (from fixture)
            tp: Tensor parallelism degree
            pp: Pipeline parallelism degree
        """
        pytest.importorskip("transformers")
        from megatron.bridge.models import AutoBridge

        # Create bridge from local HF model
        bridge = AutoBridge.from_hf_pretrained(olmo2_toy_model_path)

        # Convert to Megatron provider
        provider = bridge.to_megatron_provider()

        # Configure parallelism
        provider.tensor_model_parallel_size = tp
        provider.pipeline_model_parallel_size = pp
        provider.finalize()

        # Verify provider configuration
        assert provider.tensor_model_parallel_size == tp
        assert provider.pipeline_model_parallel_size == pp
        assert provider.hidden_size == HF_OLMO2_TOY_MODEL_CONFIG["hidden_size"]
        assert provider.num_layers == HF_OLMO2_TOY_MODEL_CONFIG["num_hidden_layers"]
        assert provider.num_attention_heads == HF_OLMO2_TOY_MODEL_CONFIG["num_attention_heads"]

    def test_olmo2_basic_conversion(self, olmo2_toy_model_path):
        """
        Test basic OLMo2 model conversion without parallelism.

        Args:
            olmo2_toy_model_path: Path to the OLMo2 toy model directory (from fixture)
        """
        pytest.importorskip("transformers")
        from megatron.bridge.models import AutoBridge

        # Create bridge from local HF model
        bridge = AutoBridge.from_hf_pretrained(olmo2_toy_model_path)

        # Convert to Megatron provider
        provider = bridge.to_megatron_provider()

        # Verify provider type
        from megatron.bridge.models.olmo2 import OLMo2ModelProvider

        assert isinstance(provider, OLMo2ModelProvider)

        # Verify configuration matches the toy model
        assert provider.hidden_size == HF_OLMO2_TOY_MODEL_CONFIG["hidden_size"]
        assert provider.num_layers == HF_OLMO2_TOY_MODEL_CONFIG["num_hidden_layers"]
        assert provider.num_attention_heads == HF_OLMO2_TOY_MODEL_CONFIG["num_attention_heads"]
        assert provider.num_query_groups == HF_OLMO2_TOY_MODEL_CONFIG["num_key_value_heads"]
        assert provider.ffn_hidden_size == HF_OLMO2_TOY_MODEL_CONFIG["intermediate_size"]
        assert provider.seq_length == HF_OLMO2_TOY_MODEL_CONFIG["max_position_embeddings"]
        assert provider.vocab_size == HF_OLMO2_TOY_MODEL_CONFIG["vocab_size"]
        assert provider.rotary_base == HF_OLMO2_TOY_MODEL_CONFIG["rope_theta"]
        assert provider.layernorm_epsilon == HF_OLMO2_TOY_MODEL_CONFIG["rms_norm_eps"]
        assert provider.init_method_std == HF_OLMO2_TOY_MODEL_CONFIG["initializer_range"]
