# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test for MindOneModelHubPlugin"""

import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from mindspore_gs.ptq.models import BaseQuantForCausalLM
from mindspore_gs.ptq.plugins.mindone_plugin import MindOneModelHubPlugin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindone")))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindone_model_hub_plugin_load():
    """
    Feature: MindOneModelHubPlugin load
    Description: Test load method loads models and quant cells
    Expectation: Load returns instance without errors
    """
    plugin = MindOneModelHubPlugin.load()
    assert isinstance(plugin, MindOneModelHubPlugin)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindone_model_hub_plugin_load_models():
    """
    Feature: MindOneModelHubPlugin _load_models
    Description: Test model loading functionality
    Expectation: Models are imported without errors
    """
    plugin = MindOneModelHubPlugin()
    # Should not raise any exceptions
    # pylint: disable=protected-access
    plugin._load_models()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindone_model_hub_plugin_load_quant_cells():
    """
    Feature: MindOneModelHubPlugin _load_quant_cells
    Description: Test quant cells loading functionality
    Expectation: Quant cells loading is handled gracefully (currently empty)
    """
    plugin = MindOneModelHubPlugin()
    # Should not raise any exceptions - currently empty implementation
    # pylint: disable=protected-access
    plugin._load_quant_cells()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindone_model_hub_plugin_create_model_valid_config():
    """
    Feature: MindOneModelHubPlugin create_model
    Description: Test create_model with valid directory containing config.json
    Expectation: Returns model instance or handles ImportError gracefully
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_type': 'test_model',
                'model_config': {
                    'hidden_size': 768,
                    'num_attention_heads': 12,
                    'num_hidden_layers': 12
                }
            }, f)

        plugin = MindOneModelHubPlugin()
        # Mock the from_pretrained method to avoid actual model loading
        with patch('mindspore_gs.ptq.models.mindone_models.mindone_model.MindOneModel.from_pretrained') \
                as mock_from_pretrained:
            mock_model = MagicMock(spec=BaseQuantForCausalLM)
            mock_from_pretrained.return_value = mock_model

            model = plugin.create_model(temp_dir)
            assert model is mock_model
            mock_from_pretrained.assert_called_once_with(temp_dir)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindone_model_hub_plugin_create_model_invalid_path():
    """
    Feature: MindOneModelHubPlugin create_model invalid
    Description: Test create_model with invalid directory path
    Expectation: Raises appropriate error for invalid path
    """
    plugin = MindOneModelHubPlugin()

    # Test with non-existent directory
    with pytest.raises((FileNotFoundError, ValueError)):
        plugin.create_model("non_existent_directory")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mindone_model_hub_plugin_create_model_invalid_config():
    """
    Feature: MindOneModelHubPlugin create_model invalid
    Description: Test create_model with invalid config.json content
    Expectation: Raises appropriate error for invalid config
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("invalid json content")

        plugin = MindOneModelHubPlugin()
        # This should raise some form of parsing error
        with pytest.raises((ValueError, RuntimeError)):
            plugin.create_model(temp_dir)
