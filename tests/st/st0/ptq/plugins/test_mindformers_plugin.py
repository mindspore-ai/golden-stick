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
"""Test for MFModelHubPlugin"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))
from mindspore_gs.ptq.plugins.mindformers_plugin import MFModelHubPlugin
from mindspore_gs.ptq.models import BaseQuantForCausalLM


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_model_hub_plugin_load():
    """
    Feature: MFModelHubPlugin load
    Description: Test load method loads models and quant cells
    Expectation: Load returns instance without errors
    """
    plugin = MFModelHubPlugin.load()
    assert isinstance(plugin, MFModelHubPlugin)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_model_hub_plugin_load_models():
    """
    Feature: MFModelHubPlugin _load_models
    Description: Test model loading functionality
    Expectation: Models are imported without errors
    """
    plugin = MFModelHubPlugin()
    # pylint: disable=protected-access
    plugin._load_models()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_model_hub_plugin_load_quant_cells():
    """
    Feature: MFModelHubPlugin _load_quant_cells
    Description: Test quant cells loading functionality
    Expectation: Quant cells are imported and registered without errors
    """
    plugin = MFModelHubPlugin()
    # pylint: disable=protected-access
    plugin._load_quant_cells()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_model_hub_plugin_create_model_valid_config():
    """
    Feature: MFModelHubPlugin create_model
    Description: Test create_model with valid configuration
    Expectation: Returns model instance or handles ImportError gracefully
    """
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp_file:
        tmp_file.write(b"model:\n  model_name: qwen3_7b")
        tmp_file.flush()

    try:
        plugin = MFModelHubPlugin()
        # Mock the from_pretrained method to avoid actual model loading
        with patch('mindspore_gs.ptq.plugins.mindformers_plugin.MFModel.from_pretrained') \
             as mock_from_pretrained:
            mock_model = MagicMock(spec=BaseQuantForCausalLM)
            mock_from_pretrained.return_value = mock_model
            model = plugin.create_model(tmp_file.name)
            assert model is mock_model
            mock_from_pretrained.assert_called_once_with(tmp_file.name)
    except ImportError as e:
        # Handle case where mindformers or dependencies are not available
        assert "mindformers" in str(e).lower() or "mfmodel" in str(e).lower()
    finally:
        os.unlink(tmp_file.name)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_model_hub_plugin_create_model_invalid_path():
    """
    Feature: MFModelHubPlugin create_model invalid
    Description: Test create_model with invalid file path
    Expectation: Raises appropriate error for invalid path
    """
    plugin = MFModelHubPlugin()

    # Test with non-existent file
    with pytest.raises((FileNotFoundError, ValueError)):
        plugin.create_model("non_existent_file.yaml")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_model_hub_plugin_create_model_invalid_yaml():
    """
    Feature: MFModelHubPlugin create_model invalid
    Description: Test create_model with invalid YAML content
    Expectation: Raises appropriate error for invalid YAML
    """
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp_file:
        tmp_file.write(b"invalid: yaml: content")
        tmp_file.flush()

    try:
        plugin = MFModelHubPlugin()
        # This should raise some form of parsing error
        with pytest.raises((ValueError, RuntimeError)):
            plugin.create_model(tmp_file.name)
    finally:
        os.unlink(tmp_file.name)
