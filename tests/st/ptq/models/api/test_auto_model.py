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
"""test AutoQuantForCausalLM."""

import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindone")))

from mindspore_gs.ptq import AutoQuantForCausalLM


class TestAutoQuantForCausalLM:
    """Test cases for AutoQuantForCausalLM class."""

    def setup_method(self):
        """Setup method to prepare test environment."""
        self.work_dir = tempfile.mkdtemp()
        self.yaml_path = os.path.join(self.work_dir, "test_model.yaml")
        self.model_dir = os.path.join(self.work_dir, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @patch('mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindformers_plugin')
    def test_from_pretrained_yaml_file(self, mock_load_plugin):
        """
        Feature: AutoQuantForCausalLM.from_pretrained with yaml file.
        Description: Test from_pretrained with yaml file path.
        Expectation: Should load mindformers plugin and create model.
        """
        # Mock registry and model class
        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_load_plugin.return_value = mock_model_class

        result = AutoQuantForCausalLM.from_pretrained(self.yaml_path)

        # Verify plugin loading
        mock_load_plugin.assert_called_once()

        # Verify model creation
        mock_model_class.from_pretrained.assert_called_once_with(self.yaml_path)
        assert result == mock_model

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    @patch('mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindone_plugin')
    def test_from_pretrained_directory(self, mock_load_plugin):
        """
        Feature: AutoQuantForCausalLM.from_pretrained with directory path.
        Description: Test from_pretrained with directory path.
        Expectation: Should load mindone plugin and create model.
        """
        # Create config.json in directory
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({"model_type": "glm4v"}, f)

        # Mock registry and model class
        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        mock_load_plugin.return_value = mock_model_class

        result = AutoQuantForCausalLM.from_pretrained(self.model_dir)

        # Verify plugin loading
        mock_load_plugin.assert_called_once()

        # Verify model creation
        mock_model_class.from_pretrained.assert_called_once_with(self.model_dir)
        assert result == mock_model

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_from_pretrained_unsupported_path(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained with unsupported path.
        Description: Test from_pretrained with unsupported path type.
        Expectation: Should raise ValueError.
        """
        unsupported_path = "not_a_file_or_directory"

        with pytest.raises(ValueError, match="Unsupported model type"):
            AutoQuantForCausalLM.from_pretrained(unsupported_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_load_mindformers_plugin_success(self):
        """
        Feature: AutoQuantForCausalLM._load_mindformers_plugin
        Description: Test _load_mindformers_plugin when import succeeds.
        Expectation: Should successfully import MODEL_REGISTRY and return it.
        """
        from mindspore_gs.ptq.ptq.algorithms.anti_outliers import LinearSmoothQuant
        from mindspore_gs.ptq.ptq.algorithms.clipper import LinearClipper
        from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
        # Call the actual function to test import
        # pylint: disable=protected-access
        model_hub = AutoQuantForCausalLM._load_mindformers_plugin()
        model_registry = model_hub.get_model_registry()

        model_list = ['qwen3', 'qwen3_moe', 'deepseek_v3', 'telechat2']
        assert isinstance(model_registry, dict)
        assert len(model_registry) == len(model_list)
        for model_name in model_list:
            assert model_name in model_registry
        assert LinearSmoothQuant.linear_map is not None
        assert LinearClipper.linear_map is not None
        assert Quantizer.layer_map is not None

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_load_mindone_plugin_success(self):
        """
        Feature: AutoQuantForCausalLM._load_mindone_plugin
        Description: Test _load_mindone_plugin when import succeeds.
        Expectation: Should successfully import MODEL_REGISTRY and return it.
        """
        # Call the actual function to test import
        # pylint: disable=protected-access
        model_hub = AutoQuantForCausalLM._load_mindone_plugin()
        model_registry = model_hub.get_model_registry()

        model_list = ['glm4v']
        assert isinstance(model_registry, dict)
        assert len(model_registry) == len(model_list)
        for model_name in model_list:
            assert model_name in model_registry
