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
"""Unit tests for MindOneModel class."""

import os
import sys
import json
from unittest.mock import MagicMock

import pytest
import numpy as np
from mindspore import Tensor, Parameter

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../mindone")))
from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel


class DummyQuantModel(MindOneModel):
    """Minimal implementation for testing MindOneModel methods."""

    def __init__(self, model_path=None):
        """Initialize dummy model."""
        self.network = MagicMock()
        self.network.parameters_dict.return_value = {
            "layer.0.weight": Parameter(Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))),
            "layer.1.weight": Parameter(Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))),
        }
        self._original_sf_path = model_path

    def _network(self):
        """Return mock network."""
        return self.network

    def _transformer_layers(self):
        """Return mock transformer layers."""
        return (MagicMock,)

    # pylint: disable=unused-argument
    def forward(self, input_ids, max_new_tokens=1):
        """Return mock forward result."""
        return {"output": "mock_output"}

    def get_description_file(self):
        """Return mock quantization description."""
        return {
            "layer.0.weight": "W8A8",
            "layer.1.weight": "W4A16",
        }


class TestMindOneModel:
    """Test cases for MindOneModel class."""

    # pylint: disable=protected-access
    def setup_method(self):
        """Setup method to prepare test environment."""
        # Clear model registry before each test
        MindOneModel._model_registry.clear()

        # Create temporary directories
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_model_path = os.path.join(self.work_dir, "test_mindone_model")
        os.makedirs(self.test_model_path, exist_ok=True)

        # Create dummy model instance
        self.model = DummyQuantModel(self.test_model_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    # pylint: disable=protected-access
    def test_reg_model(self):
        """
        Feature: MindOneModel reg_model decorator.
        Description: Test model registration using decorator.
        Expectation: Model should be registered successfully.
        """
        @MindOneModel.reg_model("test_model")
        class TestModel(MindOneModel):
            pass

        assert "test_model" in MindOneModel._model_registry
        assert MindOneModel._model_registry["test_model"] == TestModel

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    # pylint: disable=protected-access
    def test_get_model_registry(self):
        """
        Feature: MindOneModel get_model_registry.
        Description: Test getting model registry.
        Expectation: Should return the model registry dictionary.
        """
        @MindOneModel.reg_model("registry_test")
        class RegistryTestModel(MindOneModel):
            pass

        registry = MindOneModel.get_model_registry()
        assert isinstance(registry, dict)
        assert "registry_test" in registry
        assert registry["registry_test"] == RegistryTestModel

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_success(self):
        """
        Feature: MindOneModel from_pretrained.
        Description: Test creating model from pretrained path.
        Expectation: Should create model instance successfully.
        """
        # Create config.json file
        config_data = {"model_type": "test_from_pretrained"}
        config_path = os.path.join(self.test_model_path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f)

        @MindOneModel.reg_model("test_from_pretrained")
        class TestFromPretrainedModel(MindOneModel):
            def __init__(self, model_path):
                self.network = MagicMock()
                self._original_sf_path = model_path

        model = MindOneModel.from_pretrained(self.test_model_path)
        assert isinstance(model, TestFromPretrainedModel)
