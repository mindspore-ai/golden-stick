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
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))

from mindspore_gs.ptq import AutoQuantForCausalLM
from mindspore_gs.ptq import BaseQuantForCausalLM


class TestAutoQuantForCausalLM:
    """Test cases for AutoQuantForCausalLM class."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_from_pretrained_success(self):
        """
        Feature: AutoQuantForCausalLM from_pretrained with valid model hub.
        Description: Test successful model creation from pretrained path when a valid model hub is registered.
        Expectation: Model instance is created successfully.
        """
        # Mock a model hub that can create a model
        mock_model = MagicMock()
        mock_model_hub = MagicMock()
        mock_model_hub.from_pretrained.return_value = mock_model

        # Register the mock model hub
        with patch.object(BaseQuantForCausalLM, 'get_model_hub_registry',
                          return_value={'test_hub': mock_model_hub}):
            result = AutoQuantForCausalLM.from_pretrained("test_path")
            assert result == mock_model
            mock_model_hub.from_pretrained.assert_called_once_with("test_path")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_from_pretrained_with_value_error(self):
        """
        Feature: AutoQuantForCausalLM from_pretrained with ValueError handling.
        Description: Test model creation when first hub raises ValueError but second succeeds.
        Expectation: Model instance is created from the second hub.
        """
        # Mock two model hubs, first raises ValueError, second succeeds
        mock_model = MagicMock()
        mock_model_hub1 = MagicMock()
        mock_model_hub1.from_pretrained.side_effect = ValueError("Not supported")
        mock_model_hub2 = MagicMock()
        mock_model_hub2.from_pretrained.return_value = mock_model

        # Register the mock model hubs
        with patch.object(BaseQuantForCausalLM, 'get_model_hub_registry',
                          return_value={'hub1': mock_model_hub1, 'hub2': mock_model_hub2}):
            result = AutoQuantForCausalLM.from_pretrained("test_path")
            assert result == mock_model
            mock_model_hub1.from_pretrained.assert_called_once_with("test_path")
            mock_model_hub2.from_pretrained.assert_called_once_with("test_path")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_from_pretrained_all_fail(self):
        """
        Feature: AutoQuantForCausalLM from_pretrained with all hubs failing.
        Description: Test model creation when all hubs raise ValueError.
        Expectation: No exception is raised, method handles gracefully.
        """
        # Mock model hubs that all raise ValueError
        mock_model_hub1 = MagicMock()
        mock_model_hub1.from_pretrained.side_effect = ValueError("Not supported")
        mock_model_hub2 = MagicMock()
        mock_model_hub2.from_pretrained.side_effect = ValueError("Not supported")

        # Register the mock model hubs
        with patch.object(BaseQuantForCausalLM, 'get_model_hub_registry',
                          return_value={'hub1': mock_model_hub1, 'hub2': mock_model_hub2}):
            # This should not raise an exception but should handle gracefully
            # In the current implementation, it would return None implicitly
            result = AutoQuantForCausalLM.from_pretrained("test_path")
            assert result is None

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_from_pretrained_no_hubs(self):
        """
        Feature: AutoQuantForCausalLM from_pretrained with no registered hubs.
        Description: Test model creation when no hubs are registered.
        Expectation: No exception is raised, method handles gracefully.
        """
        # Mock empty registry
        with patch.object(BaseQuantForCausalLM, 'get_model_hub_registry',
                          return_value={}):
            # This should not raise an exception but should handle gracefully
            result = AutoQuantForCausalLM.from_pretrained("test_path")
            assert result is None
