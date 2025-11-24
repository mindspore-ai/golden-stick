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
"""Test AutoQuantForCausalLM."""

import sys
import os
import shutil
import tempfile
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))
from mindspore_gs.ptq.models.auto_model import AutoQuantForCausalLM



class TestAutoQuantForCausalLM:
    """Test cases for AutoQuantForCausalLM class - normal interface cases."""

    def setup_method(self):
        """Setup method to prepare test environment."""
        self.work_dir = tempfile.mkdtemp()
        self.yaml_path = os.path.join(self.work_dir, "test_model.yaml")

    def teardown_method(self):
        """Teardown method to clean up test environment."""
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_from_pretrained_normal_yaml_path(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained normal case with yaml path
        Description: Test from_pretrained method normal call and model creation with yaml file path
        Expectation: Method correctly called, model successfully created, returns correct model instance
        """
        # Create a yaml file
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("trainer:\n")
            f.write("  model_name: 'qwen3'\n")
            f.write("  type: CausalLanguageModelingTrainer\n")

        with patch('mindspore_gs.ptq.models.mindformers_models.qwen3.QWen3.__init__', return_value=None) \
            as mock_qwen3_model_init:
            _ = AutoQuantForCausalLM.from_pretrained(self.yaml_path)
            mock_qwen3_model_init.assert_called_once()
