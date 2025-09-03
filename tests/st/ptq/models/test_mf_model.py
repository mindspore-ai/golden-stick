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
"""test BaseQuantForCausalLMImpl."""


import os
import sys
from unittest.mock import MagicMock
import pytest

from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))


class DummyQuantModel(MFModel):
    """Minimal implementation, just enough to let calibrate
    interface run successfully."""
    def __init__(self):
        pass

    def _network(self):
        return MagicMock()

    def _transformer_layers(self):
        return (MagicMock,)


class TestMFModel:
    """Test cases for MFModel class."""

    @pytest.fixture
    def model(self):
        """return dummy quant model."""
        return DummyQuantModel()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_input_validation(self, model):
        """
        Feature: MFModel from_pretrained interface.
        Description: Test the input parameters is valid for from_pretrained interface.
        Expectation: Raise error for invalid input parameters.
        """
        with pytest.raises(ValueError):
            yaml_path = "./test.yaml"
            model.from_pretrained(yaml_path)
        with pytest.raises(ValueError):
            yaml_path = "./test.json"
            model.from_pretrained(yaml_path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_calibrate_input_validation(self, model):
        """
        Feature: BaseQuantForCausalLMImpl calibrate interface.
        Description: Test the input parameters is valid for calibrate interface.
        Expectation: Raise error for invalid input parameters.
        """
        with pytest.raises(TypeError):
            model.calibrate(ptq_config=123,
                            layers_policy=MagicMock(),
                            datasets=MagicMock())

        with pytest.raises(TypeError):
            model.calibrate(ptq_config={},
                            layers_policy=123,
                            datasets=MagicMock())

        with pytest.raises(TypeError):
            model.calibrate(ptq_config={},
                            layers_policy=MagicMock(),
                            datasets=123)
