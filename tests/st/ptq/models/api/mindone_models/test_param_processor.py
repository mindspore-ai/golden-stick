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
"""Unit tests for ParamProcessor class."""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
from mindspore import Tensor, Parameter, dtype

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.utils import QuantType
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../mindone")))
from mindspore_gs.ptq.models.mindone_models.param_processor import ParamProcessor


class TestParamProcessor:
    """Test cases for ParamProcessor class."""

    def setup_method(self):
        """Setup method to prepare test environment."""
        # Create sample parameter dictionary
        self.sample_param_dict = {
            "layer.0.weight": Parameter(Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=dtype.int8)),
            "layer.0.weight_scale": Parameter(Tensor(np.array([[0.1, 0.2], [0.3, 0.4]]), dtype=dtype.bfloat16)),
            "layer.1.weight": Parameter(Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), dtype=dtype.int8)),
            "layer.2.weight": Parameter(Tensor(np.array([[9.0, 10.0], [11.0, 12.0]]), dtype=dtype.int8)),
        }

        # Create sample quantization description
        self.sample_quant_desc = {
            "layer.0.weight": QuantType.W4A16.value,
            "layer.0.weight_scale": QuantType.W4A16.value,
            "layer.1.weight": QuantType.W8A16.value,
            "layer.2.weight": QuantType.W4A8_DYNAMIC.value,
        }

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_deploy_with_none_backend(self):
        """
        Feature: ParamProcessor deploy with NONE backend.
        Description: Test deploy method with BackendTarget.NONE.
        Expectation: Should return original param_dict without modification.
        """
        processor = ParamProcessor(BackendTarget.NONE, self.sample_quant_desc)

        result = processor.deploy(self.sample_param_dict)

        # Should return the same dictionary (reference equality)
        assert result is self.sample_param_dict
        assert len(result) == len(self.sample_param_dict)
        assert result == self.sample_param_dict

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_deploy_with_unsupported_backend(self):
        """
        Feature: ParamProcessor deploy with unsupported backend.
        Description: Test deploy method with unsupported backend type.
        Expectation: Should raise ValueError.
        """
        # Create a mock unsupported backend
        unsupported_backend = MagicMock()

        processor = ParamProcessor(unsupported_backend, self.sample_quant_desc)

        with pytest.raises(ValueError, match="Unsupported backend"):
            processor.deploy(self.sample_param_dict)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_deploy_with_ascend_backend_w4ax(self):
        """
        Feature: ParamProcessor deploy with ASCEND backend and W4Ax type.
        Description: Test deploy method processes W4Ax quantized weight parameters.
        Expectation: Should process weight parameters ending with .weight.
        """
        processor = ParamProcessor(BackendTarget.ASCEND, self.sample_quant_desc)
        result = processor.deploy(self.sample_param_dict)

        # Verify result is modified
        layer_0_weight = result["layer.0.weight"]
        assert isinstance(layer_0_weight, Parameter)
        assert layer_0_weight.dtype == dtype.qint4x2
        assert layer_0_weight.shape == (1, 2)

        layer_0_weight_scale = result["layer.0.weight_scale"]
        assert isinstance(layer_0_weight_scale, Parameter)
        assert layer_0_weight_scale.dtype == dtype.bfloat16
        assert layer_0_weight_scale.shape == (2, 2)

        layer_1_weight = result["layer.1.weight"]
        assert isinstance(layer_1_weight, Parameter)
        assert layer_1_weight.dtype == dtype.int8
        assert layer_1_weight.shape == (2, 2)

        layer_2_weight = result["layer.2.weight"]
        assert isinstance(layer_2_weight, Parameter)
        assert layer_2_weight.dtype == dtype.qint4x2
        assert layer_2_weight.shape == (1, 2)
