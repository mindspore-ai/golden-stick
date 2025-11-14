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
"""test BaseQuantForCausalLM."""


import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))

from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


class TestBaseQuantForCausalLM:
    """Test cases for BaseQuantForCausalLM class."""

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_abstract_methods_raise_not_implemented(self):
        """
        Feature: BaseQuantForCausalLM abstract methods.
        Description: Test that abstract methods raise NotImplementedError.
        Expectation: NotImplementedError is raised when calling abstract methods.
        """
        # Create an instance of the base class
        base_model = BaseQuantForCausalLM()

        # Test from_pretrained
        with pytest.raises(NotImplementedError):
            base_model.from_pretrained()

        # Test forward
        with pytest.raises(NotImplementedError):
            base_model.forward(None)

        # Test calibrate
        with pytest.raises(NotImplementedError):
            base_model.calibrate(None, None, None)

        # Test save_quantized
        with pytest.raises(NotImplementedError):
            base_model.save_quantized(None)

        # Test fake_quant
        with pytest.raises(NotImplementedError):
            base_model.fake_quant(None, None)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
