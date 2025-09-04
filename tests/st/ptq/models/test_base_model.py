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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))

from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


class TestBaseQuantForCausalLM:
    """Test cases for BaseQuantForCausalLM class."""

    def setup_method(self):
        """Setup method to clear registry before each test."""
        BaseQuantForCausalLM._model_hub_registry.clear()  # pylint: disable=protected-access

    def teardown_method(self):
        """Teardown method to clear registry after each test."""
        BaseQuantForCausalLM._model_hub_registry.clear()  # pylint: disable=protected-access

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_reg_model_hub_decorator(self):
        """
        Feature: BaseQuantForCausalLM reg_model_hub decorator.
        Description: Test the reg_model_hub decorator with alias.
        Expectation: Class is registered with the specified alias.
        """
        # Create a test class
        class TestModel:
            pass

        # Register using decorator with alias
        decorated_class = BaseQuantForCausalLM.reg_model_hub("test_alias")(TestModel)

        # Check that the class is registered
        assert "test_alias" in BaseQuantForCausalLM._model_hub_registry  # pylint: disable=protected-access
        assert BaseQuantForCausalLM._model_hub_registry["test_alias"] == TestModel  # pylint: disable=protected-access
        # Check that the decorator returns the original class
        assert decorated_class == TestModel

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_reg_model_hub_decorator_without_alias(self):
        """
        Feature: BaseQuantForCausalLM reg_model_hub decorator without alias.
        Description: Test the reg_model_hub decorator without alias.
        Expectation: Class is registered with its class name.
        """
        # Create a test class
        class TestModelWithoutAlias:
            pass

        # Register using decorator without alias
        decorated_class = BaseQuantForCausalLM.reg_model_hub()(TestModelWithoutAlias)

        # Check that the class is registered with its name
        assert "TestModelWithoutAlias" in BaseQuantForCausalLM._model_hub_registry  # pylint: disable=protected-access
        assert BaseQuantForCausalLM._model_hub_registry["TestModelWithoutAlias"] == TestModelWithoutAlias  # pylint: disable=protected-access
        # Check that the decorator returns the original class
        assert decorated_class == TestModelWithoutAlias

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_reg_model_hub_duplicate_registration(self):
        """
        Feature: BaseQuantForCausalLM reg_model_hub duplicate registration.
        Description: Test registering the same name twice raises RuntimeError.
        Expectation: RuntimeError is raised when attempting to register duplicate name.
        """
        class TestModel1:
            pass

        class TestModel2:
            pass

        # Register the first model
        BaseQuantForCausalLM.reg_model_hub("test_model")(TestModel1)

        # Attempt to register a second model with the same name
        with pytest.raises(RuntimeError) as exc_info:
            BaseQuantForCausalLM.reg_model_hub("test_model")(TestModel2)

        assert "Duplicated model-hub reg" in str(exc_info.value)
        assert "test_model" in str(exc_info.value)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_get_model_hub_registry(self):
        """
        Feature: BaseQuantForCausalLM get_model_hub_registry.
        Description: Test getting the model hub registry.
        Expectation: Returns dictionary with registered models.
        """
        # Register some models
        class ModelA:
            pass

        class ModelB:
            pass

        BaseQuantForCausalLM.reg_model_hub("model_a")(ModelA)
        BaseQuantForCausalLM.reg_model_hub("model_b")(ModelB)

        # Get the registry
        registry = BaseQuantForCausalLM.get_model_hub_registry()

        # Check that it contains the registered models
        assert isinstance(registry, dict)
        assert "model_a" in registry
        assert "model_b" in registry
        assert registry["model_a"] == ModelA
        assert registry["model_b"] == ModelB

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
