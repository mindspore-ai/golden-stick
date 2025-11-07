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
"""Test BaseQuantForCausalLM."""

import inspect

import pytest

from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


class TestBaseQuantForCausalLM:
    """Test cases for BaseQuantForCausalLM class - normal interface cases."""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_base_construct(self):
        """
        Feature: BaseQuantForCausalLM basic construction
        Description: Test BaseQuantForCausalLM default constructor and basic functionality
        Expectation: Model object created correctly, basic functionality works normally
        """
        # Verify default constructor
        base_model = BaseQuantForCausalLM()
        assert base_model is not None
        assert isinstance(base_model, BaseQuantForCausalLM)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_abstract_method_from_pretrained(self):
        """
        Feature: BaseQuantForCausalLM abstract method from_pretrained
        Description: Test from_pretrained abstract method interface
        Expectation: Abstract method defined correctly, interface specification meets expectations
        """
        # Verify that from_pretrained is an abstract method
        with pytest.raises(NotImplementedError):
            BaseQuantForCausalLM.from_pretrained()

        # Verify method signature (classmethod)
        assert hasattr(BaseQuantForCausalLM, 'from_pretrained')
        sig = inspect.signature(BaseQuantForCausalLM.from_pretrained)
        # Verify it accepts **kwargs
        assert 'kwargs' in sig.parameters
        # Verify it's a classmethod by checking the descriptor
        assert hasattr(BaseQuantForCausalLM.__dict__['from_pretrained'], '__func__')

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_abstract_method_forward(self):
        """
        Feature: BaseQuantForCausalLM abstract method forward
        Description: Test forward abstract method interface
        Expectation: Abstract method defined correctly, interface specification meets expectations
        """
        # Create an instance of the base class
        base_model = BaseQuantForCausalLM()

        # Verify that forward is an abstract method
        with pytest.raises(NotImplementedError):
            base_model.forward(None)

        # Verify method signature
        assert hasattr(base_model, 'forward')
        sig = inspect.signature(base_model.forward)
        # Verify it accepts input_ids and max_new_tokens
        assert 'input_ids' in sig.parameters
        assert 'max_new_tokens' in sig.parameters
        # Verify default value for max_new_tokens
        assert sig.parameters['max_new_tokens'].default == 1

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_abstract_method_calibrate(self):
        """
        Feature: BaseQuantForCausalLM abstract method calibrate
        Description: Test calibrate abstract method interface
        Expectation: Abstract method defined correctly, interface specification meets expectations
        """
        # Create an instance of the base class
        base_model = BaseQuantForCausalLM()

        # Verify that calibrate is an abstract method
        with pytest.raises(NotImplementedError):
            base_model.calibrate(None, None, None)

        # Verify method signature
        assert hasattr(base_model, 'calibrate')
        sig = inspect.signature(base_model.calibrate)
        # Verify it accepts ptq_config, layers_policy, datasets, and **kwargs
        assert 'ptq_config' in sig.parameters
        assert 'layers_policy' in sig.parameters
        assert 'datasets' in sig.parameters
        assert 'kwargs' in sig.parameters

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_abstract_method_save_quantized(self):
        """
        Feature: BaseQuantForCausalLM abstract method save_quantized
        Description: Test save_quantized abstract method interface
        Expectation: Abstract method defined correctly, interface specification meets expectations
        """
        # Create an instance of the base class
        base_model = BaseQuantForCausalLM()

        # Verify that save_quantized is an abstract method
        with pytest.raises(NotImplementedError):
            base_model.save_quantized(None)

        # Verify method signature
        assert hasattr(base_model, 'save_quantized')
        sig = inspect.signature(base_model.save_quantized)
        # Verify it accepts save_path
        assert 'save_path' in sig.parameters

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_abstract_method_fake_quant(self):
        """
        Feature: BaseQuantForCausalLM abstract method fake_quant
        Description: Test fake_quant abstract method interface
        Expectation: Abstract method defined correctly, interface specification meets expectations
        """
        # Create an instance of the base class
        base_model = BaseQuantForCausalLM()

        # Verify that fake_quant is an abstract method
        with pytest.raises(NotImplementedError):
            base_model.fake_quant(None, None)

        # Verify method signature
        assert hasattr(base_model, 'fake_quant')
        sig = inspect.signature(base_model.fake_quant)
        # Verify it accepts ptq_config, layers_policy, and quant_safetensors_path
        assert 'ptq_config' in sig.parameters
        assert 'layers_policy' in sig.parameters
        assert 'quant_safetensors_path' in sig.parameters
        # Verify default value for quant_safetensors_path
        assert sig.parameters['quant_safetensors_path'].default == ""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_abstract_method_call_error(self):
        """
        Feature: BaseQuantForCausalLM abstract method call error
        Description: Test that calling abstract methods directly raises NotImplementedError
        Expectation: NotImplementedError is raised when calling abstract methods directly
        """
        # Create an instance of the base class
        base_model = BaseQuantForCausalLM()

        # Test from_pretrained raises NotImplementedError
        with pytest.raises(NotImplementedError):
            BaseQuantForCausalLM.from_pretrained()

        # Test forward raises NotImplementedError
        with pytest.raises(NotImplementedError):
            base_model.forward(None)

        # Test calibrate raises NotImplementedError
        with pytest.raises(NotImplementedError):
            base_model.calibrate(None, None, None)

        # Test save_quantized raises NotImplementedError
        with pytest.raises(NotImplementedError):
            base_model.save_quantized(None)

        # Test fake_quant raises NotImplementedError
        with pytest.raises(NotImplementedError):
            base_model.fake_quant(None, None)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_method_signatures_consistency(self):
        """
        Feature: BaseQuantForCausalLM method signatures consistency
        Description: Test that all abstract methods have consistent and correct signatures
        Expectation: All method signatures are correct and consistent
        """
        # Verify from_pretrained signature
        sig_from_pretrained = inspect.signature(BaseQuantForCausalLM.from_pretrained)
        assert 'kwargs' in sig_from_pretrained.parameters

        # Verify forward signature
        sig_forward = inspect.signature(BaseQuantForCausalLM.forward)
        assert 'input_ids' in sig_forward.parameters
        assert 'max_new_tokens' in sig_forward.parameters
        assert sig_forward.parameters['max_new_tokens'].default == 1

        # Verify calibrate signature
        sig_calibrate = inspect.signature(BaseQuantForCausalLM.calibrate)
        assert 'ptq_config' in sig_calibrate.parameters
        assert 'layers_policy' in sig_calibrate.parameters
        assert 'datasets' in sig_calibrate.parameters
        assert 'kwargs' in sig_calibrate.parameters

        # Verify save_quantized signature
        sig_save_quantized = inspect.signature(BaseQuantForCausalLM.save_quantized)
        assert 'save_path' in sig_save_quantized.parameters

        # Verify fake_quant signature
        sig_fake_quant = inspect.signature(BaseQuantForCausalLM.fake_quant)
        assert 'ptq_config' in sig_fake_quant.parameters
        assert 'layers_policy' in sig_fake_quant.parameters
        assert 'quant_safetensors_path' in sig_fake_quant.parameters
        assert sig_fake_quant.parameters['quant_safetensors_path'].default == ""

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_concrete_implementation(self):
        """
        Feature: BaseQuantForCausalLM concrete implementation
        Description: Test that concrete plugin implementations work correctly
        Expectation: Concrete plugin can be instantiated and methods work correctly
        """
        # Create a concrete model class
        class ConcreteModel(BaseQuantForCausalLM):
            """Concrete model implementation for testing."""
            @classmethod
            def from_pretrained(cls, **kwargs):
                return cls()

            def forward(self, input_ids, max_new_tokens=1):
                return "forward_result"

            def calibrate(self, ptq_config, layers_policy, datasets, **kwargs):
                return "calibrate_result"

            def save_quantized(self, save_path):
                return "save_result"

            def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
                return "fake_quant_result"

        # Test concrete model
        model = ConcreteModel()
        assert isinstance(model, BaseQuantForCausalLM)
        assert isinstance(model, ConcreteModel)

        # Test create_model
        result = model.forward(None)
        assert result == "forward_result"

        result = model.calibrate(None, None, None)
        assert result == "calibrate_result"

        result = model.save_quantized(None)
        assert result == "save_result"

        result = model.fake_quant(None, None)
        assert result == "fake_quant_result"

        # Test polymorphism - classmethod
        instance = ConcreteModel.from_pretrained()
        assert isinstance(instance, ConcreteModel)
        assert isinstance(instance, BaseQuantForCausalLM)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_inheritance_system(self):
        """
        Feature: BaseQuantForCausalLM inheritance system
        Description: Test subclass implementation, method override, and polymorphism
        Expectation: Inheritance system correct, polymorphism functionality normal
        """
        # Test subclass inheritance
        class SubModel(BaseQuantForCausalLM):
            """Subclass model implementation."""
            def __init__(self):
                super().__init__()
                self.custom_init = True

            @classmethod
            def from_pretrained(cls, **kwargs):
                return cls()

            def forward(self, input_ids, max_new_tokens=1):
                return "sub_forward_result"

            def calibrate(self, ptq_config, layers_policy, datasets, **kwargs):
                return "sub_calibrate_result"

            def save_quantized(self, save_path):
                return "sub_save_result"

            def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
                return "sub_fake_quant_result"

        # Verify subclass can be instantiated
        model = SubModel()
        assert isinstance(model, BaseQuantForCausalLM)
        assert isinstance(model, SubModel)
        assert model.custom_init is True

        # Test method override
        result = model.forward(None)
        assert result == "sub_forward_result"

        result = model.calibrate(None, None, None)
        assert result == "sub_calibrate_result"

        result = model.save_quantized(None)
        assert result == "sub_save_result"

        result = model.fake_quant(None, None)
        assert result == "sub_fake_quant_result"

        # Test polymorphism - classmethod
        instance = SubModel.from_pretrained()
        assert isinstance(instance, SubModel)
        assert isinstance(instance, BaseQuantForCausalLM)

        # Test that base class methods still raise NotImplementedError if not overridden
        class IncompleteModel(BaseQuantForCausalLM):  # pylint: disable=abstract-method
            """Incomplete model implementation."""

        incomplete = IncompleteModel()
        with pytest.raises(NotImplementedError):
            incomplete.forward(None)

        with pytest.raises(NotImplementedError):
            incomplete.calibrate(None, None, None)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_partial_implementation_error(self):
        """
        Feature: BaseQuantForCausalLM partial implementation error
        Description: Test that partial implementation still raises NotImplementedError for unimplemented methods
        Expectation: Unimplemented methods still raise NotImplementedError
        """
        # Create a partially implemented model
        class PartialModel(BaseQuantForCausalLM):  # pylint: disable=abstract-method
            """Partially implemented model."""
            @classmethod
            def from_pretrained(cls, **kwargs):
                return cls()

            def forward(self, input_ids, max_new_tokens=1):
                return "forward_result"

            # calibrate, save_quantized, fake_quant not implemented

        model = PartialModel()

        # Test implemented methods work
        result = model.forward(None)
        assert result == "forward_result"

        instance = PartialModel.from_pretrained()
        assert isinstance(instance, PartialModel)

        # Test unimplemented methods raise NotImplementedError
        with pytest.raises(NotImplementedError):
            model.calibrate(None, None, None)

        with pytest.raises(NotImplementedError):
            model.save_quantized(None)

        with pytest.raises(NotImplementedError):
            model.fake_quant(None, None)
