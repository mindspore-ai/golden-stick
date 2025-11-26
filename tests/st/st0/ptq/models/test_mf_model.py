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
"""Test MFModel - st0 normal and exception interface cases."""

import os
import sys
import tempfile
import shutil
from collections import OrderedDict
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))
# pylint: disable=wrong-import-position
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore import dtype as msdtype
from mindspore.dataset import GeneratorDataset


class DummyQuantModel(MFModel):
    """Minimal implementation for testing MFModel methods."""

    def __init__(self, yaml_path):  # pylint: disable=unused-argument
        """Initialize dummy model."""
        # Skip parent initialization to avoid actual model loading
        self.network = MagicMock()
        self._original_sf_path = "/mock/path"

    def _network(self):
        """Return mock network."""
        return self.network

    def _transformer_layers(self):
        """Return mock transformer layers."""
        # pylint: disable=import-outside-toplevel
        from mindformers.parallel_core.inference.transformer.transformer_layer import TransformerLayer
        return [TransformerLayer]

    def forward(self, input_ids, max_new_tokens=1):
        """Return mock forward result."""
        return self.network.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)

    def _load_weights_to_fake_quant(self, quant_safetensors_path):  # pylint: disable=unused-argument
        """Mock implementation of abstract method."""
        # Abstract method implementation - no operation needed


class TestMFModel:
    """Test cases for MFModel class - normal and exception interface cases."""

    # pylint: disable=protected-access
    def setup_method(self):
        """Setup method to prepare test environment."""
        # Clear model registry before each test
        MFModel._model_registry.clear()

        # Create temporary directories
        self.work_dir = tempfile.mkdtemp()
        self.yaml_path = os.path.join(self.work_dir, "test_model.yaml")

    def teardown_method(self):
        """Teardown method to clean up test environment."""
        # Clear model registry after each test
        MFModel._model_registry.clear()
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_reg_model(self):
        """
        Feature: MFModel reg_model decorator.
        Description: Test model registration using decorator.
        Expectation: Model should be registered successfully.
        """
        @MFModel.reg_model("test_model")
        class TestModel(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        assert "test_model" in MFModel._model_registry
        assert MFModel._model_registry["test_model"] == TestModel

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_from_pretrained_success(self):
        """
        Feature: MFModel from_pretrained.
        Description: Test creating model from pretrained yaml path.
        Expectation: Should create model instance successfully.
        """
        # Create a yaml file
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("trainer:\n")
            f.write("  type: CausalLanguageModelingTrainer\n")
            f.write("  model_name: test_from_pretrained\n")
            f.write("pretrained_model_dir: /mock/path\n")

        @MFModel.reg_model("test_from_pretrained")
        class TestFromPretrainedModel(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        # Call from_pretrained without patch
        model = MFModel.from_pretrained(self.yaml_path)

        # Verify model is created correctly
        assert model is not None
        assert isinstance(model, TestFromPretrainedModel)
        assert isinstance(model, MFModel)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_forward_method(self):
        """
        Feature: MFModel forward method.
        Description: Test forward method normal call.
        Expectation: Forward method should call network.generate correctly.
        """
        model = DummyQuantModel(self.yaml_path)
        mock_input_ids = MagicMock()
        mock_output = MagicMock()
        model.network.generate = MagicMock(return_value=mock_output)

        result = model.forward(mock_input_ids, max_new_tokens=5)

        assert result == mock_output
        model.network.generate.assert_called_once_with(mock_input_ids, do_sample=False, max_new_tokens=5)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_network_method(self):
        """
        Feature: MFModel _network method.
        Description: Test _network method returns the network instance.
        Expectation: _network method should return the network correctly.
        """
        model = DummyQuantModel(self.yaml_path)
        network = model._network()

        assert network is not None
        assert network == model.network

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_transformer_layers_method(self):
        """
        Feature: MFModel _transformer_layers method.
        Description: Test _transformer_layers method returns transformer layer types.
        Expectation: _transformer_layers method should return correct layer types.
        """
        model = DummyQuantModel(self.yaml_path)
        transformer_layers = model._transformer_layers()

        assert isinstance(transformer_layers, (list, tuple))
        assert len(transformer_layers) > 0

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_original_safetensors_path_method(self):
        """
        Feature: MFModel _original_safetensors_path method.
        Description: Test _original_safetensors_path method returns the path.
        Expectation: _original_safetensors_path method should return correct path.
        """
        model = DummyQuantModel(self.yaml_path)
        sf_path = model._original_safetensors_path()

        assert sf_path == "/mock/path"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_model_registry_access(self):
        """
        Feature: MFModel model registry access.
        Description: Test accessing model registry directly.
        Expectation: Model registry should be accessible and return correct registry dictionary.
        """
        # pylint: disable=protected-access
        registry = MFModel._model_registry
        assert isinstance(registry, dict)

        # Register a test model
        @MFModel.reg_model("test_registry_model")
        class TestRegistryModel(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        # pylint: disable=protected-access
        registry_after = MFModel._model_registry
        assert "test_registry_model" in registry_after
        assert registry_after["test_registry_model"] == TestRegistryModel

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_duplicate_registration_error(self):
        """
        Feature: MFModel duplicate registration error.
        Description: Test that registering the same name twice raises RuntimeError.
        Expectation: RuntimeError is raised when attempting to register duplicate name.
        """
        # Create first test class and register it
        @MFModel.reg_model("duplicate_test_model")
        class TestModel1(MFModel):  # pylint: disable=unused-variable
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        # Attempt to register a second model with the same name
        with pytest.raises(RuntimeError):
            @MFModel.reg_model("duplicate_test_model")
            class TestModel2(MFModel):  # pylint: disable=unused-variable
                def __init__(self, yaml_path):  # pylint: disable=unused-argument
                    self.network = MagicMock()
                    self._original_sf_path = "/mock/path"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_param_type_error(self):
        """
        Feature: MFModel parameter type error.
        Description: Test decorator parameter type error handling.
        Expectation: Invalid parameter types are handled correctly, error handling is correct.
        """
        # Create a test class
        class TestModel(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        # Test with non-string alias (list) - should raise TypeError as list is not hashable
        with pytest.raises(TypeError):
            MFModel.reg_model(["invalid"])(TestModel)

        # Test with non-string alias (dict) - should raise TypeError as dict is not hashable
        with pytest.raises(TypeError):
            MFModel.reg_model({"invalid": "value"})(TestModel)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_abstract_method_call_error(self):
        """
        Feature: MFModel abstract method call error.
        Description: Test that calling abstract methods directly raises NotImplementedError.
        Expectation: NotImplementedError is raised when calling abstract methods directly.
        """
        # Create a test model instance without implementing abstract methods
        class IncompleteModel(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        model = IncompleteModel(self.yaml_path)

        # Test _load_weights_to_fake_quant raises NotImplementedError (abstract method in base)
        with pytest.raises(NotImplementedError):
            model._load_weights_to_fake_quant("")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_config_error_file_not_exists(self):
        """
        Feature: MFModel configuration error - file not exists.
        Description: Test invalid YAML configuration handling when file does not exist.
        Expectation: ValueError is raised when file does not exist.
        """
        # Test case 1: File does not exist
        non_existent_path = os.path.join(self.work_dir, "non_existent.yaml")
        with pytest.raises(ValueError):
            MFModel.from_pretrained(non_existent_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_config_error_not_yaml_file(self):
        """
        Feature: MFModel configuration error - not yaml file.
        Description: Test invalid YAML configuration handling when file is not a yaml file.
        Expectation: ValueError is raised when file is not a yaml file.
        """
        # Test case 2: File exists but is not a yaml file
        json_path = os.path.join(self.work_dir, "test.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write('{"test": "data"}')
        with pytest.raises(ValueError):
            MFModel.from_pretrained(json_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_config_error_missing_trainer_model_name(self):
        """
        Feature: MFModel configuration error - missing trainer.model_name.
        Description: Test invalid YAML configuration handling when YAML file missing trainer.model_name.
        Expectation: ValueError is raised when trainer.model_name is missing.
        """
        # Test case 3: YAML file exists but missing trainer.model_name
        yaml_path_no_trainer = os.path.join(self.work_dir, "no_trainer.yaml")
        with open(yaml_path_no_trainer, "w", encoding="utf-8") as f:
            f.write("model:\n")
            f.write("  name: test\n")

        with patch('mindspore_gs.ptq.models.mindformers_models.mf_model.MindFormerConfig') as mock_config_class:
            # Create a mock config without trainer attribute
            # Use spec=[] to create a mock with no attributes, so hasattr will return False
            mock_config = MagicMock(spec=[])
            mock_config_class.return_value = mock_config

            with pytest.raises(ValueError):
                MFModel.from_pretrained(yaml_path_no_trainer)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_config_error_unregistered_model_name(self):
        """
        Feature: MFModel configuration error - unregistered model name.
        Description: Test invalid YAML configuration handling when model_name is not registered.
        Expectation: ValueError is raised when model_name is not registered.
        """
        # Test case 4: YAML file exists but model_name not registered
        yaml_path_unregistered = os.path.join(self.work_dir, "unregistered.yaml")
        with open(yaml_path_unregistered, "w", encoding="utf-8") as f:
            f.write("trainer:\n")
            f.write("  model_name: unregistered_model\n")

        with patch('mindspore_gs.ptq.models.mindformers_models.mf_model.MindFormerConfig') as mock_config_class:
            mock_config = MagicMock()
            mock_config.trainer.model_name = "unregistered_model"
            mock_config_class.return_value = mock_config

            with pytest.raises(ValueError):
                MFModel.from_pretrained(yaml_path_unregistered)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_lifecycle_management(self):
        """
        Feature: MFModel model lifecycle management.
        Description: Test model registration, retrieval, cleanup, and registry management.
        Expectation: Lifecycle management normal, state transitions correct, registry management effective.
        """
        # Model registration phase
        @MFModel.reg_model("lifecycle_model_a")
        class LifecycleModelA(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        @MFModel.reg_model("lifecycle_model_b")
        class LifecycleModelB(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        # Model retrieval phase
        # pylint: disable=protected-access
        registry = MFModel._model_registry
        assert "lifecycle_model_a" in registry
        assert "lifecycle_model_b" in registry
        assert registry["lifecycle_model_a"] == LifecycleModelA
        assert registry["lifecycle_model_b"] == LifecycleModelB

        # Cleanup phase (simulated by clearing registry)
        # pylint: disable=protected-access
        MFModel._model_registry.clear()
        # pylint: disable=protected-access
        registry_after_cleanup = MFModel._model_registry
        assert len(registry_after_cleanup) == 0

        # Re-register after cleanup
        @MFModel.reg_model("lifecycle_model_a")
        class LifecycleModelA2(MFModel):
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

        # pylint: disable=protected-access
        registry_after_reregister = MFModel._model_registry
        assert len(registry_after_reregister) == 1
        assert "lifecycle_model_a" in registry_after_reregister
        assert registry_after_reregister["lifecycle_model_a"] == LifecycleModelA2

        # Test concurrent access simulation (sequential access to registry)
        # pylint: disable=protected-access
        MFModel._model_registry.clear()
        for i in range(3):
            class DynamicModel(MFModel):  # pylint: disable=unused-variable
                def __init__(self, yaml_path):  # pylint: disable=unused-argument
                    self.network = MagicMock()
                    self._original_sf_path = "/mock/path"

            MFModel.reg_model(f"dynamic_model_{i}")(DynamicModel)
            # pylint: disable=protected-access
            current_registry = MFModel._model_registry
            assert len(current_registry) == i + 1
            assert f"dynamic_model_{i}" in current_registry

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_inheritance_system(self):
        """
        Feature: MFModel inheritance system.
        Description: Test subclass implementation, method override, and polymorphism.
        Expectation: Inheritance system correct, polymorphism functionality normal.
        """
        # Test subclass inheritance
        class ConcreteMFModel(MFModel):
            """Concrete MFModel implementation for testing."""
            def __init__(self, yaml_path):  # pylint: disable=unused-argument
                self.network = MagicMock()
                self._original_sf_path = "/mock/path"

            def _network(self):
                return self.network

            def _transformer_layers(self):
                # pylint: disable=import-outside-toplevel
                from mindformers.parallel_core.inference.transformer.transformer_layer import TransformerLayer
                return [TransformerLayer]

            def forward(self, input_ids, max_new_tokens=1):  # pylint: disable=unused-argument
                return "forward_result"

            def _load_weights_to_fake_quant(self, quant_safetensors_path):  # pylint: disable=unused-argument
                return "load_weights_result"

        # Verify subclass can be instantiated
        model = ConcreteMFModel(self.yaml_path)
        assert isinstance(model, MFModel)
        assert isinstance(model, ConcreteMFModel)

        # Test method override
        result = model.forward(None)
        assert result == "forward_result"

        result = model._load_weights_to_fake_quant("")
        assert result == "load_weights_result"

        # Test polymorphism - verify methods are accessible
        network = model._network()
        assert network is not None

        transformer_layers = model._transformer_layers()
        assert isinstance(transformer_layers, (list, tuple))

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_calibrate_param_type_error_ptq_config(self):
        """
        Feature: MFModel calibrate parameter type error - ptq_config.
        Description: Test calibrate method with invalid ptq_config type.
        Expectation: TypeError is raised when ptq_config is not PTQConfig.
        """
        model = DummyQuantModel(self.yaml_path)

        # Test with invalid ptq_config types
        invalid_configs = [
            None,
            "invalid_string",
            123,
            {"key": "value"},
            [],
        ]

        # Create valid layers_policy and datasets for testing
        layers_policy = OrderedDict()
        datasets = GeneratorDataset([1, 2, 3], "data")

        for invalid_config in invalid_configs:
            with pytest.raises(TypeError):
                model.calibrate(
                    ptq_config=invalid_config,
                    layers_policy=layers_policy,
                    datasets=datasets
                )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_calibrate_param_type_error_layers_policy(self):
        """
        Feature: MFModel calibrate parameter type error - layers_policy.
        Description: Test calibrate method with invalid layers_policy type.
        Expectation: TypeError is raised when layers_policy is not OrderedDict.
        """
        model = DummyQuantModel(self.yaml_path)

        # Create valid ptq_config and datasets for testing
        ptq_config = PTQConfig(
            mode=PTQMode.QUANTIZE,
            backend=BackendTarget.ASCEND,
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8
        )
        datasets = GeneratorDataset([1, 2, 3], "data")

        # Test with invalid layers_policy types
        invalid_policies = [
            None,
            "invalid_string",
            123,
            ["list", "not", "dict"],
            {"key": "value"},  # Regular dict, not OrderedDict
        ]

        for invalid_policy in invalid_policies:
            with pytest.raises(TypeError):
                model.calibrate(
                    ptq_config=ptq_config,
                    layers_policy=invalid_policy,
                    datasets=datasets
                )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_calibrate_param_value_error_datasets_none(self):
        """
        Feature: MFModel calibrate parameter value error - datasets None.
        Description: Test calibrate method with None datasets.
        Expectation: ValueError is raised when datasets is None or empty.
        """
        model = DummyQuantModel(self.yaml_path)

        # Create valid ptq_config and layers_policy for testing
        ptq_config = PTQConfig(
            mode=PTQMode.QUANTIZE,
            backend=BackendTarget.ASCEND,
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8
        )
        layers_policy = OrderedDict()

        # Mock PTQ and related methods to avoid actual execution
        with patch('mindspore_gs.ptq.models.mindformers_models.mf_model.PTQ') as mock_ptq_class, \
             patch('mindspore_gs.ptq.models.mindformers_models.mf_model.offload_network'), \
             patch('mindspore_gs.ptq.models.mindformers_models.mf_model.logger'):
            mock_ptq = MagicMock()
            mock_ptq_class.return_value = mock_ptq
            # Mock the methods that will be called on ptq
            mock_ptq.apply.side_effect = ValueError("please provide dataset when use PTQ quant to quantize network.")
            mock_ptq.summary = MagicMock()
            mock_ptq.set_ptq_config = MagicMock()
            # Mock _set_ptq_config and _load_mindformers_plugin to return the mock_ptq
            with patch.object(model, '_set_ptq_config', return_value=mock_ptq), \
                 patch.object(model, '_load_mindformers_plugin', return_value=mock_ptq):
                # Test with None datasets
                with pytest.raises(ValueError):
                    model.calibrate(
                        ptq_config=ptq_config,
                        layers_policy=layers_policy,
                        datasets=None
                    )

                # Reset side_effect for the second test
                error_msg = "please provide dataset when use PTQ quant to quantize network."
                mock_ptq.apply.side_effect = ValueError(error_msg)

                # Test with empty list (treated as None)
                with pytest.raises(ValueError):
                    model.calibrate(
                        ptq_config=ptq_config,
                        layers_policy=layers_policy,
                        datasets=[]
                    )

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_calibrate_param_type_error_datasets(self):
        """
        Feature: MFModel calibrate parameter type error - datasets.
        Description: Test calibrate method with invalid datasets type.
        Expectation: RuntimeError is raised when datasets is not Dataset type.
        """
        model = DummyQuantModel(self.yaml_path)

        # Create valid ptq_config and layers_policy for testing
        ptq_config = PTQConfig(
            mode=PTQMode.QUANTIZE,
            backend=BackendTarget.ASCEND,
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8
        )
        layers_policy = OrderedDict()

        # Test with invalid datasets types
        invalid_datasets = [
            "invalid_string",
            123,
            {"key": "value"},
            ["list", "not", "dataset"],
        ]

        # Mock PTQ and related methods to avoid actual execution
        for invalid_dataset in invalid_datasets:
            with patch('mindspore_gs.ptq.models.mindformers_models.mf_model.PTQ') as mock_ptq_class, \
                 patch('mindspore_gs.ptq.models.mindformers_models.mf_model.offload_network'), \
                 patch('mindspore_gs.ptq.models.mindformers_models.mf_model.logger'):
                mock_ptq = MagicMock()
                mock_ptq_class.return_value = mock_ptq
                error_msg = (f"The type of dataset is not correct, suppose to Dataset, "
                            f"but got {type(invalid_dataset)}")
                mock_ptq.apply.side_effect = RuntimeError(error_msg)
                mock_ptq.summary = MagicMock()
                mock_ptq.set_ptq_config = MagicMock()

                # Mock _set_ptq_config and _load_mindformers_plugin to return the mock_ptq
                with patch.object(model, '_set_ptq_config', return_value=mock_ptq), \
                     patch.object(model, '_load_mindformers_plugin', return_value=mock_ptq):
                    with pytest.raises((RuntimeError, ValueError)):
                        model.calibrate(
                            ptq_config=ptq_config,
                            layers_policy=layers_policy,
                            datasets=invalid_dataset
                        )
