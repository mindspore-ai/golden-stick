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

import os
import shutil
import tempfile
import inspect
from unittest.mock import patch, MagicMock
import pytest

from mindspore_gs.ptq.models.auto_model import AutoQuantForCausalLM
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


class TestAutoQuantForCausalLM:
    """Test cases for AutoQuantForCausalLM class - normal interface cases."""

    def setup_method(self):
        """Setup method to prepare test environment."""
        self.work_dir = tempfile.mkdtemp()
        self.yaml_path = os.path.join(self.work_dir, "test_model.yaml")
        self.model_dir = os.path.join(self.work_dir, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)

    def teardown_method(self):
        """Teardown method to clean up test environment."""
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_normal_yaml_path(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained normal case with yaml path
        Description: Test from_pretrained method normal call and model creation with yaml file path
        Expectation: Method correctly called, model successfully created, returns correct model instance
        """
        # Create a yaml file
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("model_type: test_model\n")

        # Mock model hub and model
        mock_model_hub = MagicMock()
        mock_model = MagicMock()
        mock_model_hub.from_pretrained.return_value = mock_model

        # Use patch as context manager instead of decorator
        plugin_path = 'mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindformers_plugin'
        with patch(plugin_path) as mock_load_plugin:
            mock_load_plugin.return_value = mock_model_hub

            # Call from_pretrained
            result = AutoQuantForCausalLM.from_pretrained(self.yaml_path)

            # Verify plugin loading was called
            mock_load_plugin.assert_called_once()

            # Verify model hub's from_pretrained was called with correct path
            mock_model_hub.from_pretrained.assert_called_once_with(self.yaml_path)

            # Verify correct model instance is returned
            assert result == mock_model
            assert isinstance(result, MagicMock)

            # Verify model hub integration: parameters passed correctly
            call_args = mock_model_hub.from_pretrained.call_args
            assert call_args[0][0] == self.yaml_path

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_normal_directory_path(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained normal case with directory path
        Description: Test from_pretrained method normal call and model creation with directory path
        Expectation: Method correctly called, model successfully created, returns correct model instance
        """
        # Mock model hub and model
        mock_model_hub = MagicMock()
        mock_model = MagicMock()
        mock_model_hub.from_pretrained.return_value = mock_model

        # Use patch as context manager instead of decorator
        with patch('os.path.isdir') as mock_isdir, \
             patch('mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindone_plugin') as mock_load_plugin:
            mock_isdir.return_value = True
            mock_load_plugin.return_value = mock_model_hub

            # Call from_pretrained
            result = AutoQuantForCausalLM.from_pretrained(self.model_dir)

            # Verify directory check was called
            mock_isdir.assert_called_once_with(self.model_dir)

            # Verify plugin loading was called
            mock_load_plugin.assert_called_once()

            # Verify model hub's from_pretrained was called with correct path
            mock_model_hub.from_pretrained.assert_called_once_with(self.model_dir)

            # Verify correct model instance is returned
            assert result == mock_model

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_invalid_path_error(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained invalid path error
        Description: Test from_pretrained method raises ValueError for invalid path
        Expectation: Invalid path raises ValueError with correct error message
        """
        # Test with invalid path (neither yaml nor directory)
        invalid_path = "invalid_path_that_does_not_exist.xyz"

        # Verify path is invalid
        assert not invalid_path.endswith(".yaml")
        assert not os.path.isdir(invalid_path)

        # Call from_pretrained and expect ValueError
        with pytest.raises(ValueError):
            AutoQuantForCausalLM.from_pretrained(invalid_path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_param_type_error(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained parameter type error
        Description: Test from_pretrained method parameter type error handling
        Expectation: Parameter type error handled correctly, exception information accurate
        """
        # Test with non-string parameter (integer)
        with pytest.raises((AttributeError, TypeError)):
            AutoQuantForCausalLM.from_pretrained(123)

        # Test with non-string parameter (None)
        with pytest.raises((AttributeError, TypeError)):
            AutoQuantForCausalLM.from_pretrained(None)

        # Test with non-string parameter (list)
        with pytest.raises((AttributeError, TypeError)):
            AutoQuantForCausalLM.from_pretrained(["path", "to", "model"])

        # Test with non-string parameter (dict)
        with pytest.raises((AttributeError, TypeError)):
            AutoQuantForCausalLM.from_pretrained({"path": "model.yaml"})

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_get_model_hub_registry_access(self):
        """
        Feature: AutoQuantForCausalLM model hub registry access
        Description: Test get_model_hub_registry method for accessing registry
        Expectation: Registry access normal, returns correct registry dictionary
        """
        # Get registry through BaseQuantForCausalLM (since AutoQuantForCausalLM uses it)
        registry = BaseQuantForCausalLM.get_model_hub_registry()

        # Verify registry is a dictionary
        assert isinstance(registry, dict)

        # Verify registry can be accessed
        assert registry is not None

        # Verify registry type check
        assert hasattr(registry, 'get')
        assert hasattr(registry, '__getitem__')
        assert hasattr(registry, '__contains__')

        # Verify registry content can be queried
        # The registry may be empty or contain registered models
        assert isinstance(registry.get("mindformers"), (type, type(None)))
        assert isinstance(registry.get("mindone"), (type, type(None)))

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("method_name", ["_load_mindformers_plugin", "_load_mindone_plugin"])
    def test_load_plugin_method(self, method_name):
        """
        Feature: AutoQuantForCausalLM plugin loading methods
        Description: Test plugin loading method existence and signature
        Expectation: Method exists, is static, and has correct signature
        """
        # Verify method exists
        assert hasattr(AutoQuantForCausalLM, method_name)

        # Get the method
        method = getattr(AutoQuantForCausalLM, method_name)

        # Verify it's a static method
        assert inspect.isfunction(method) or inspect.ismethod(method)

        # Verify method signature (should take no parameters)
        sig = inspect.signature(method)
        assert len(sig.parameters) == 0

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_empty_registry_error(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained empty registry error
        Description: Test from_pretrained method when registry is empty or hub returns None
        Expectation: Empty registry handled correctly, AttributeError raised when hub is None
        """
        # Create a yaml file
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("model_type: test_model\n")

        # Mock plugin loading to return None (empty registry scenario)
        plugin_path = 'mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindformers_plugin'
        with patch(plugin_path) as mock_load_plugin:
            mock_load_plugin.return_value = None

            # Call from_pretrained and expect AttributeError when trying to call None.from_pretrained()
            with pytest.raises(AttributeError):
                AutoQuantForCausalLM.from_pretrained(self.yaml_path)

            # Verify plugin loading was called
            mock_load_plugin.assert_called_once()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_from_pretrained_model_hub_value_error_propagation(self):
        """
        Feature: AutoQuantForCausalLM.from_pretrained model hub ValueError propagation
        Description: Test from_pretrained method when model hub raises ValueError
        Expectation: ValueError from model hub is caught and re-raised with additional context
        """
        # Create a yaml file
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("model_type: test_model\n")

        # Mock model hub to raise ValueError
        mock_model_hub = MagicMock()
        original_error = ValueError("Model configuration invalid")
        mock_model_hub.from_pretrained.side_effect = original_error

        plugin_path = 'mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindformers_plugin'
        with patch(plugin_path) as mock_load_plugin:
            mock_load_plugin.return_value = mock_model_hub

            # ValueError should be caught and re-raised
            with pytest.raises(ValueError):
                AutoQuantForCausalLM.from_pretrained(self.yaml_path)

            # Verify plugin loading was called
            mock_load_plugin.assert_called_once()
            mock_model_hub.from_pretrained.assert_called_once_with(self.yaml_path)

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("method_name,module_name", [
        ("_load_mindformers_plugin", "mindspore_gs.ptq.models.mindformers_models"),
        ("_load_mindone_plugin", "mindspore_gs.ptq.models.mindone_models")
    ])
    def test_load_plugin_import_error(self, method_name, module_name):
        """
        Feature: AutoQuantForCausalLM plugin loading import error
        Description: Test plugin loading method when import fails
        Expectation: ImportError raised when import fails
        """
        # Mock the import statement to raise ImportError
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == module_name:
                raise ImportError(f"No module named '{module_name}'")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            method = getattr(AutoQuantForCausalLM, method_name)
            with pytest.raises(ImportError):
                method()

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    @pytest.mark.parametrize("method_name,hub_name", [
        ("_load_mindformers_plugin", "mindformers"),
        ("_load_mindone_plugin", "mindone")
    ])
    def test_load_plugin_success(self, method_name, hub_name):
        """
        Feature: AutoQuantForCausalLM plugin loading success
        Description: Test plugin loading method when import succeeds
        Expectation: Plugin loaded successfully, returns model hub with get_model_registry method
        """
        # Mock model hub class with get_model_registry method
        mock_model_registry = {'test_model': MagicMock()}
        mock_model_hub = MagicMock()
        mock_model_hub.get_model_registry = MagicMock(return_value=mock_model_registry)

        # Mock the registry to return our mock hub
        registry_path = 'mindspore_gs.ptq.models.auto_model.BaseQuantForCausalLM.get_model_hub_registry'
        with patch(registry_path) as mock_get_registry:
            mock_registry = {hub_name: mock_model_hub}
            mock_get_registry.return_value = mock_registry

            # Mock the import to succeed
            with patch('builtins.__import__', return_value=MagicMock()):
                method = getattr(AutoQuantForCausalLM, method_name)
                result = method()

                # Verify plugin loading returns model hub
                assert result is not None
                assert result == mock_model_hub

                # Verify model hub has get_model_registry method
                assert hasattr(result, 'get_model_registry')
                assert callable(result.get_model_registry)

                # Verify get_model_registry returns a dictionary
                registry = result.get_model_registry()
                assert isinstance(registry, dict)
                assert 'test_model' in registry

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_model_detection_and_selection(self):
        """
        Feature: AutoQuantForCausalLM model detection and selection
        Description: Test automatic model detection and selection functionality
        Expectation: Model detection functionality normal, selection logic correct
        """
        # Test yaml path detection and selection
        yaml_path = "test_model.yaml"
        assert yaml_path.endswith(".yaml")

        # Test directory path detection and selection
        assert os.path.isdir(self.model_dir)
        assert not self.model_dir.endswith(".yaml")

        # Mock model hubs for both paths
        mock_mindformers_hub = MagicMock()
        mock_mindone_hub = MagicMock()
        mock_model = MagicMock()

        # Test yaml path routes to mindformers
        with patch('mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindformers_plugin') as mock_load_mf:
            mock_load_mf.return_value = mock_mindformers_hub
            mock_mindformers_hub.from_pretrained.return_value = mock_model

            result = AutoQuantForCausalLM.from_pretrained(yaml_path)

            # Verify mindformers plugin was loaded (not mindone)
            mock_load_mf.assert_called_once()
            mock_mindformers_hub.from_pretrained.assert_called_once_with(yaml_path)
            assert result == mock_model

        # Test directory path routes to mindone
        with patch('os.path.isdir') as mock_isdir, \
             patch('mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindone_plugin') as mock_load_mo:
            mock_isdir.return_value = True
            mock_load_mo.return_value = mock_mindone_hub
            mock_mindone_hub.from_pretrained.return_value = mock_model

            result = AutoQuantForCausalLM.from_pretrained(self.model_dir)

            # Verify mindone plugin was loaded (not mindformers)
            mock_isdir.assert_called_once_with(self.model_dir)
            mock_load_mo.assert_called_once()
            mock_mindone_hub.from_pretrained.assert_called_once_with(self.model_dir)
            assert result == mock_model

    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_exception_handling_mechanism(self):
        """
        Feature: AutoQuantForCausalLM exception handling mechanism
        Description: Test exception handling mechanism and error recovery
        Expectation: Exception handling mechanism normal, error recovery functionality effective
        """
        # Create a yaml file
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("model_type: test_model\n")

        # Test ValueError exception handling
        mock_model_hub = MagicMock()
        mock_model_hub.from_pretrained.side_effect = ValueError("Model configuration invalid")

        plugin_path = 'mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindformers_plugin'
        with patch(plugin_path) as mock_load_plugin:
            mock_load_plugin.return_value = mock_model_hub

            # Verify ValueError is caught and re-raised
            with pytest.raises(ValueError):
                AutoQuantForCausalLM.from_pretrained(self.yaml_path)

        # Test that non-ValueError exceptions propagate without modification
        mock_model_hub2 = MagicMock()
        mock_model_hub2.from_pretrained.side_effect = RuntimeError("Internal error")

        plugin_path2 = 'mindspore_gs.ptq.models.auto_model.AutoQuantForCausalLM._load_mindformers_plugin'
        with patch(plugin_path2) as mock_load_plugin2:
            mock_load_plugin2.return_value = mock_model_hub2

            # Verify RuntimeError propagates without being caught
            with pytest.raises(RuntimeError):
                AutoQuantForCausalLM.from_pretrained(self.yaml_path)
