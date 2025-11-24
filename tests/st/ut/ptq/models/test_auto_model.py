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
"""Unit tests for AutoQuantForCausalLM without mindformers dependency.

This test suite validates:
- Normal interface behavior for YAML and directory inputs
- Error handling for invalid and non-string inputs
- Propagation of plugin create_model exceptions
- Argument passthrough to plugin
- Static method signature constraints

MindFormers-dependent behaviors are stubbed via load_plugin patching.
"""

import os
import sys
import shutil
import tempfile
import inspect
from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))
from mindspore_gs.ptq.models.auto_model import AutoQuantForCausalLM
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


# pylint: disable=abstract-method
class DummyModel(BaseQuantForCausalLM):
    """Minimal dummy model used for UT validation."""

    def __init__(self, tag):
        self.tag = tag


class TestAutoQuantForCausalLM:
    """UT suite covering normal and exceptional inputs for AutoQuantForCausalLM."""

    def setup_method(self):
        """Create temporary workspace and paths for test isolation."""
        self.work_dir = tempfile.mkdtemp()
        self.yaml_path = os.path.join(self.work_dir, "test_model.yaml")
        self.model_dir = os.path.join(self.work_dir, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)

    def teardown_method(self):
        """Cleanup temporary workspace after each test."""
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_yaml_path_returns_model(self):
        """YAML input returns a BaseQuantForCausalLM instance via plugin stub."""
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("trainer:\nmodel_name: 'x'\n")

        def create_model_fn(path):
            assert path == self.yaml_path
            assert path.endswith('.yaml')
            return DummyModel("yaml")

        class StubPlugin:
            def create_model(self, path):
                return create_model_fn(path)

        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', return_value=StubPlugin()):
            model = AutoQuantForCausalLM.from_pretrained(self.yaml_path)
            assert isinstance(model, BaseQuantForCausalLM)
            assert isinstance(model, DummyModel)
            assert model.tag == "yaml"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_dir_path_returns_model(self):
        """Directory input returns a model instance via plugin stub."""
        def create_model_fn(path):
            assert path == self.model_dir
            assert os.path.isdir(path)
            return DummyModel("dir")

        class StubPlugin:
            def create_model(self, path):
                return create_model_fn(path)

        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', return_value=StubPlugin()):
            model = AutoQuantForCausalLM.from_pretrained(self.model_dir)
            assert isinstance(model, DummyModel)
            assert model.tag == "dir"

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_invalid_path_raises_value_error(self):
        """Invalid path triggers ValueError from load_plugin selection."""
        invalid_path = "not_exist.xyz"
        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', side_effect=ValueError("Unsupported model type")):
            with pytest.raises(ValueError):
                AutoQuantForCausalLM.from_pretrained(invalid_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_non_string_input_raises_value_error(self):
        """Non-string input triggers ValueError from load_plugin selection."""
        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', side_effect=ValueError("Unsupported model type")):
            with pytest.raises(ValueError):
                AutoQuantForCausalLM.from_pretrained(123)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_plugin_none_raises_attribute_error(self):
        """Plugin returns None causing AttributeError on attribute access."""
        with open(self.yaml_path, 'w', encoding='utf-8') as f:
            f.write("trainer:\n")
        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', return_value=None):
            with pytest.raises(AttributeError):
                AutoQuantForCausalLM.from_pretrained(self.yaml_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_create_model_value_error_propagates(self):
        """ValueError raised by plugin.create_model is propagated."""
        class StubPlugin:
            def create_model(self, path):
                raise ValueError("bad config")
        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', return_value=StubPlugin()):
            with pytest.raises(ValueError):
                AutoQuantForCausalLM.from_pretrained(self.yaml_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_create_model_runtime_error_propagates(self):
        """RuntimeError raised by plugin.create_model is propagated."""
        class StubPlugin:
            def create_model(self, path):
                raise RuntimeError("internal")
        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', return_value=StubPlugin()):
            with pytest.raises(RuntimeError):
                AutoQuantForCausalLM.from_pretrained(self.yaml_path)

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_argument_passthrough(self):
        """Complex path is passed through unchanged to plugin.create_model."""
        weird_path = os.path.join(self.work_dir, "a b/测试.yaml")
        os.makedirs(os.path.dirname(weird_path), exist_ok=True)
        with open(weird_path, 'w', encoding='utf-8') as f:
            f.write("trainer:\n")

        captured = {}

        class StubPlugin:
            def create_model(self, path):
                captured['path'] = path
                return DummyModel("x")

        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', return_value=StubPlugin()):
            _ = AutoQuantForCausalLM.from_pretrained(weird_path)
            assert captured['path'] == weird_path

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_static_method_signature(self):
        """from_pretrained is a static method accepting a single argument."""
        fn = AutoQuantForCausalLM.from_pretrained
        assert inspect.isfunction(fn) or inspect.ismethod(fn)
        sig = inspect.signature(fn)
        assert len(sig.parameters) == 1

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_empty_string_input(self):
        """Empty string input yields ValueError from load_plugin selection."""
        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', side_effect=ValueError("Unsupported model type")):
            with pytest.raises(ValueError):
                AutoQuantForCausalLM.from_pretrained("")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_onecard
    def test_long_path_string(self):
        """Long YAML path is accepted and passed through to plugin stub."""
        long_name = "a" * 200 + ".yaml"
        long_path = os.path.join(self.work_dir, long_name)
        with open(long_path, 'w', encoding='utf-8') as f:
            f.write("trainer:\n")

        class StubPlugin:
            def create_model(self, path):
                assert path == long_path
                return DummyModel("long")

        with patch('mindspore_gs.ptq.models.auto_model.load_plugin', return_value=StubPlugin()):
            model = AutoQuantForCausalLM.from_pretrained(long_path)
            assert isinstance(model, DummyModel)
            assert model.tag == "long"
