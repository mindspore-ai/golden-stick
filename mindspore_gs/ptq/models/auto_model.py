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
"""
Auto Model Quantization Module

This module provides high-level interfaces for automatic model quantization,
simplifying the process of quantizing large language models. It automatically
detects and selects the appropriate quantized model implementation based on
the pretrained model path.

The auto model quantization class serves as the main entry point for users
who want to quickly apply quantization to their models without needing to
manually select specific model implementations.

Examples:
    >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
    >>>
    >>> # Automatically detect and load the appropriate model
    >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/model.yaml")
    >>>
    >>> # Calibrate and quantize the model
    >>> model.calibrate(ptq_config, layers_policy, calibration_dataset)
    >>>
    >>> # Save the quantized model
    >>> model.save_quantized("/path/to/save/location")
"""

import os

from mindspore_gs.common import logger
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


class AutoQuantForCausalLM:
    """Auto Model Quantization Class

    This class provides automatic model detection and selection for
    quantizing causal language models. It uses a registry mechanism to
    automatically identify and instantiate the appropriate quantized
    model implementation based on the pretrained model configuration.

    The class implements a factory pattern that scans through all registered
    model hubs and attempts to create a model instance from each one until
    a successful match is found.

    Examples:
        >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
        >>>
        >>> # Automatically select the appropriate model implementation
        >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/model.yaml")
    """

    @staticmethod
    def from_pretrained(pretained) -> BaseQuantForCausalLM:
        """Create a quantized model instance from a pretrained model path.

        This method automatically detects the model type from the provided
        pretrained model path and selects the appropriate quantized model
        implementation. It iterates through all registered model hubs and
        attempts to create a model instance from each one.

        Args:
            pretained (str): Path or identifier of the pretrained model.
                This can be a local file path to a model configuration
                file or a model identifier recognized by the system.

        Returns:
            BaseQuantForCausalLM. A quantized model instance that inherits
            from BaseQuantForCausalLM. The specific type depends on the
            detected model framework and configuration.

        Examples:
            >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
            >>>
            >>> # Automatically select Qwen3 model implementation
            >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/qwen3_model.yaml")
            >>>
            >>> # Automatically select DeepSeekV3 model implementation
            >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/deepseek_config.yaml")
        """
        # FIXME: hangangqiang2: we load mindformers models by default.
        # FIXME: hangangqiang2: we should load them as plugin in the future.
        # pylint: disable=unused-import
        if pretained.endswith(".yaml"):
            model_registry = AutoQuantForCausalLM._load_mindformers_plugin()
            model_name = AutoQuantForCausalLM._get_mindformers_model_name(pretained)
        elif os.path.isdir(pretained):
            model_registry = AutoQuantForCausalLM._load_mindone_plugin()
            model_name = AutoQuantForCausalLM._get_mindone_model_name(pretained)
        else:
            raise ValueError(f"Unsupported model type: {pretained}")

        try:
            logger.info(f"Create model from {pretained}")
            return model_registry[model_name].from_pretrained(pretained)
        except ValueError as e:
            raise ValueError(f"Create model from {pretained} failed, error: {e}") from e

    @staticmethod
    def _load_mindformers_plugin():
        """_load_mindformers_plugin"""
        try:
            # pylint: disable=unused-import
            from .mindformers_models import QWen3, QWen3MoE, DeepSeekV3
        except ImportError as e:
            raise ImportError(f"mindformers plugin load failed, error: {e}") from e
        from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
        return MFModel.get_model_registry()

    @staticmethod
    def _load_mindone_plugin():
        """_load_mindone_plugin"""
        try:
            # pylint: disable=unused-import
            from .mindone_models import GLM4v
            from .mindone_models import Qwen3_MOE
        except ImportError as e:
            raise ImportError(f"mindone plugin load failed, error: {e}") from e
        from mindspore_gs.ptq.models.mindone_models.mindone_model import MindOneModel
        return MindOneModel.get_model_registry()

    @staticmethod
    def _get_mindformers_model_name(model_path):
        """get_mindformers_model_name"""
        from mindformers import MindFormerConfig
        mfconfig = MindFormerConfig(model_path)
        return mfconfig.trainer.model_name

    @staticmethod
    def _get_mindone_model_name(model_path):
        """get_mindone_model_name"""
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        return config.model_type
