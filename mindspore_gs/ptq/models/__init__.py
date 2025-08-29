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
PTQ Models Package

This package provides high-level interfaces and implementations for
post-training quantization of large language models. It includes:

1. Auto Model Interface: Automatic model detection and selection
2. Base Model Classes: Standardized interfaces for quantized models
3. Framework-Specific Implementations: Support for different model frameworks
4. Utility Functions: Helper functions for model management and quantization

The package is organized to provide a simple and consistent API for
users while supporting multiple model frameworks and quantization
algorithms through a plugin-based architecture.

Key Components:
    - AutoQuantForCausalLM: Main entry point for automatic model quantization
    - BaseQuantForCausalLM: Base class defining standard quantization interfaces
    - Framework-specific implementations (MindFormers, etc.)

Example:
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

from .auto_model import AutoQuantForCausalLM
from .base_model import BaseQuantForCausalLM
