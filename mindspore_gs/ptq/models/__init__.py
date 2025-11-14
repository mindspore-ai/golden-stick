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
PTQ Models Package - High-Level Model Interfaces for Post-Training Quantization

This package implements the model layer of the Golden Stick PTQ architecture, 
providing standardized interfaces and implementations for quantizing large 
language models. The architecture follows a hierarchical design with:

1. Auto Model Interface: Automatic framework detection and model selection
2. Base Model Classes: Abstract interfaces defining quantization contracts
3. Framework-Specific Implementations: Concrete implementations for different ecosystems
4. Plugin Integration: Seamless integration with the plugin system for extensibility

The package embodies the Golden Stick design principle of providing both 
"Level 0" (configuration-based) and "Level 1" (simple API) interfaces, 
enabling both fine-grained control and ease of use.

Architecture Components:
    - AutoQuantForCausalLM: Main entry point following Transformers-like API design
    - BaseQuantForCausalLM: Base class defining standard quantization interfaces
    - Framework-specific implementations (MindFormers, MindOne ecosystems)
    - Plugin-based model loading and quantization cell integration

Key Features:
    - Multi-framework support through plugin architecture
    - Consistent quantization API across different model types
    - Automatic calibration and quantization pipeline
    - Support for various quantization algorithms and techniques
    - Model saving and loading with huggingface safetensors

Example Usage (Level 1 - Simple API):
    TDB

Example Usage (Level 0 - Configuration API):
    >>> import mindspore
    >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
    >>> from mindspore_gs.ptq import PTQConfig
    >>> from mindspore_gs.common import BackendTarget
    >>>
    >>> # Fine-grained configuration
    >>> config = PTQConfig(
    ...     weight_quant_dtype=mindspore.int8,
    ...     act_quant_dtype=mindspore.int8,
    ... )
    >>> layer_policy = {
    ...     "layers.0": PTQConfig(weight_quant_dtype=mindspore.int8),
    ... }
    >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/pretrained")
    >>> model.calibrate(config, layer_policy, calibration_dataset)
    >>> model.save_quantized('/path/to/save/quantized', backend=BackendTarget.ASCEND)
"""

from .auto_model import AutoQuantForCausalLM
from .base_model import BaseQuantForCausalLM
