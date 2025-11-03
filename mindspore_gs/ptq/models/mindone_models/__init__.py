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
MindOne Models Package

This package provides quantized model implementations for the
MindOne framework. It includes specific implementations for
various large language models that are compatible with the
MindOne ecosystem.

Supported Models:
    - GLM4V: Quantized implementation for GLM4V models

The implementations in this package are designed to work seamlessly
with MindOne's model loading, training, and inference capabilities
while providing efficient quantization support for deployment scenarios.

Key Features:
    - Seamless integration with MindOne framework
    - Support for distributed computing and tensor parallelism
    - Efficient parameter management for large-scale models
    - SafeTensors format support for model persistence
    - Automatic model detection and selection

Example:
    >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
    >>>
    >>> # Automatically detect and load MindOne-compatible model
    >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/mindone_model")
"""

from .mindone_model import MindOneModel
from .glm4v import GLM4v
