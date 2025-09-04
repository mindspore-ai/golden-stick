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
MindFormers Models Package

This package provides quantized model implementations for the
MindFormers framework. It includes specific implementations for
various large language models that are compatible with the
MindFormers ecosystem.

Supported Models:
    - QWen3: Quantized implementation for Qwen3 models
    - QWen3MoE: Quantized implementation for Qwen3 Mixture-of-Experts models
    - DeepSeekV3: Quantized implementation for DeepSeekV3 models

The implementations in this package are designed to work seamlessly
with MindFormers' model loading, training, and inference capabilities
while providing efficient quantization support for deployment scenarios.

Key Features:
    - Seamless integration with MindFormers framework
    - Support for distributed computing and tensor parallelism
    - Efficient parameter management for large-scale models
    - SafeTensors format support for model persistence
    - Automatic model detection and selection

Example:
    >>> from mindspore_gs.ptq.models import AutoQuantForCausalLM
    >>>
    >>> # Automatically detect and load MindFormers-compatible model
    >>> model = AutoQuantForCausalLM.from_pretrained("/path/to/mindformers_model.yaml")
"""

from .mf_model import MFModel
from .qwen3 import QWen3
from .qwen3_moe import QWen3MoE
from .deepseekv3 import DeepSeekV3
from .telechat2 import Telechat2
