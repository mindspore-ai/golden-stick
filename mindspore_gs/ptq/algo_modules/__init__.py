# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Algorithm Modules for Post-Training Quantization.

This module provides the core algorithm components for PTQ (Post-Training Quantization) 
in the Golden Stick framework. It implements the modular algorithm architecture that 
enables flexible composition of quantization techniques.

Key Components:
    - AlgoModule: Base class for all quantization algorithms
    - Quantizer: Core quantization algorithm implementation
    - LinearSmoothQuant: Smooth quantization for outlier mitigation
    - LinearAutoSmoother: Automatic smoothing for activation outliers
    - LinearClipper: Weight clipping for improved quantization

The architecture follows a modular design where each algorithm module can be 
independently developed, tested, and composed to create complex quantization 
pipelines. This enables researchers to easily experiment with new quantization 
techniques while maintaining compatibility with existing infrastructure.
"""

from .algo_module import AlgoModule
from .quantizer import Quantizer
