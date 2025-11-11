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
"""Quantization Cells for Post-Training Quantization.

This module provides the fundamental building blocks for implementing quantization 
algorithms in the Golden Stick PTQ framework. Quantization cells are specialized 
neural network layers that encapsulate quantization logic and can be composed 
to create complex quantization pipelines.

Architecture Overview:
    - Base Quantization Cell: Abstract base class defining quantization interface
    - Framework-Specific Implementations: Quantization cells tailored for different 
      model frameworks (MindOne, MindFormers, etc.)
    - Algorithm-Specific Implementations: Specialized cells implementing specific 
      quantization techniques (SmoothQuant, GPTQ, AWQ, etc.)
    - Wrapper Patterns: Cell wrappers that add quantization functionality to 
      existing layers

The quantization cells module follows a modular design where each cell implements 
a specific quantization technique and can be composed with other cells to create 
custom quantization pipelines. This enables researchers to experiment with new 
quantization algorithms while maintaining compatibility with the broader framework.

Integration with Algorithms:
    Quantization cells work in conjunction with algorithm modules to provide 
    a complete quantization solution. The cells implement the low-level quantization 
    operations while algorithms coordinate the overall quantization strategy.
"""
