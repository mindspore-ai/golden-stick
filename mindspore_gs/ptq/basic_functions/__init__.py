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
"""Basic Functions and Utilities for Post-Training Quantization.

This module provides fundamental building blocks and utility functions that 
support the PTQ (Post-Training Quantization) framework. These functions 
form the foundation upon which higher-level quantization algorithms are built.

Key Components:
    - Basic quantization operations and transformations
    - Mathematical utilities for quantization algorithms
    - Parameter management and distribution utilities
    - Data processing and transformation functions
    - Safe tensor management and serialization
    - ...

The basic functions module follows the Golden Stick architecture principle of 
providing reusable, well-tested components that can be composed to create 
complex quantization workflows. These functions are designed to be framework-
agnostic and can be used across different model ecosystems.
"""
