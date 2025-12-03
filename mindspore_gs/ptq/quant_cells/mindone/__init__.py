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
"""Wrapper cells for PTQ for MindOne."""

from .linear_weight_quant_wrappers import WeightQuantLinearCell
from .linear_all_quant_wrappers import AllQuantLinearCell
from .linear_dynamic_quant_wrappers import DynamicQuantLinearCell
from .linear_gptq_quant_wrappers import GptqWeightQuantLinearCell

from .linear_smooth_wrappers import SmoothQuantLinearCell
from .linear_smooth_wrappers import OSLSmoothQuantLinearCell
from .linear_smooth_wrappers import AWQSmoothQuantLinearCell

from .linear_clip_wrappers import ClipLinearCell

from .fake_quant_linear import FakeQuantA16WxWrapper
from .fake_quant_linear import FakeQuantW8A8Wrapper
from .fake_quant_linear import FakeQuantW8A8DynamicWrapper

__all__ = [
    "WeightQuantLinearCell",
    "AllQuantLinearCell",
    "DynamicQuantLinearCell",
    "GptqWeightQuantLinearCell",
    "SmoothQuantLinearCell",
    "OSLSmoothQuantLinearCell",
    "AWQSmoothQuantLinearCell",
    "ClipLinearCell",
    "FakeQuantA16WxWrapper",
    "FakeQuantW8A8Wrapper",
    "FakeQuantW8A8DynamicWrapper",
]
