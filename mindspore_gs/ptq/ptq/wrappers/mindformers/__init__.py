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
"""Wrapper cells for PTQ for MindFormers."""

from .linear_smooth_wrappers import SmoothQuantLinearCell, AWQSmoothLinearCell, OutlierSuppressionPlusLinearCell, SearchOutlierSuppressionLiteLinearCell
from .linear_weight_quant_wrappers import WeightQuantLinearCell
from .linear_gptq_quant_wrappers import GptqWeightQuantLinearCell
from .linear_clip_wrappers import ClipLinearCell
from .linear_all_quant_wrappers import AllQuantLinearCell
from .linear_dynamic_quant_wrappers import DynamicQuantLinearCell
from .linear_gptq_dynamic_quant_wrappers import GptqDynamicQuantLinearCell
from .kvcache_quant_wrappers import QuantPageAttentionMgrCell, DynamicQuantPageAttentionMgrCell

SearchOutlierSuppressionLiteLinearCell.reg_self()
SmoothQuantLinearCell.reg_self()
AWQSmoothLinearCell.reg_self()
OutlierSuppressionPlusLinearCell.reg_self()
WeightQuantLinearCell.reg_self()
GptqWeightQuantLinearCell.reg_self()
ClipLinearCell.reg_self()
AllQuantLinearCell.reg_self()
DynamicQuantLinearCell.reg_self()
GptqDynamicQuantLinearCell.reg_self()
QuantPageAttentionMgrCell.reg_self()
DynamicQuantPageAttentionMgrCell.reg_self()
