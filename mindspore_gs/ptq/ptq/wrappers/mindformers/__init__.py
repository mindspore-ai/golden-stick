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

from mindformers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindformers.modules import PagedAttentionMgr
from mindspore_gs.ptq.ptq.algorithms.anti_outliers import LinearSmoother
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from .smooth_wrappers import SmoothLinearCell
from .quant_wrappers import QuantLinearCell, QuantPageAttentionMgrCell

LinearSmoother.reg_linear_map(Linear, SmoothLinearCell)
LinearSmoother.reg_linear_map(ColumnParallelLinear, SmoothLinearCell)
LinearSmoother.reg_linear_map(RowParallelLinear, SmoothLinearCell)

Quantizer.reg_layer_map(Linear, QuantLinearCell)
Quantizer.reg_layer_map(ColumnParallelLinear, QuantLinearCell)
Quantizer.reg_layer_map(RowParallelLinear, QuantLinearCell)
Quantizer.reg_layer_map(PagedAttentionMgr, QuantPageAttentionMgrCell)
