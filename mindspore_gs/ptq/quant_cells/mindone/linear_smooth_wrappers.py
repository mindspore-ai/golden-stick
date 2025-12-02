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
"""ptq wrapper cells for mindformers."""

from mindspore import nn
from mindspore import mint
from mindspore import ops as msops

from mindspore_gs.ptq.ptq_config import OutliersSuppressionType, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.algo_modules.mindone import (SmoothQuantSmoother,
                                                   OSLSmoother,
                                                   AWQSmoother)
from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from mindspore_gs.ptq.basic_functions.basic_quant_func import quant_tensor
from .linear_wrapper import WrapperLinearCell


class SmoothQuantLinearCell(WrapperLinearCell):
    """Smooth Quant algorithm for Linear Cell"""

    @staticmethod
    def reg_self():
        """register SmoothQuantLinearCell"""
        class SmoothQuantChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.SMOOTH

        SmoothQuantSmoother.reg_layer_map(nn.Dense, SmoothQuantLinearCell, SmoothQuantChecker())
        SmoothQuantSmoother.reg_layer_map(mint.nn.Linear, SmoothQuantLinearCell, SmoothQuantChecker())

    def _quant_info(self):
        """_quant_info"""
        return "SmoothQuant"


class OSLSmoothQuantLinearCell(WrapperLinearCell):
    """Outlier Suppression Lite algorithm for Smooth Linear Cell"""

    @staticmethod
    def reg_self():
        """register OSLSmoothQuantLinearCell"""
        class OSLChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE

        OSLSmoother.reg_layer_map(nn.Dense, OSLSmoothQuantLinearCell, OSLChecker())
        OSLSmoother.reg_layer_map(mint.nn.Linear, OSLSmoothQuantLinearCell, OSLChecker())

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.quant_forward = False
        self.dense = msops.Dense()

    def _quant_info(self):
        """_quant_info"""
        return "OSL"

    def set_smooth_scale(self, smooth_scale):
        """set_smooth_scale"""
        self.smooth_scale = smooth_scale

    def _quant_forward(self, x):
        """_quant_forward"""
        x = x / self.smooth_scale
        _, _, x = quant_tensor(x,
                               msops.min,
                               msops.max,
                               self.cfg.act_narrow_range,
                               self.cfg.act_symmetric,
                               False,
                               0,
                               quant_dtype=self.cfg.act_quant_dtype,
                               quant_axis=-1,
                               if_quant_data=True,
                               if_pesudo_quant=True,
                               is_transpose=self.transpose_b,
                               high_precision_params=False)

        weight = self.layer.weight.data * self.smooth_scale
        _, _, weight = quant_tensor(weight,
                                    msops.min,
                                    msops.max,
                                    self.cfg.weight_narrow_range,
                                    self.cfg.weight_symmetric,
                                    False,
                                    0,
                                    quant_dtype=self.cfg.weight_quant_dtype,
                                    quant_axis=0,
                                    if_quant_data=True,
                                    if_pesudo_quant=True,
                                    is_transpose=self.transpose_b,
                                    high_precision_params=False)
        bias = None
        if self.layer.has_bias:
            bias = self.layer.bias
        return self.dense(x, weight, bias)

    # pylint: disable=unused-argument
    def construct(self, x, *args, **kwargs):
        """construct"""
        if self.quant_forward:
            return self._quant_forward(x)
        return self.layer(x)


class AWQSmoothQuantLinearCell(WrapperLinearCell):
    """AWQ algorithm for smooth Linear Cell"""

    @staticmethod
    def reg_self():
        """register AWQSmoothQuantLinearCell"""
        class AWQChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.outliers_suppression == OutliersSuppressionType.AWQ

        AWQSmoother.reg_layer_map(nn.Dense, AWQSmoothQuantLinearCell, AWQChecker())

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.quant_forward = False
        self.dense = msops.Dense()
        self.group_size = self.cfg.group_size if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP \
            else linear.weight.shape[-1]

    def _quant_info(self):
        """_quant_info"""
        return "AWQ"

    def set_smooth_scale(self, smooth_scale):
        """set_smooth_scale"""
        self.smooth_scale = smooth_scale

    def _quant_forward(self, x):
        """_quant_forward"""
        x = x / self.smooth_scale
        weight = self.layer.weight.data * self.smooth_scale
        _, _, weight = quant_tensor(weight,
                                    msops.min,
                                    msops.max,
                                    self.cfg.weight_narrow_range,
                                    self.cfg.weight_symmetric,
                                    True,
                                    self.group_size,
                                    quant_dtype=self.cfg.weight_quant_dtype,
                                    quant_axis=0,
                                    if_quant_data=True,
                                    if_pesudo_quant=True,
                                    is_transpose=self.transpose_b,
                                    high_precision_params=False)
        bias = None
        if self.layer.has_bias:
            bias = self.layer.bias
        return self.dense(x, weight, bias)

    # pylint: disable=unused-argument
    def construct(self, x, *args, **kwargs):
        """construct"""
        if self.quant_forward:
            return self._quant_forward(x)
        return self.layer(x)
