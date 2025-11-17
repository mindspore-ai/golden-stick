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
"""ptq wrapper cells for mindone."""

import numpy as np

from mindspore import nn, Parameter, Tensor, dtype, mint
from mindspore import ops as msops

from mindspore_gs.common import logger
from mindspore_gs.common.gs_enum import BackendTarget
from mindspore_gs.ptq.ptq_config import QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_quant_func import quant_tensor
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.ptq.utils import QuantType

from .linear_weight_quant_wrappers import WeightQuantLinearCell


class AllQuantLinearCell(WeightQuantLinearCell):
    """QuantLinearCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class A8W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype == dtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TENSOR

        Quantizer.reg_layer_map(nn.Dense, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(mint.nn.Linear, AllQuantLinearCell, A8W8Checker())

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.x_quant_max, self.x_quant_min = msops.max, msops.min

    def _quant_info(self):
        res = super()._quant_info()
        if self.cfg.act_quant_dtype == dtype.int8:
            return f'{res}-A8-{str(self.cfg.act_quant_granularity)}'
        raise RuntimeError(f"Unexpected act_quant_dtype: {self.cfg.act_quant_dtype}.")

    def quant(self):
        """quant"""
        # quant weight
        super().quant()
        if hasattr(self.layer, "smooth_scale"):
            cat_samples = self.cat_samples / self.layer.smooth_scale
        else:
            cat_samples = self.cat_samples
        # quant activation
        x_scale, x_zp, _ = quant_tensor(cat_samples, self.x_quant_min, self.x_quant_max,
                                        self.cfg.act_narrow_range, self.cfg.act_symmetric,
                                        self.cfg.act_quant_granularity == QuantGranularity.PER_GROUP,
                                        self.cfg.group_size,
                                        self.cfg.act_quant_dtype, -1, False)

        self.input_scale = Parameter(Tensor(x_scale, self.compute_type))
        self.input_offset = Parameter(Tensor(x_zp, dtype=dtype.int32))

        if hasattr(self.layer, "smooth_scale"):
            self.smooth_scale = self.layer.smooth_scale
        self.deq_scale = Parameter(Tensor(self._get_dequant_scale(self.input_scale.asnumpy(),
                                                                  self.weight_scale.asnumpy()),
                                                                  dtype=self.compute_type))

        self.cfg.dumper.dump_data(self.layer_name, "|activation_params|input0_activation_inputs", self.cat_samples)
        self.cfg.dumper.dump_data(self.layer_name, "|activation_params|output0_activation_scale", x_scale)
        self.cfg.dumper.dump_data(self.layer_name, "|activation_params|output1_activation_zp", x_zp)

    def _get_dequant_scale(self, input_scale, weight_scale):
        """_get_dequant_scale"""
        dequant_scale = input_scale.astype(np.float32) * weight_scale.astype(np.float32)
        return dequant_scale

    def quant_type_dict(self):
        """quant_type_dict"""
        quant_type = {
            self.weight_scale.name: QuantType.W8A8.value,
            self.weight_offset.name: QuantType.W8A8.value,
            self.weight.name: QuantType.W8A8.value,
            self.input_scale.name: QuantType.W8A8.value,
            self.input_offset.name: QuantType.W8A8.value,
            self.deq_scale.name: QuantType.W8A8.value,
        }
        if hasattr(self.layer, "smooth_scale"):
            quant_type.update({self.smooth_scale.name: QuantType.W8A8.value})
        if self.has_bias:
            quant_type.update({self.bias.name: QuantType.W8A8.value})
        return quant_type
