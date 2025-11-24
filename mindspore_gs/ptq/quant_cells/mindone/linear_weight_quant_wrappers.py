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

from mindspore import mint
from mindspore import nn, Parameter, Tensor, dtype
from mindspore import ops as msops

from mindspore_gs.ptq.ptq_config import QuantGranularity, PrecisionRecovery
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_functions.basic_quant_func import quant_tensor
from mindspore_gs.ptq.algo_modules import Quantizer
from mindspore_gs.ptq.utils import QuantType
from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from .linear_wrapper import WrapperLinearCell

class WeightQuantLinearCell(WrapperLinearCell):
    """WeightQuantLinearCell"""

    @staticmethod
    def reg_self():
        """register WeightQuantLinearCell"""
        class A16WxChecker(Checker):
            def check(self, config: InnerPTQConfig):
                support_dtype = [dtype.int8, dtype.qint4x2]
                return (config.weight_quant_dtype in support_dtype and config.act_quant_dtype is None
                        and config.precision_recovery == PrecisionRecovery.NONE)

        Quantizer.reg_layer_map(nn.Dense, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(mint.nn.Linear, WeightQuantLinearCell, A16WxChecker())

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        if self.cfg.act_per_channel:
            raise ValueError("only per-tensor activation quantization now.")

        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if self.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if self.transpose_b else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]

        self.compute_type = self.layer.weight.dtype
        self.w_quant_max, self.w_quant_min = msops.max, msops.min

    def _quant_info(self):
        if self.cfg.weight_quant_dtype == dtype.int8:
            return f'W8-{str(self.cfg.weight_quant_granularity)}'
        if self.cfg.weight_quant_dtype == dtype.qint4x2:
            return f'W4-{str(self.cfg.weight_quant_granularity)}'
        raise RuntimeError(f"Unexpected weight_quant_dtype: {self.cfg.weight_quant_dtype}.")

    def quant(self):
        """quant"""
        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|input0_weight", self.layer.weight)
        # quant weight
        w_scale, w_zp, q_weight = quant_tensor(self.layer.weight, self.w_quant_min, self.w_quant_max,
                                               self.cfg.weight_narrow_range, self.cfg.weight_symmetric,
                                               self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP,
                                               self.cfg.group_size, self.cfg.weight_quant_dtype,
                                               self.weight_quantizer_axis,
                                               is_transpose=self.transpose_b)
        if self.cfg.weight_quant_granularity == QuantGranularity.PER_CHANNEL:
            w_scale = np.squeeze(w_scale)
            w_zp = np.squeeze(w_zp)

        del self.layer.weight
        self.layer.weight = None
        self.weight = Parameter(q_weight.astype(dtype=dtype.int8))
        self.weight_scale = Parameter(Tensor(w_scale, dtype=self.compute_type))
        self.weight_offset = Parameter(Tensor(w_zp, dtype=dtype.int32))
        self.has_bias = self.layer.has_bias
        if self.has_bias:
            self.bias = self.layer.bias
            self.layer.bias = None

        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|output0_qweight", self.weight)
        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|output1_weight_scale", self.weight_scale)
        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|output2_weight_zp", self.weight_offset)

    def process(self):
        super().process()
        self.quant()
        self.cat_samples = None

    def quant_type_dict(self):
        """quant_type_dict"""
        type_ = ""
        if self.cfg.weight_quant_dtype == dtype.int8:
            type_ = QuantType.W8A16.value
        elif self.cfg.weight_quant_dtype == dtype.qint4x2:
            type_ = QuantType.W4A16.value
        quant_type = {
            self.weight_scale.name: type_,
            self.weight_offset.name: type_,
            self.weight.name: type_
        }
        if self.has_bias:
            quant_type.update({self.bias.name: type_})
        return quant_type
