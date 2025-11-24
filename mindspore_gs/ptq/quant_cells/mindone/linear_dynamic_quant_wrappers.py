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

from mindspore import nn, mint
from mindspore import dtype

from mindspore_gs.ptq.ptq_config import QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.algo_modules.quantizer import Quantizer
from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from mindspore_gs.ptq.utils import QuantType
from .linear_weight_quant_wrappers import WeightQuantLinearCell


class DynamicQuantLinearCell(WeightQuantLinearCell):
    """WeightQuantLinearCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class DynamicA8W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype == dtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TOKEN

        Quantizer.reg_layer_map(nn.Dense, DynamicQuantLinearCell, DynamicA8W8Checker())
        Quantizer.reg_layer_map(mint.nn.Linear, DynamicQuantLinearCell, DynamicA8W8Checker())

    def _quant_info(self):
        res = super()._quant_info()
        if self.cfg.act_quant_dtype == dtype.int8:
            return f'{res}-A8-{str(self.cfg.act_quant_granularity)}'
        raise RuntimeError(f"Unexpected act_quant_dtype: {self.cfg.act_quant_dtype}.")

    def quant_type_dict(self):
        """quant_type_dict"""
        quant_type = {
            self.weight_scale.name: QuantType.W8A8_DYNAMIC.value,
            self.weight_offset.name: QuantType.W8A8_DYNAMIC.value,
            self.weight.name: QuantType.W8A8_DYNAMIC.value
        }
        if self.has_bias:
            quant_type.update({self.bias.name: QuantType.W4A8_DYNAMIC.value})
        return quant_type
