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


from mindspore import nn, Parameter, dtype
from mindspore.common.initializer import initializer

from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq_config import PrecisionRecovery
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from .fake_quant_base import (FakeQuantLinearCell,
                              DeQuant)


class FakeQuantA16WxWrapper(WrapperCell):
    """FakeQuantA16WxWrapper"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class FakeQuantChecker(Checker):
            def check(self, config: InnerPTQConfig):
                support_dtype = [dtype.int8, dtype.qint4x2]
                return (config.weight_quant_dtype in support_dtype and config.act_quant_dtype is None
                        and config.precision_recovery == PrecisionRecovery.NONE)

        Quantizer.reg_fake_quant_layer_map(nn.Dense, FakeQuantA16WxWrapper, FakeQuantChecker())

    def _quant_info(self) -> str:
        if self.cfg.weight_quant_dtype == dtype.int8:
            return f'FakeQuant-W8-{str(self.cfg.weight_quant_granularity)}'
        if self.cfg.weight_quant_dtype == dtype.qint4x2:
            return f'FakeQuant-W4-{str(self.cfg.weight_quant_granularity)}'
        raise RuntimeError(f"Unexpected weight_quant_dtype: {self.cfg.weight_quant_dtype}.")

    def add_hook(self, experimental=False):
        pass

    def remove_hook(self, experimental=False):
        pass

    def deploy(self):
        return FakeQuantA16WxLinearCell(self.layer_name, self.layer, self.context, self.cfg)


class FakeQuantA16WxLinearCell(FakeQuantLinearCell):
    """FakeQuantA16WxLinearCell"""
    def __init__(self, layer_name, linear: nn.Cell, context, cfg: InnerPTQConfig):
        super().__init__(layer_name, linear, context, cfg)
        self.weight = Parameter(initializer("zeros", linear.weight.shape, self.cfg.weight_quant_dtype))
        self.weight_scale = Parameter(initializer("ones", (linear.weight.shape[0],), self.compute_dtype))
        self.weight_offset = Parameter(initializer("zeros", (linear.weight.shape[0],), dtype.int32))
        if linear.has_bias:
            self.bias = Parameter(initializer("zeros", (linear.weight.shape[0],), self.compute_dtype))

        self.de_quant = DeQuant(self.compute_dtype)
        self.layer.weight = None
        if linear.has_bias:
            self.layer.bias = None

    def dequant_input(self, x, weight):
        """process input"""
        weight_scale = self.weight_scale.reshape((-1, 1))
        weight_offset = self.weight_offset.reshape((-1, 1))
        weight = self.de_quant(weight, weight_scale, weight_offset)
        return x, weight
