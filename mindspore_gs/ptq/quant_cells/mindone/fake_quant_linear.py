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

from mindspore import nn, Parameter, dtype, mint
from mindspore import ops as msops
from mindspore.common.initializer import initializer

from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from mindspore_gs.ptq.algo_modules import Quantizer
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq_config import QuantGranularity
from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell
from .fake_quant_base import (FakeQuantLinearCell,
                              DeQuant,
                              FakeQuant,
                              DynamicFakeQuant)


class FakeQuantA16WxWrapper(QuantCell):
    """FakeQuantA16WxWrapper"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class FakeQuantChecker(Checker):
            def check(self, config: InnerPTQConfig):
                support_dtype = [dtype.int8, dtype.qint4x2]
                return config.weight_quant_dtype in support_dtype and config.act_quant_dtype is None

        Quantizer.reg_fake_quant_layer_map(nn.Dense, FakeQuantA16WxWrapper, FakeQuantChecker())
        Quantizer.reg_fake_quant_layer_map(mint.nn.Linear, FakeQuantA16WxWrapper, FakeQuantChecker())

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
        self.group_size = cfg.group_size
        self.weight = Parameter(initializer("zeros", linear.weight.shape, dtype.int8))
        if linear.has_bias:
            self.bias = Parameter(initializer("zeros", (linear.weight.shape[0],), self.compute_dtype))
        if self.group_size > 0:
            output_channels, input_channels = linear.weight.shape
            self.weight_scale = Parameter(initializer("ones", (input_channels // self.group_size, output_channels),
                                                      self.compute_dtype))
            self.weight_offset = Parameter(initializer("zeros", (input_channels // self.group_size, output_channels),
                                                       dtype.int32))
        else:
            self.weight_scale = Parameter(initializer("ones", (linear.weight.shape[0],), self.compute_dtype))
            self.weight_offset = Parameter(initializer("zeros", (linear.weight.shape[0],), dtype.int32))

        self.de_quant = DeQuant(self.compute_dtype)
        self.layer.weight = None
        if linear.has_bias:
            self.layer.bias = None

    def dequant_input(self, x, weight):
        """process input"""
        if self.group_size > 0:
            weight_scale = msops.repeat_elements(self.weight_scale, rep=self.group_size, axis=0).T
            weight_offset = msops.repeat_elements(self.weight_offset, rep=self.group_size, axis=0).T
        else:
            weight_scale = self.weight_scale.reshape((-1, 1))
            weight_offset = self.weight_offset.reshape((-1, 1))
        weight = self.de_quant(weight, weight_scale, weight_offset)
        return x, weight


class FakeQuantW8A8Wrapper(QuantCell):
    """FakeQuantWrapper"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class FakeQuantChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype == dtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TENSOR

        Quantizer.reg_fake_quant_layer_map(nn.Dense, FakeQuantW8A8Wrapper, FakeQuantChecker())
        Quantizer.reg_fake_quant_layer_map(mint.nn.Linear, FakeQuantW8A8Wrapper, FakeQuantChecker())

    def _quant_info(self) -> str:
        return 'FakeQuant'

    def add_hook(self, experimental=False):
        pass

    def remove_hook(self, experimental=False):
        pass

    def deploy(self):
        return FakeQuantW8A8LinearCell(self.layer_name, self.layer, self.context, self.cfg)


class FakeQuantW8A8LinearCell(FakeQuantLinearCell):
    """FakeQuantW8A8LinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: nn.Cell, context, cfg: InnerPTQConfig):
        super().__init__(layer_name, linear, context, cfg)
        self.input_scale = Parameter(initializer("ones", (1,), self.compute_dtype))
        self.input_offset = Parameter(initializer("zeros", (1,), dtype.int32))
        self.weight_scale = Parameter(initializer("ones", (linear.weight.shape[0],), self.compute_dtype))
        self.weight_offset = Parameter(initializer("zeros", (linear.weight.shape[0],), dtype.int32))
        self.weight = Parameter(initializer("zeros", linear.weight.shape, dtype.int8))
        if linear.has_bias:
            self.bias = Parameter(initializer("zeros", (linear.weight.shape[0],), self.compute_dtype))
        self.de_quant = DeQuant(self.compute_dtype)
        self.fake_quant = FakeQuant(dtype.int8, self.compute_dtype)
        self.layer.weight = None
        if linear.has_bias:
            self.layer.bias = None

    def dequant_input(self, x, weight):
        """process input"""
        weight_scale = self.weight_scale.reshape((-1, 1))
        weight_offset = self.weight_offset.reshape((-1, 1))
        weight = self.de_quant(weight, weight_scale, weight_offset)
        x = self.fake_quant(x, self.input_scale, self.input_offset)
        return x, weight

class FakeQuantW8A8DynamicWrapper(QuantCell):
    """FakeQuantW8A8DynamicWrapper"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class FakeQuantChecker(Checker):
            def check(self, config: InnerPTQConfig):
                weight_support_dtype = [dtype.int8, dtype.qint4x2]
                return config.weight_quant_dtype in weight_support_dtype and \
                    config.act_quant_dtype == dtype.int8 and \
                        config.act_quant_granularity is QuantGranularity.PER_TOKEN

        Quantizer.reg_fake_quant_layer_map(nn.Dense, FakeQuantW8A8DynamicWrapper, FakeQuantChecker())
        Quantizer.reg_fake_quant_layer_map(mint.nn.Linear, FakeQuantW8A8DynamicWrapper, FakeQuantChecker())

    def _quant_info(self) -> str:
        return 'FakeQuant'

    def add_hook(self, experimental=False):
        pass

    def remove_hook(self, experimental=False):
        pass

    def deploy(self):
        return FakeQuantW8A8DynamicLinearCell(self.layer_name, self.layer, self.context, self.cfg)


class FakeQuantW8A8DynamicLinearCell(FakeQuantLinearCell):
    """FakeQuantW8A8DynamicLinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: nn.Cell, context, cfg: InnerPTQConfig):
        super().__init__(layer_name, linear, context, cfg)
        self.weight_scale = Parameter(initializer("ones", (linear.weight.shape[0],), self.compute_dtype))
        self.weight_offset = Parameter(initializer("zeros", (linear.weight.shape[0],), dtype.int32))
        self.weight = Parameter(initializer("zeros", linear.weight.shape, dtype.int8))
        if linear.has_bias:
            self.bias = Parameter(initializer("zeros", (linear.weight.shape[0],), self.compute_dtype))
        self.fake_quant = DynamicFakeQuant(dtype.int8, self.compute_dtype)
        self.de_quant = DeQuant(self.compute_dtype)

        self.layer.weight = None
        if linear.has_bias:
            self.layer.bias = None

    def dequant_input(self, x, weight):
        """process input"""
        # for moe matmul in the mindone, x shape may be(0, ic)
        if x.shape[0] == 0:
            x = msops.cast(x, self.compute_dtype)
        else:
            x = self.fake_quant(x)
        weight_scale = self.weight_scale.reshape((-1, 1))
        weight_offset = self.weight_offset.reshape((-1, 1))
        weight = self.de_quant(weight, weight_scale, weight_offset)
        return x, weight
