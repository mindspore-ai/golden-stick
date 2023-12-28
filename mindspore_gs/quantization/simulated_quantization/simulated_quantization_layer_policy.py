# Copyright 2022 Huawei Technologies Co., Ltd
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
"""DefaultLayerPolicy."""

import abc
from typing import Optional
from functools import partial

from mindspore.nn import Cell
from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.layer_policy import LayerPolicy
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer
from .simulated_fake_quantizers import SimulatedFakeQuantizerPerChannel, SimulatedFakeQuantizerPerLayer
from .simulated_quantization_config import SimulatedQuantizationConfig
from .quant_cells.fake_quant_with_min_max_observer import QuantConfig as OpQuantConfig
from .quant_cells import Conv2dQuant, DenseQuant, Conv2dBnFoldQuantOneConv, Conv2dBnWithoutFoldQuant, \
    Conv2dBnFoldQuant, ActQuant


class SimulatedLayerPolicy(LayerPolicy, abc.ABC):
    """
    Derived class of LayerPolicy. Sim-QAT layer policy.
    Use linear perchannel fake quantizer as weight fake quantizer, linear perlayer fake quantizer as act fake quantizer.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` ``one_conv_fold``.
    """

    def __init__(self, weight_names: [], act_names: [],
                 config: SimulatedQuantizationConfig = SimulatedQuantizationConfig()):
        super(SimulatedLayerPolicy, self).__init__()
        self._config: SimulatedQuantizationConfig = config
        if config.weight_quant_dtype == QuantDtype.INT8:
            self._num_bits = 8
        else:
            raise TypeError("Only support int8 weight quant now!")
        if config.weight_per_channel:
            self._weight_quantizer_partial = partial(SimulatedFakeQuantizerPerChannel,
                                                     ema=False,
                                                     symmetric=config.weight_symmetric,
                                                     quant_dtype=config.weight_quant_dtype,
                                                     quant_delay=config.weight_quant_delay,
                                                     narrow_range=config.weight_narrow_range)
        else:
            self._weight_quantizer_partial = partial(SimulatedFakeQuantizerPerLayer, ema=False,
                                                     symmetric=config.weight_symmetric,
                                                     quant_dtype=config.weight_quant_dtype,
                                                     quant_delay=config.weight_quant_delay,
                                                     narrow_range=config.weight_narrow_range)
        if config.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._act_quantizer: Optional[FakeQuantizer] = SimulatedFakeQuantizerPerLayer(
            symmetric=config.act_symmetric, quant_dtype=config.act_quant_dtype, quant_delay=config.act_quant_delay,
            narrow_range=config.act_narrow_range)
        self._input_quantizer: Optional[FakeQuantizer] = SimulatedFakeQuantizerPerLayer(
            symmetric=config.act_symmetric, quant_dtype=config.act_quant_dtype, quant_delay=config.act_quant_delay,
            narrow_range=config.act_narrow_range)
        self._output_quantizer: Optional[FakeQuantizer] = SimulatedFakeQuantizerPerLayer(
            symmetric=config.act_symmetric, quant_dtype=config.act_quant_dtype, quant_delay=config.act_quant_delay,
            narrow_range=config.act_narrow_range)
        self._weight_names = weight_names
        self._act_names = act_names

    def get_weight_name_and_quantizers(self):
        return [(name, self._weight_quantizer_partial) for name in self._weight_names]

    def get_act_name_and_quantizers(self):
        return [(name, self._act_quantizer) for name in self._act_names]

    def get_input_quantizer(self) -> Optional[FakeQuantizer]:
        return self._input_quantizer

    def get_output_quantizer(self) -> Optional[FakeQuantizer]:
        return self._output_quantizer

    def set_output_not_insert_fq(self, index: Optional[int] = None):
        self._output_quantizer = None

    def get_config(self) -> SimulatedQuantizationConfig:
        return self._config

    def get_quantizer(self):
        return OpQuantConfig(self._weight_quantizer_partial, self._output_quantizer)

    @abc.abstractmethod
    def wrap_cell(self, handler: Cell) -> Cell:
        raise NotImplementedError


class ConvLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return Conv2dQuant(handler, self, self.get_quantizer())


class DenseLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return DenseQuant(handler, self, self.get_quantizer())


class ConvBnLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.ConvBn.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        if handler.has_bn:
            if self._config.bn_fold:
                if self._config.one_conv_fold:
                    conv_quant = Conv2dBnFoldQuantOneConv(handler, self, quant_config=self.get_quantizer())
                else:
                    conv_quant = Conv2dBnFoldQuant(handler, self, quant_config=self.get_quantizer(),
                                                   freeze_bn=self._config.freeze_bn)
            else:
                conv_quant = Conv2dBnWithoutFoldQuant(handler, self, quant_config=self.get_quantizer())
        else:
            conv_quant = Conv2dQuant(handler, self, self.get_quantizer())
        return conv_quant


class ActLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for activation layer.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return ActQuant(handler, self)
