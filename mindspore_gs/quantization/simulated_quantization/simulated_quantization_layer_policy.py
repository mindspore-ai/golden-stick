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

from mindspore.nn import Cell
from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer
from mindspore_gs.quantization.ops.nn import Conv2dQuant, DenseQuant, Conv2dBnFoldQuantOneConv, \
    Conv2dBnWithoutFoldQuant, Conv2dBnFoldQuant, ActQuant
from .simulated_fake_quantizers import SimulatedFakeQuantizerPerChannel, SimulatedFakeQuantizerPerLayer
from .simulated_quantization_config import SimulatedQuantizationConfig


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
        if config.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._weight_names = weight_names
        self._act_names = act_names

    def get_weight_quantizer(self, weight_name="", perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        if self._config.weight_per_channel:
            channel_axis = perchannel_args.channel_axis
            num_channels = perchannel_args.num_channels
            if channel_axis == -1:
                raise RuntimeError("Please provide channel axis of weight for per-channel weight quantize.")
            if num_channels == -1:
                raise RuntimeError("Please provide channel number of weight for per-channel weight quantize.")
            weight_quantizer = SimulatedFakeQuantizerPerChannel(ema=False, symmetric=self._config.weight_symmetric,
                                                                quant_dtype=self._config.weight_quant_dtype,
                                                                quant_delay=self._config.weight_quant_delay,
                                                                narrow_range=self._config.weight_narrow_range,
                                                                channel_axis=channel_axis, num_channels=num_channels)
        else:
            weight_quantizer = SimulatedFakeQuantizerPerLayer(ema=False, symmetric=self._config.weight_symmetric,
                                                              quant_dtype=self._config.weight_quant_dtype,
                                                              quant_delay=self._config.weight_quant_delay,
                                                              narrow_range=self._config.weight_narrow_range)
        return weight_quantizer

    def _get_input_quantizer(self, input_index=-1, perchannel_args: PerChannelArgs = PerChannelArgs(),
                             **kwargs) -> FakeQuantizer:
        return SimulatedFakeQuantizerPerLayer(symmetric=self._config.act_symmetric,
                                              quant_dtype=self._config.act_quant_dtype,
                                              quant_delay=self._config.act_quant_delay,
                                              narrow_range=self._config.act_narrow_range)

    def _get_output_quantizer(self, perchannel_args: PerChannelArgs = PerChannelArgs(), **kwargs) -> FakeQuantizer:
        return SimulatedFakeQuantizerPerLayer(symmetric=self._config.act_symmetric,
                                              quant_dtype=self._config.act_quant_dtype,
                                              quant_delay=self._config.act_quant_delay,
                                              narrow_range=self._config.act_narrow_range)

    def get_config(self) -> SimulatedQuantizationConfig:
        return self._config

    @abc.abstractmethod
    def wrap_cell(self, handler: Cell) -> Cell:
        raise NotImplementedError


class ConvLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return Conv2dQuant(handler, self)


class DenseLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.Dense.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return DenseQuant(handler, self)


class ConvBnLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for nn.ConvBn.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        if handler.has_bn:
            if self._config.bn_fold:
                if self._config.one_conv_fold:
                    conv_quant = Conv2dBnFoldQuantOneConv(handler, self)
                else:
                    conv_quant = Conv2dBnFoldQuant(handler, self, freeze_bn=self._config.freeze_bn)
            else:
                conv_quant = Conv2dBnWithoutFoldQuant(handler, self)
        else:
            conv_quant = Conv2dQuant(handler, self)
        return conv_quant


class ActLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LayerPolicy used for activation layer.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return ActQuant(handler, self)
