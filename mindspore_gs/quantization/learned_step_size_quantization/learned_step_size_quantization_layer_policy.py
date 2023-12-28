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
"""learned step size quantization layer policy"""

import abc

from mindspore.nn import Cell
from mindspore_gs.quantization.ops.nn import Conv2dQuant, DenseQuant, Conv2dBnFoldQuantOneConv, \
    Conv2dBnWithoutFoldQuant, Conv2dBnFoldQuant
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer
from mindspore_gs.quantization.simulated_quantization.simulated_quantization_layer_policy import SimulatedLayerPolicy
from .learned_step_size_fake_quantizers import LearnedStepSizeFakeQuantizerPerLayer, \
    LearnedStepSizeFakeQuantizePerChannel
from .learned_step_size_quantization_config import LearnedStepSizeQuantizationConfig


class LearnedStepSizeQuantizationLayerPolicy(SimulatedLayerPolicy, abc.ABC):
    """
    Derived class of SimulatedLayerPolicy. LSQ layer policy.
    """

    def __init__(self, weight_names: [], act_names: [],
                 config: LearnedStepSizeQuantizationConfig = LearnedStepSizeQuantizationConfig()):
        super(LearnedStepSizeQuantizationLayerPolicy, self).__init__(weight_names, act_names, config)
        self._config: LearnedStepSizeQuantizationConfig = config
        if config.act_per_channel:
            raise ValueError("act quant only support perlayer now!")

    def get_weight_quantizer(self, weight_name="", **kwargs) -> FakeQuantizer:
        if self._config.weight_per_channel:
            channel_axis = kwargs.get('channel_axis', None)
            num_channels = kwargs.get('num_channels', None)
            if channel_axis is None:
                raise RuntimeError("Please provide channel axis of weight for per-channel weight quantize.")
            if num_channels is None:
                raise RuntimeError("Please provide channel number of weight for per-channel weight quantize.")
            weight_quantizer = \
                LearnedStepSizeFakeQuantizePerChannel(symmetric=self._config.weight_symmetric,
                                                      quant_dtype=self._config.weight_quant_dtype,
                                                      neg_trunc=self._config.weight_neg_trunc,
                                                      quant_delay=self._config.weight_quant_delay,
                                                      narrow_range=self._config.weight_narrow_range,
                                                      channel_axis=channel_axis, num_channels=num_channels)
        else:
            weight_quantizer = LearnedStepSizeFakeQuantizerPerLayer(symmetric=self._config.weight_symmetric,
                                                                    quant_dtype=self._config.weight_quant_dtype,
                                                                    quant_delay=self._config.weight_quant_delay,
                                                                    narrow_range=self._config.weight_narrow_range)
        return weight_quantizer

    def _get_input_quantizer(self, input_index=-1, **kwargs) -> FakeQuantizer:
        return LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=self._config.act_quant_delay, quant_dtype=self._config.act_quant_dtype,
            symmetric=self._config.act_symmetric,
            narrow_range=self._config.act_narrow_range)

    def _get_output_quantizer(self, **kwargs) -> FakeQuantizer:
        return LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=self._config.act_quant_delay, quant_dtype=self._config.act_quant_dtype,
            symmetric=self._config.act_symmetric,
            narrow_range=self._config.act_narrow_range)

    def get_config(self) -> LearnedStepSizeQuantizationConfig:
        return self._config

    @abc.abstractmethod
    def wrap_cell(self, handler: Cell) -> Cell:
        raise NotImplementedError


class LearnedStepSizeQuantizationConvLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return Conv2dQuant(handler, self)


class LearnedStepSizeQuantizationDenseLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return DenseQuant(handler, self)


class LearnedStepSizeQuantizationConvBnLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
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
