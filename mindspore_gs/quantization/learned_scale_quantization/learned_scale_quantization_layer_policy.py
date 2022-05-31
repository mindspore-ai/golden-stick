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
"""learned scale quantization layer policy"""

from typing import Optional
from functools import partial
from mindspore.nn import Cell
from mindspore.nn.layer.quant import Conv2dQuant, DenseQuant, Conv2dBnFoldQuantOneConv, Conv2dBnWithoutFoldQuant, \
    Conv2dBnFoldQuant
from .learned_scale_fake_quantizers import LearnedScaleFakeQuantizerPerLayer, LearnedScaleFakeQuantizePerChannel
from .learned_scale_quantization_config import LearnedScaleQuantizationConfig
from ..simulated_quantization.simulated_quantization_layer_policy import SimulatedLayerPolicy
from ..fake_quantizer import FakeQuantizer
from ..quantize_wrapper_cell import QuantizeWrapperCell


class LearnedScaleQuantizationLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LSQ layer policy.
    """
    def __init__(self, weight_names: [], act_names: [],
                 config: LearnedScaleQuantizationConfig = LearnedScaleQuantizationConfig()):
        super(LearnedScaleQuantizationLayerPolicy, self).__init__(weight_names, act_names, config)
        if config.weight_per_channel:
            self._weight_quantizer_partial = partial(LearnedScaleFakeQuantizePerChannel,
                                                     quant_delay=config.weight_quant_delay,
                                                     num_bits=self._num_bits,
                                                     neg_trunc=config.weight_neg_trunc)
        else:
            self._weight_quantizer_partial = partial(LearnedScaleFakeQuantizerPerLayer,
                                                     quant_delay=config.weight_quant_delay,
                                                     num_bits=self._num_bits,
                                                     neg_trunc=config.weight_neg_trunc)
        if config.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._act_quantizer: Optional[FakeQuantizer] = LearnedScaleFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, num_bits=self._num_bits, neg_trunc=config.act_neg_trunc)
        self._input_quantizer: Optional[FakeQuantizer] = LearnedScaleFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, num_bits=self._num_bits)
        self._output_quantizer: Optional[FakeQuantizer] = LearnedScaleFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, num_bits=self._num_bits)


class ConvLayerPolicy(LearnedScaleQuantizationLayerPolicy):
    def wrap_cell(self, handler: Cell) -> Cell:
        conv_quant = Conv2dQuant.from_float(handler, self.get_quant_config())
        return QuantizeWrapperCell(conv_quant, self)


class DenseLayerPolicy(LearnedScaleQuantizationLayerPolicy):
    def wrap_cell(self, handler: Cell) -> Cell:
        dense_quant = DenseQuant.from_float(handler, self.get_quant_config())
        return QuantizeWrapperCell(dense_quant, self)


class ConvBnLayerPolicy(LearnedScaleQuantizationLayerPolicy):
    def wrap_cell(self, handler: Cell) -> Cell:
        if self._config.bn_fold:
            if self._config.one_conv_fold:
                conv_bn_quant = Conv2dBnFoldQuantOneConv.from_float(handler, self.get_quant_config())
            else:
                conv_bn_quant = Conv2dBnFoldQuant.from_float(handler, self.get_quant_config())
        else:
            conv_bn_quant = Conv2dBnWithoutFoldQuant.from_float(handler, self.get_quant_config())
        return QuantizeWrapperCell(conv_bn_quant, self)
