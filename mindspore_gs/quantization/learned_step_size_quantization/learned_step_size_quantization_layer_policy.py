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

from typing import Optional
from functools import partial
from mindspore.nn import Cell
from mindspore_gs.ops.nn import Conv2dQuant, DenseQuant, Conv2dBnFoldQuantOneConv, Conv2dBnWithoutFoldQuant, \
    Conv2dBnFoldQuant
from .learned_step_size_fake_quantizers import LearnedStepSizeFakeQuantizerPerLayer, \
    LearnedStepSizeFakeQuantizePerChannel
from .learned_step_size_quantization_config import LearnedStepSizeQuantizationConfig
from ..simulated_quantization.simulated_quantization_layer_policy import SimulatedLayerPolicy
from ..fake_quantizer import FakeQuantizer
from ..quantize_wrapper_cell import QuantizeWrapperCell


class LearnedStepSizeQuantizationLayerPolicy(SimulatedLayerPolicy):
    """
    Derived class of SimulatedLayerPolicy. LSQ layer policy.
    """

    def __init__(self, weight_names: [], act_names: [],
                 config: LearnedStepSizeQuantizationConfig = LearnedStepSizeQuantizationConfig()):
        super(LearnedStepSizeQuantizationLayerPolicy, self).__init__(weight_names, act_names, config)
        if config.weight_per_channel:
            self._weight_quantizer_partial = partial(LearnedStepSizeFakeQuantizePerChannel,
                                                     quant_delay=config.weight_quant_delay,
                                                     num_bits=self._num_bits,
                                                     neg_trunc=config.weight_neg_trunc,
                                                     symmetric=config.weight_symmetric,
                                                     narrow_range=config.weight_narrow_range)
        else:
            self._weight_quantizer_partial = partial(LearnedStepSizeFakeQuantizerPerLayer,
                                                     quant_delay=config.weight_quant_delay,
                                                     num_bits=self._num_bits,
                                                     neg_trunc=config.weight_neg_trunc,
                                                     symmetric=config.weight_symmetric,
                                                     narrow_range=config.weight_narrow_range)

        if config.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._act_quantizer: Optional[FakeQuantizer] = LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, num_bits=self._num_bits, neg_trunc=config.act_neg_trunc,
            symmetric=config.act_symmetric, narrow_range=config.act_narrow_range)
        self._input_quantizer: Optional[FakeQuantizer] = LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, num_bits=self._num_bits, symmetric=config.act_symmetric,
            narrow_range=config.act_narrow_range)
        self._output_quantizer: Optional[FakeQuantizer] = LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, num_bits=self._num_bits, symmetric=config.act_symmetric,
            narrow_range=config.act_narrow_range)


class LearnedStepSizeQuantizationConvLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        conv_quant = Conv2dQuant.from_conv2d(handler, self.get_quant_config())
        return QuantizeWrapperCell(conv_quant, self)


class LearnedStepSizeQuantizationDenseLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        dense_quant = DenseQuant.from_dense(handler, self.get_quant_config())
        return QuantizeWrapperCell(dense_quant, self)


class LearnedStepSizeQuantizationConvBnLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        if handler.has_bn:
            if self._config.bn_fold:
                if self._config.one_conv_fold:
                    conv_quant = Conv2dBnFoldQuantOneConv.from_convbn(handler, self.get_quant_config())
                else:
                    conv_quant = Conv2dBnFoldQuant.from_convbn(handler, self.get_quant_config(),
                                                               {"freeze_bn": self._config.freeze_bn})
            else:
                conv_quant = Conv2dBnWithoutFoldQuant.from_convbn(handler, self.get_quant_config())
        else:
            conv_quant = Conv2dQuant.from_convbn(handler, self.get_quant_config())
        return QuantizeWrapperCell(conv_quant, self)
