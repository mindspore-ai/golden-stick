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
from typing import Optional
from functools import partial
from mindspore.nn import Cell
from mindspore_gs.quantization.simulated_quantization.quant_cells import Conv2dQuant, DenseQuant, \
    Conv2dBnFoldQuantOneConv, Conv2dBnWithoutFoldQuant, Conv2dBnFoldQuant
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
        if config.weight_per_channel:
            self._weight_quantizer_partial = partial(LearnedStepSizeFakeQuantizePerChannel,
                                                     quant_delay=config.weight_quant_delay,
                                                     quant_dtype=config.weight_quant_dtype,
                                                     neg_trunc=config.weight_neg_trunc,
                                                     symmetric=config.weight_symmetric,
                                                     narrow_range=config.weight_narrow_range)
        else:
            self._weight_quantizer_partial = partial(LearnedStepSizeFakeQuantizerPerLayer,
                                                     quant_delay=config.weight_quant_delay,
                                                     quant_dtype=config.weight_quant_dtype,
                                                     neg_trunc=config.weight_neg_trunc,
                                                     symmetric=config.weight_symmetric,
                                                     narrow_range=config.weight_narrow_range)

        if config.act_per_channel:
            raise NotImplementedError("act quant only support perlayer now!")
        self._act_quantizer: Optional[FakeQuantizer] = LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, quant_dtype=config.act_quant_dtype, neg_trunc=config.act_neg_trunc,
            symmetric=config.act_symmetric, narrow_range=config.act_narrow_range)
        self._input_quantizer: Optional[FakeQuantizer] = LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, quant_dtype=config.act_quant_dtype, symmetric=config.act_symmetric,
            narrow_range=config.act_narrow_range)
        self._output_quantizer: Optional[FakeQuantizer] = LearnedStepSizeFakeQuantizerPerLayer(
            quant_delay=config.act_quant_delay, quant_dtype=config.act_quant_dtype, symmetric=config.act_symmetric,
            narrow_range=config.act_narrow_range)

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
        return Conv2dQuant(handler, self, quant_config=self.get_quantizer())


class LearnedStepSizeQuantizationDenseLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
    """

    def wrap_cell(self, handler: Cell) -> Cell:
        return DenseQuant(handler, self, quant_config=self.get_quantizer())


class LearnedStepSizeQuantizationConvBnLayerPolicy(LearnedStepSizeQuantizationLayerPolicy):
    """
    Derived class of LearnedStepSizeQuantizationLayerPolicy. LayerPolicy used for nn.Conv2d.
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
            conv_quant = Conv2dQuant(handler, self, quant_config=self.get_quantizer())
        return conv_quant
