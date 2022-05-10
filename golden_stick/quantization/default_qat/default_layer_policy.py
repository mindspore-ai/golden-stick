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
from typing import Optional
from functools import partial
from mindspore.nn import Cell
from mindspore.nn.layer.quant import QuantConfig, Conv2dQuant, DenseQuant, Conv2dBnFoldQuantOneConv
from ..layer_policy import LayerPolicy
from ..quantize_wrapper_cell import QuantizeWrapperCell
from ..fake_quantizer import FakeQuantizer
from .default_fake_quantizer import DefaultFakeQuantizerPerChannel, DefaultFakeQuantizerPerLayer, \
    LearnedFakeQuantizerPerLayer


class DefaultLayerPolicy(LayerPolicy):
    """
    Derived class of LayerPolicy. Default layer policy.
    Use linear perchannel fake quantizer as weight fake quantizer, linear perlayer fake quantizer as act fake quantizer.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` ``one_conv_fold``.
    """

    def __init__(self, weight_names: [], act_names: [], config=None):
        if config is None:
            config = {}
        self._weight_quantizer = partial(DefaultFakeQuantizerPerChannel, num_bits=8)
        self._act_quantizer = DefaultFakeQuantizerPerLayer()
        self._input_quantizer: Optional[FakeQuantizer] = DefaultFakeQuantizerPerLayer()
        self._output_quantizer: Optional[FakeQuantizer] = DefaultFakeQuantizerPerLayer()
        self._weight_names = weight_names
        self._act_names = act_names
        self._input_num = 0
        self._inputs_insert_fq = []

    def get_weight_name_and_quantizers(self):
        return [(name, self._weight_quantizer) for name in self._weight_names]

    def get_act_name_and_quantizers(self):
        return [(name, self._act_quantizer) for name in self._act_names]

    def get_input_quantizer(self) -> Optional[FakeQuantizer]:
        return self._input_quantizer

    def get_output_quantizer(self) -> Optional[FakeQuantizer]:
        return self._output_quantizer

    def set_input_number(self, input_num: int):
        self._input_num = input_num
        for _ in range(0, self._input_num):
            self._inputs_insert_fq.append(True)

    def set_input_not_insert_fq(self, index: Optional[int] = None):
        if index is None:
            for i in range(0, self._input_num):
                self._inputs_insert_fq[i] = False
        else:
            if index >= self._input_num:
                raise RuntimeError("Index out of range of input number")
            self._inputs_insert_fq[index] = False

    def get_input_need_insert_fq(self):
        return self._inputs_insert_fq

    def set_output_not_insert_fq(self, index: Optional[int] = None):
        self._output_quantizer = None

    def get_quant_config(self):
        return QuantConfig(self._weight_quantizer, self._act_quantizer)

    def wrap_cell(self, handler: Cell) -> Cell:
        return QuantizeWrapperCell(handler, self)


class ActivationLayerPolicy(DefaultLayerPolicy):
    def __init__(self, insert_before_input=False, insert_after_output=True):
        super().__init__([], [])
        self._input_quantizer: Optional[FakeQuantizer] = LearnedFakeQuantizerPerLayer()
        self._output_quantizer: Optional[FakeQuantizer] = LearnedFakeQuantizerPerLayer()
        self._insert_before_input = insert_before_input
        self._insert_after_output = insert_after_output


class ConvLayerPolicy(DefaultLayerPolicy):
    def wrap_cell(self, handler: Cell) -> Cell:
        conv_quant = Conv2dQuant.from_float(handler, self.get_quant_config())
        return QuantizeWrapperCell(conv_quant, self)


class DenseLayerPolicy(DefaultLayerPolicy):
    def wrap_cell(self, handler: Cell) -> Cell:
        dense_quant = DenseQuant.from_float(handler, self.get_quant_config())
        return QuantizeWrapperCell(dense_quant, self)


class ConvBnLayerPolicy(DefaultLayerPolicy):
    def wrap_cell(self, handler: Cell) -> Cell:
        conv_bn_quant = Conv2dBnFoldQuantOneConv.from_float(handler, self.get_quant_config())
        return QuantizeWrapperCell(conv_bn_quant, self)
