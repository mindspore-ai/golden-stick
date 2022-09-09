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
"""SlbLayerPolicy."""

from typing import Optional
from functools import partial

from mindspore.nn import Cell
from mindspore.nn.layer.quant import QuantConfig as OpQuantConfig
from ..layer_policy import LayerPolicy
from ..quantize_wrapper_cell import QuantizeWrapperCell
from ..fake_quantizer import FakeQuantizer
from .slb_fake_quantizer import SlbFakeQuantizerPerLayer
from .slb_quant import Conv2dSlbQuant
from .slb_quant_config import SlbQuantConfig


class SlbLayerPolicy(LayerPolicy):
    """
    Derived class of LayerPolicy. slb layer policy.
    Use slb perlayer fake quantizer as weight fake quantizer.

    Supported Config:
        ``quant_dtype``.
    """

    def __init__(self, weight_names: [], act_names: [], config: SlbQuantConfig = SlbQuantConfig()):
        self._config = config
        weight_num_bits = config.weight_quant_dtype.num_bits
        if weight_num_bits not in [1, 2, 4]:
            raise TypeError("Only support int4|int2|int1 weight quant now!")

        self._weight_quantizer_partial = partial(SlbFakeQuantizerPerLayer, num_bits=weight_num_bits)
        self._act_quantizer: Optional[FakeQuantizer] = None
        self._input_quantizer: Optional[FakeQuantizer] = None
        self._output_quantizer: Optional[FakeQuantizer] = None
        self._weight_names = weight_names
        self._act_names = act_names
        self._input_num = 0
        self._inputs_insert_fq = []

    def get_weight_name_and_quantizers(self):
        return [(name, self._weight_quantizer_partial) for name in self._weight_names]

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
        return OpQuantConfig(self._weight_quantizer_partial, self._act_quantizer)

    def wrap_cell(self, handler: Cell) -> Cell:
        return QuantizeWrapperCell(handler, self)


class ConvLayerPolicy(SlbLayerPolicy):
    def wrap_cell(self, handler: Cell) -> Cell:
        conv_quant = Conv2dSlbQuant.from_float(handler, self.get_quant_config(), self._config.weight_quant_dtype)
        return QuantizeWrapperCell(conv_quant, self)
