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
"""QuantizeWrapperActivation."""

from mindspore.nn import Cell
from .fake_quantizer import FakeQuantizer


class QuantizeWrapperActivation(Cell):
    """
    Decorator of Activation Cell class for decorate a cell to a quant-cell with fake-quant algorithm.

    Args:
        act (Cell): normal cell to be wrapped.
        pre_quantizer (Quantizer): Define how weight data to be fake-quant.
        post_quantizer (Quantizer): Define how activation data to be fake-quant.
    """

    def __init__(self, act: Cell, pre_quantizer: FakeQuantizer = None, post_quantizer: FakeQuantizer = None):
        super().__init__()
        self._handler: callable = act
        self._pre_quantizer = pre_quantizer
        self._post_quantizer = post_quantizer

    def construct(self, x):
        if self._pre_quantizer is not None:
            quant_param = self._pre_quantizer.compute_quant_param(x)
            x = self._pre_quantizer.fake_quant(x, quant_param)
        x = self._handler(x)
        if self._post_quantizer is not None:
            quant_param = self._post_quantizer.compute_quant_param(x)
            x = self._post_quantizer.fake_quant(x, quant_param)
        return x
