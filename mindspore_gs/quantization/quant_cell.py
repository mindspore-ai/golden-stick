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
"""
QuantCell, wrap objected cell with fake-quantizer, LinearQuantCell for example..
"""
import abc

from mindspore.nn.cell import Cell
from mindspore_gs import Backend
from mindspore_gs.quantization.layer_policy import LayerPolicy
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer, FakeQuantParamCell


class QuantCell(Cell):
    """
    Decorator of normal Cell class for decorate a cell to a quant-cell with fake-quant algorithm.

    Also provide `convert` method to convert self into a standard quant cell of MindSpore.

    Args:
        handler (Cell): normal cell to be wrapped.
        policy (LayerPolicy): Define how fake-quant handler Cell.
    """
    def __init__(self, handler: Cell, policy: LayerPolicy):
        super().__init__()
        self._handler = handler
        self._policy = policy
        self._input_quantizer = None
        self._output_quantizer = None
        self._inputs_insert_fq = None
        if self._policy:
            self._input_quantizer: FakeQuantizer = self._policy.get_input_quantizer()
            self._output_quantizer: FakeQuantizer = self._policy.get_output_quantizer()
            self._inputs_insert_fq = self._policy.get_input_need_insert_fq()

    def handler(self):
        return self._handler

    def input_quantizer(self):
        return self._input_quantizer

    def output_quantizer(self):
        return self._output_quantizer

    @abc.abstractmethod
    def weight_quantizer(self):
        raise NotImplementedError

    # pylint: disable=W0613
    def convert(self, backend: Backend = Backend.MS):
        if self._input_quantizer:
            self._input_quantizer: FakeQuantParamCell = self._input_quantizer.convert_to_fakequantparam()
        if self._output_quantizer:
            self._output_quantizer: FakeQuantParamCell = self._output_quantizer.convert_to_fakequantparam()

    # pylint: disable=arguments-differ
    @abc.abstractmethod
    def core_construct(self, *args):
        raise NotImplementedError

    def construct(self, *inputs):
        """
        override construct of Cell.
        """
        if self._input_quantizer is None:
            outputs = self.core_construct(*inputs)
        else:
            if len(self._inputs_insert_fq) != len(inputs):
                raise ValueError(f"The num of cell inputs is incorrect, set input number: "
                                 f"{len(self._inputs_insert_fq)}, real input number: {len(inputs)}")
            fq_inputs = []
            for i in range(0, len(self._inputs_insert_fq)):
                if self._inputs_insert_fq[i]:
                    ori_input = inputs[i]
                    fq_inputs.append(self._input_quantizer(ori_input))
                else:
                    fq_inputs.append(inputs[i])
            outputs = self.core_construct(*fq_inputs)

        # fake-quant output
        if self._output_quantizer is None:
            return outputs
        if not isinstance(outputs, list):
            return self._output_quantizer(outputs)
        fq_outputs = []
        for i in range(0, len(outputs)):
            ori_output = outputs[i]
            fq_outputs.append(self._output_quantizer(ori_output))
        return fq_outputs
