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
"""QuantizeWrapperCell."""

from mindspore.nn import Cell
from .layer_policy import LayerPolicy


class QuantizeWrapperCell(Cell):
    """
    Decorator of normal Cell class for decorate a cell to a quant-cell with fake-quant algorithm.

    Args:
        act (Cell): normal cell to be wrapped.
        layer_policy (FakeQuantizer): Define how input data and output data to be fake-quant.
    """

    def __init__(self, handler: Cell, layer_policy: LayerPolicy):
        super().__init__()
        self._handler: Cell = handler
        self._policy = layer_policy
        self._input_quantizer = self._policy.get_input_quantizer()
        self._output_quantizer = self._policy.get_output_quantizer()
        self._inputs_insert_fq = self._policy.get_input_need_insert_fq()

    def get_handler(self):
        return self._handler

    def get_input_quantizer(self):
        return self._input_quantizer

    def get_output_quantizer(self):
        return self._output_quantizer

    def construct(self, *inputs):
        """
        Defines the computation of QuantizeWrapperCell to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        # fake-quant input, forward handler
        if self._input_quantizer is None:
            outputs = self._handler(*inputs)
        else:
            if len(self._inputs_insert_fq) > len(inputs):
                raise ValueError("The num of cell inputs is incorrect.")
            fq_inputs = []
            for i in range(0, len(self._inputs_insert_fq)):
                if self._inputs_insert_fq[i]:
                    ori_input = inputs[i]
                    fq_inputs.append(self._input_quantizer(ori_input))
                else:
                    fq_inputs.append(inputs[i])
            outputs = self._handler(*fq_inputs)

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
