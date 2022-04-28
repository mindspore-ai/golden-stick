# Copyright 2021 Huawei Technologies Co., Ltd
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
from .fake_quantizer import FakeQuantizer
from .layer_policy import LayerPolicy
from .quantize_wrapper_act import QuantizeWrapperActivation


class QuantizeWrapperCell(Cell):
    """
    Decorator of Activation Cell class for decorate a cell to a quant-cell with fake-quant algorithm.

    Args:
        handler (Cell): normal cell to be wrapped.
        layer_policy (FakeQuantizer): Define how weight data to be fake-quant.
    """

    def __init__(self, handler: Cell, layer_policy: LayerPolicy):
        super().__init__()
        self._handler: Cell = handler
        self._policy = layer_policy
        self._w_scale = 1.0
        self._w_zp = 0
        self._o_scale = 1.0
        self._o_zp = 0
        self._input_quantizer = self._policy.get_input_quantizer()
        self._output_quantizer = self._policy.get_output_quantizer()
        self._input_insert_quantizer = self._policy.get_input_need_insert_fq()
        # fake-quant weight
        for weight_name, quantizer in self._policy.get_weight_name_and_quantizers():
            assert weight_name is not None
            assert quantizer is not None
            weight = getattr(self._handler, weight_name)
            fq_data = quantizer(weight)
            setattr(self._handler, weight_name, fq_data)

    def construct(self, *inputs, **kwargs):
        """
        Defines the computation of QuantizeWrapperCell to be performed.

        Returns:
            Tensor, returns the computed result.
        """

        # fake-quant input
        fq_inputs = inputs
        # if self._input_quantizer is None:
        #     fq_inputs = inputs
        # else:
        #     fq_inputs = []
        #     input_len = len(inputs)
        #     for i in range(0, input_len):
        #         ori_input = inputs[i]
        #         if self._input_insert_quantizer[i]:
        #             fq_inputs.append(self._input_quantizer(ori_input))
        #         else:
        #             fq_inputs.append(ori_input)

        # forward handler
        outputs = self._handler(*fq_inputs, **kwargs)

        # fake-quant output
        if self._output_quantizer is None:
            return outputs
        if not isinstance(outputs, list):
            return self._output_quantizer(outputs)
        output_len = len(outputs)
        if output_len == 0:
            return outputs
        else:
            fq_outputs = []
            for i in range(0, output_len):
                ori_output = outputs[i]
                fq_outputs.append(self._output_quantizer(ori_output))
            return fq_outputs
