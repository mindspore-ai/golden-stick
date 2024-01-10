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
"""test interfaces of rtn."""
import os
import sys
import numpy as np

import pytest
from mindspore import context, Parameter, dtype, nn, Tensor
from mindspore_gs import Backend
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.ptq.quant_cells import LinearQuant
from mindspore_gs.ptq.convert_utils import AntiQuantCell
from mindformers.modules import Linear

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.models.llama2 import llama2
from tests.st.test_utils import check_network_contain_layer


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_llama2_woq_apply_convert():
    """
    Feature: RoundToNearestPTQ A16W8 algorithm.
    Description: Apply A16W8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.
    """

    context.set_context(device_target="Ascend")
    network = llama2(8, 512, 1024, 2)
    assert check_network_contain_layer(network, Linear)
    ptq = RTN()
    ptq.set_weight_only_quant(True)
    quant_network = ptq.apply(network)
    assert not check_network_contain_layer(quant_network, Linear, (LinearQuant,))
    assert check_network_contain_layer(quant_network, LinearQuant)
    ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
    for _, cell in ascend_network.name_cells().items():
        if not isinstance(cell, LinearQuant):
            continue
        linear: LinearQuant = cell
        assert not linear.input_quantizer()
        assert not linear.output_quantizer()
        assert isinstance(linear.weight_quantizer(), AntiQuantCell)
        weight: Parameter = linear.handler().weight
        assert isinstance(weight, Parameter)
        assert weight.dtype == dtype.int8
        assert weight.value().dtype == dtype.int8


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = Linear(5, 6, weight_init="ones")

    def construct(self, x):
        return self.linear(x)


@pytest.mark.env_onecard
def test_woq_predict():
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply RoundToNearestPTQ on SimpleNet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    context.set_context(device_target="Ascend")
    network = SimpleNet()
    ptq = RTN()
    ptq.set_weight_only_quant(True)
    quant_network = ptq.apply(network)
    assert not check_network_contain_layer(quant_network, Linear, (LinearQuant,))
    assert check_network_contain_layer(quant_network, LinearQuant)
    ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
    for _, cell in ascend_network.name_cells().items():
        if not isinstance(cell, LinearQuant):
            continue
        linear: LinearQuant = cell
        assert not linear.input_quantizer()
        assert not linear.output_quantizer()
        assert isinstance(linear.weight_quantizer(), AntiQuantCell)
        weight: Parameter = linear.handler().weight
        assert isinstance(weight, Parameter)
        assert weight.dtype == dtype.int8
        assert weight.value().dtype == dtype.int8
    inputx = Tensor(np.ones((5, 5), dtype=np.float32), dtype=dtype.float32)
    ascend_network(inputx)
