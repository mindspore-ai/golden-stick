# Copyright 2023 Huawei Technologies Co., Ltd
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

import pytest
import mindspore
from mindspore import context, Parameter, dtype, GRAPH_MODE, PYNATIVE_MODE
from mindspore_gs import Backend
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.ptq.quant_cells import LinearQuant
from mindspore_gs.ptq.convert_utils import AntiQuantCell
from mindformers.modules import Linear

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.models.llama2 import llama2, create_dummy_inputs
from tests.st.test_utils import check_network_contain_layer, relative_tolerance_acceptable


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend", "CPU"])
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_llama2_woq_apply_convert(device, mode):
    """
    Feature: RoundToNearestPTQ A16W8 algorithm.
    Description: Apply A16W8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.

    Disabled because of miss of RMSNorm ops in mindspore2.3.
    """

    context.set_context(device_target=device, mode=mode)
    network = llama2(8, 512, 1024, 2)
    assert check_network_contain_layer(network, Linear)
    ptq = RTN()
    ptq.set_weight_only_quant(True)
    quant_network = ptq.apply(network.model)
    quant_network = ptq.calibrate(quant_network)
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


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_llama2_woq_predict_1stage(device, mode):
    """
    Feature: RoundToNearestPTQ A16W8 algorithm.
    Description: Apply A16W8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.

    Disabled because of miss of RMSNorm ops in mindspore2.3.
    """

    context.set_context(device_target=device, mode=mode)
    inputs = create_dummy_inputs(8, 512, 512)
    network = llama2(8, 512, 2048, 2)
    fp_outputs = network(*inputs)

    ptq = RTN()
    ptq.set_weight_only_quant(True)
    quant_network = ptq.apply(network.model)
    quant_network = ptq.calibrate(quant_network)
    ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
    network.model = ascend_network
    quant_outputs = network(*inputs)

    assert len(fp_outputs) == 3
    assert fp_outputs[0].shape == (8, 32000)
    assert fp_outputs[0].dtype == dtype.float32
    assert fp_outputs[1].shape == (8, 512)
    assert fp_outputs[1].dtype == dtype.int32
    assert fp_outputs[2].shape == (8, 512)
    assert fp_outputs[2].dtype == dtype.float32

    assert len(quant_outputs) == 3
    assert quant_outputs[0].shape == (8, 32000)
    assert quant_outputs[0].dtype == dtype.float32
    assert quant_outputs[1].shape == (8, 512)
    assert quant_outputs[1].dtype == dtype.int32
    assert quant_outputs[2].shape == (8, 512)
    assert quant_outputs[2].dtype == dtype.float32

    context.set_context(device_target="CPU", mode=mode)
    assert relative_tolerance_acceptable(quant_outputs[0].asnumpy(), fp_outputs[0].asnumpy(), 5e-2)
    assert relative_tolerance_acceptable(quant_outputs[1].asnumpy(), fp_outputs[1].asnumpy(), 5e-2)
    assert relative_tolerance_acceptable(quant_outputs[2].asnumpy(), fp_outputs[2].asnumpy(), 5e-2)


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_llama2_woq_predict_2stage(device, mode):
    """
    Feature: RoundToNearestPTQ A16W8 algorithm.
    Description: Apply A16W8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.

    Disabled because of miss of RMSNorm ops in mindspore2.3.
    """

    context.set_context(device_target=device, mode=mode)

    def quant(inputs):
        network = llama2(8, 512, 2048, 2)
        fp_outputs = network(*inputs)

        ptq = RTN()
        ptq.set_weight_only_quant(True)
        quant_network = ptq.apply(network.model)
        quant_network = ptq.calibrate(quant_network)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        network.model = ascend_network
        mindspore.save_checkpoint(network, "test_llama2_woq_predict_2stage.ckpt")
        return fp_outputs

    def infer(inputs):
        network = llama2(8, 512, 2048, 2)
        ptq = RTN()
        ptq.set_weight_only_quant(True)
        quant_network = ptq.apply(network.model)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        network.model = ascend_network
        mindspore.load_checkpoint("test_llama2_woq_predict_2stage.ckpt", network)
        return ascend_network(*inputs)

    inputs = create_dummy_inputs(8, 512, 512)
    fp_outputs = quant(inputs)
    quant_outputs = infer(inputs)
    assert len(fp_outputs) == 3
    assert fp_outputs[0].shape == (8, 32000)
    assert fp_outputs[0].dtype == dtype.float32
    assert fp_outputs[1].shape == (8, 512)
    assert fp_outputs[1].dtype == dtype.int32
    assert fp_outputs[2].shape == (8, 512)
    assert fp_outputs[2].dtype == dtype.float32

    assert len(quant_outputs) == 3
    assert quant_outputs[0].shape == (8, 32000)
    assert quant_outputs[0].dtype == dtype.float32
    assert quant_outputs[1].shape == (8, 512)
    assert quant_outputs[1].dtype == dtype.int32
    assert quant_outputs[2].shape == (8, 512)
    assert quant_outputs[2].dtype == dtype.float32

    context.set_context(device_target="CPU", mode=mode)
    assert relative_tolerance_acceptable(quant_outputs[0].asnumpy(), fp_outputs[0].asnumpy(), 9e-2)
    assert relative_tolerance_acceptable(quant_outputs[1].asnumpy(), fp_outputs[1].asnumpy(), 5e-2)
    assert relative_tolerance_acceptable(quant_outputs[2].asnumpy(), fp_outputs[2].asnumpy(), 5e-2)
