# Copyright 2024 Huawei Technologies Co., Ltd
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
"""test Linear W8A16 algorithm."""
import os
import sys
from collections import OrderedDict

import pytest
import numpy as np
import mindspore
from mindspore import context, Parameter, dtype, GRAPH_MODE, PYNATIVE_MODE, Tensor, nn, QuantDtype
from mindspore_gs import Backend
from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell, FakeQuantParam
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.ptq.quant_cells import LinearQuant
from mindspore_gs.ptq.convert_utils import AntiquantBMMCell
from mindspore_gs.ptq.fake_quantizer import MinMaxPerChannel
from mindformers.modules import Linear

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.models.llama2 import llama2, create_dummy_inputs
from tests.st.test_utils import check_network_contain_layer, relative_tolerance_acceptable, \
    absolute_tolerance_acceptable


class SimpleNet(nn.Cell):
    """
    Network with single linear to be quant
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = Linear(in_channels=5, out_channels=6, weight_init="ones")

    def construct(self, x):
        return self.linear(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply_convert():
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply RoundToNearestPTQ on SimpleNet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = SimpleNet()
    ptq = RTN()
    ptq.set_linear_w8a16(True)
    # apply & calibrate
    new_network = ptq.apply(network)
    cells: OrderedDict = new_network.name_cells()
    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, LinearQuant)
    weight_fake_quant: MinMaxPerChannel = quant_cell.weight_quantizer()
    assert isinstance(weight_fake_quant, MinMaxPerChannel)
    assert weight_fake_quant.symmetric()
    assert weight_fake_quant.quant_dtype() == QuantDtype.INT8
    assert weight_fake_quant.is_per_channel()
    assert not weight_fake_quant.narrow_range()
    assert weight_fake_quant.num_bits() == 8

    assert quant_cell.input_quantizer() is None
    assert quant_cell.output_quantizer() is None

    quant_params = weight_fake_quant.quant_params()
    min_data = np.array(quant_params.get("min"))
    max_data = np.array(quant_params.get("max"))
    assert min_data.shape == (6, 1)
    assert max_data.shape == (6, 1)
    for min_ in min_data:
        assert min_[0] == 1.
    for max_ in max_data:
        assert max_[0] == 1.

    # convert
    new_network = ptq.convert(new_network)
    cells: OrderedDict = new_network.name_cells()

    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, LinearQuant)
    weight_fake_quant = quant_cell.weight_quantizer()
    assert isinstance(weight_fake_quant, FakeQuantParamCell)
    assert isinstance(weight_fake_quant.fq, FakeQuantParam)

    assert quant_cell.input_quantizer() is None
    assert quant_cell.output_quantizer() is None


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_woq_predict_1stage(device, mode):
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply, Convert and Predict RoundToNearestPTQ on SimpleNet.
    Expectation: Execute success.
    """

    context.set_context(device_target=device, mode=mode)
    network = SimpleNet()
    ptq = RTN()
    ptq.set_linear_w8a16(True)
    quant_network = ptq.apply(network)
    ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
    for _, cell in ascend_network.name_cells().items():
        if not isinstance(cell, LinearQuant):
            continue
        linear: LinearQuant = cell
        assert not linear.input_quantizer()
        assert not linear.output_quantizer()
        assert isinstance(linear.weight_quantizer(), AntiquantBMMCell)
        weight: Parameter = linear.handler().weight
        assert isinstance(weight, Parameter)
        assert weight.dtype == dtype.int8
        assert weight.value().dtype == dtype.int8
    inputx = Tensor(np.ones((5, 5), dtype=np.float32), dtype=dtype.float32)
    output: np.ndarray = ascend_network(inputx).asnumpy()
    assert output.shape == (5, 6)
    for ele in output.flatten():
        assert ele == 5.


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_woq_predict_2stage(device, mode):
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply, Convert and Predict RoundToNearestPTQ on SimpleNet.
    Expectation: Execute success.
    """

    context.set_context(device_target=device, mode=mode)

    def quant():
        network = SimpleNet()
        ptq = RTN()
        ptq.set_linear_w8a16(True)
        quant_network = ptq.apply(network)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        for _, cell in ascend_network.name_cells().items():
            if not isinstance(cell, LinearQuant):
                continue
            linear: LinearQuant = cell
            assert not linear.input_quantizer()
            assert not linear.output_quantizer()
            assert isinstance(linear.weight_quantizer(), AntiquantBMMCell)
            weight: Parameter = linear.handler().weight
            assert isinstance(weight, Parameter)
            assert weight.dtype == dtype.int8
            assert weight.value().dtype == dtype.int8
        mindspore.save_checkpoint(ascend_network, "test_woq_predict_2stage.ckpt")

    def infer():
        inputx = Tensor(np.ones((5, 5), dtype=np.float32), dtype=dtype.float32)
        network = SimpleNet()
        ptq = RTN()
        ptq.set_linear_w8a16(True)
        ptq.set_deploy(True)
        quant_network = ptq.apply(network)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        mindspore.load_checkpoint("test_woq_predict_2stage.ckpt", ascend_network)
        output: np.ndarray = ascend_network(inputx).asnumpy()
        assert output.shape == (5, 6)
        for ele in output.flatten():
            assert ele == 5.

    quant()
    infer()


class LinearsNet(nn.Cell):
    """
    Network with single linear to be quant
    """

    def __init__(self):
        super(LinearsNet, self).__init__()
        self.linear1 = Linear(in_channels=5, out_channels=6)
        self.linear2 = Linear(in_channels=6, out_channels=5)

    def construct(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend"])
@pytest.mark.parametrize("mode", [GRAPH_MODE])
def test_linears_woq_predict_2stage(device, mode):
    """
    Feature: RoundToNearestPTQ A16W8 algorithm.
    Description: Apply A16W8 quant on LLama2 and convert to ascend backend.
    Expectation: Execute successfully.
    """

    context.set_context(device_target=device, mode=mode)
    def quant(inputs):
        network = LinearsNet()
        cur_dir, _ = os.path.split(os.path.abspath(__file__))
        ckpt_path = os.path.join(cur_dir, "../../../data/test_ckpt/test_linears_woq_predict_2stage_fp32.ckpt")
        mindspore.load_checkpoint(ckpt_path, network)
        fp_outputs = network(*inputs)

        ptq = RTN()
        ptq.set_linear_w8a16(True)
        quant_network = ptq.apply(network)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        mindspore.save_checkpoint(ascend_network, "test_linears_woq_predict_2stage.ckpt")
        return fp_outputs

    def infer(inputs):
        network = LinearsNet()
        ptq = RTN()
        ptq.set_linear_w8a16(True)
        ptq.set_deploy(True)
        quant_network = ptq.apply(network)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        mindspore.load_checkpoint("test_linears_woq_predict_2stage.ckpt", ascend_network)
        return ascend_network(*inputs)

    arr = np.array([[-1.02, 2.03, 3.04, -4.54, 5.55], [6.78, 0.02, 0.005, 6.77, 3.22],
                    [-4.44, -5.55, -6.66, -1.11, -2.22], [9.87, 8.45, 3.67, -2.22, 3.21],
                    [0.12, 4.00, -0.94, -3.89, -1.29]], dtype=np.float16)
    inputs = [Tensor(arr, dtype=dtype.float16)]
    fp_output = quant(inputs)
    quant_output = infer(inputs)

    assert fp_output.shape == (5, 5)
    assert fp_output.dtype == dtype.float16
    assert quant_output.shape == (5, 5)
    assert quant_output.dtype == dtype.float16

    context.set_context(device_target="CPU", mode=mode)
    assert relative_tolerance_acceptable(quant_output.asnumpy(), fp_output.asnumpy(), 0.1587)


@pytest.mark.level0
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
    ptq.set_linear_w8a16(True)
    quant_network = ptq.apply(network.model)
    assert not check_network_contain_layer(quant_network, Linear, (LinearQuant,))
    assert check_network_contain_layer(quant_network, LinearQuant)
    ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
    for _, cell in ascend_network.name_cells().items():
        if not isinstance(cell, LinearQuant):
            continue
        linear: LinearQuant = cell
        assert not linear.input_quantizer()
        assert not linear.output_quantizer()
        assert isinstance(linear.weight_quantizer(), AntiquantBMMCell)
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
    ptq.set_linear_w8a16(True)
    quant_network = ptq.apply(network.model)
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
        ptq.set_linear_w8a16(True)
        quant_network = ptq.apply(network.model)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        network.model = ascend_network
        mindspore.save_checkpoint(network, "test_llama2_woq_predict_2stage.ckpt")
        return fp_outputs

    def infer(inputs):
        network = llama2(8, 512, 2048, 2)
        ptq = RTN()
        ptq.set_linear_w8a16(True)
        ptq.set_deploy(True)
        quant_network = ptq.apply(network.model)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        network.model = ascend_network
        mindspore.load_checkpoint("test_llama2_woq_predict_2stage.ckpt", network)
        return network(*inputs)

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
    assert absolute_tolerance_acceptable(quant_outputs[0].asnumpy(), fp_outputs[0].asnumpy(), 4e-2)
    assert relative_tolerance_acceptable(quant_outputs[1].asnumpy(), fp_outputs[1].asnumpy(), 5e-2)
    assert relative_tolerance_acceptable(quant_outputs[2].asnumpy(), fp_outputs[2].asnumpy(), 5e-2)
