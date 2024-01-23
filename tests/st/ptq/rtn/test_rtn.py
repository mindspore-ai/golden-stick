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
from collections import OrderedDict

import pytest
import numpy as np
import mindspore
from mindspore import nn, context, GRAPH_MODE, Parameter, dtype, Tensor
from mindspore.common.dtype import QuantDtype
from mindspore_gs import Backend
from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell, FakeQuantParam
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.ptq import RTNConfig
from mindspore_gs.ptq.quant_cells import LinearQuant
from mindspore_gs.ptq.convert_utils import AntiQuantCell
from mindspore_gs.ptq.fake_quantizer import MinMaxPerLayer, MinMaxPerChannel
from mindformers.modules import Linear

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../../'))
# pylint: disable=wrong-import-position
from tests.st.test_utils import qat_config_compare, relative_tolerance_acceptable


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor():
    """
    Feature: RoundToNearestPTQ algorithm.
    Description: Call constructor of RoundToNearestPTQ and check config.
    Expectation: RTNConfig is updated according to argument `config` of constructor.
    """

    config = {"quant_dtype": (QuantDtype.INT8, QuantDtype.INT8), "per_channel": (False, True),
              "symmetric": (True, True), "narrow_range": (False, False)}
    ptq = RTN(config)
    # pylint: disable=W0212
    quant_config: RTNConfig = ptq._config
    assert qat_config_compare(quant_config, config)

    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.INT8], "per_channel": [False, True],
              "symmetric": [True, True], "narrow_range": [False, False]}
    qat = RTN(config)
    # pylint: disable=W0212
    quant_config: RTNConfig = qat._config
    assert qat_config_compare(quant_config, config)

    config = {"quant_dtype": QuantDtype.INT8, "per_channel": [False, True], "symmetric": True, "narrow_range": False}
    qat = RTN(config)
    # pylint: disable=W0212
    quant_config: RTNConfig = qat._config
    assert qat_config_compare(quant_config, config)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor_error():
    """
    Feature: RoundToNearestPTQ algorithm.
    Description: Feed invalid config to constructor of RoundToNearestPTQ and except error.
    Expectation: Except error.
    """
    config = {"per_channel": (1, 1)}
    with pytest.raises(TypeError):
        RTN(config)

    config = {"per_channel": (False, 1)}
    with pytest.raises(TypeError):
        RTN(config)

    config = {"symmetric": (1, 1)}
    with pytest.raises(TypeError):
        RTN(config)

    config = {"symmetric": (True, 1)}
    with pytest.raises(TypeError):
        RTN(config)

    config = {"narrow_range": (1, 1)}
    with pytest.raises(TypeError):
        RTN(config)

    config = {"quant_dtype": [1, 1]}
    with pytest.raises(TypeError):
        RTN(config)

    config = {"quant_dtype": [QuantDtype.INT8, 1]}
    with pytest.raises(TypeError):
        RTN(config)

    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.INT8, QuantDtype.INT8]}
    with pytest.raises(ValueError):
        RTN(config)

    config = {"per_channel": [False, False, False]}
    with pytest.raises(ValueError):
        RTN(config)

    config = {"symmetric": [False, False, False]}
    with pytest.raises(ValueError):
        RTN(config)

    config = {"narrow_range": [False, False, False]}
    with pytest.raises(ValueError):
        RTN(config)

    config = {"per_channel": [True, True]}
    with pytest.raises(ValueError):
        RTN(config)

    config = {"quant_dtype": [QuantDtype.UINT8, QuantDtype.UINT8]}
    with pytest.raises(ValueError):
        RTN(config)

    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.UINT8]}
    with pytest.raises(ValueError):
        RTN(config)


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
    # apply
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

    act_fake_quant: MinMaxPerLayer = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, MinMaxPerLayer)
    assert act_fake_quant.symmetric()
    assert act_fake_quant.quant_dtype() == QuantDtype.INT8
    assert not act_fake_quant.is_per_channel()
    assert not act_fake_quant.narrow_range()
    assert act_fake_quant.num_bits() == 8

    assert quant_cell.output_quantizer() is None

    # calibrate
    weight_fake_quant.float_min = weight_fake_quant.float_min * -1
    weight_fake_quant.float_max = weight_fake_quant.float_max * -1
    act_fake_quant.float_min = act_fake_quant.float_min * -1
    act_fake_quant.float_max = act_fake_quant.float_max * -1
    # convert
    new_network = ptq.convert(new_network)
    cells: OrderedDict = new_network.name_cells()

    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, LinearQuant)
    weight_fake_quant = quant_cell.weight_quantizer()
    assert isinstance(weight_fake_quant, FakeQuantParamCell)
    assert isinstance(weight_fake_quant.fq, FakeQuantParam)

    act_fake_quant = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, FakeQuantParamCell)
    assert isinstance(act_fake_quant.fq, FakeQuantParam)

    assert quant_cell.output_quantizer() is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_error():
    """
    Feature: simulated quantization convert function.
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: Except TypeError.
    """
    network = SimpleNet()
    ptq = RTN()
    new_network = ptq.apply(network)
    with pytest.raises(TypeError, match="The parameter `net_opt` must be isinstance of Cell"):
        ptq.convert(100)

    with pytest.raises(TypeError, match="The parameter `ckpt_path` must be isinstance of str"):
        ptq.convert(new_network, 100)

    with pytest.raises(ValueError, match="The parameter `ckpt_path` can only be empty or a valid file"):
        ptq.convert(new_network, "file_path")


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_woq_apply():
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply RoundToNearestPTQ on SimpleNet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = SimpleNet()
    ptq = RTN()
    ptq.set_weight_only_quant(True)
    new_network = ptq.apply(network)
    new_network = ptq.calibrate(new_network)
    cells: OrderedDict = new_network.name_cells()

    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, LinearQuant)
    weight_fake_quant = quant_cell.weight_quantizer()
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
    ptq.set_weight_only_quant(True)
    quant_network = ptq.apply(network)
    quant_network = ptq.calibrate(quant_network)
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
        ptq.set_weight_only_quant(True)
        quant_network = ptq.apply(network)
        quant_network = ptq.calibrate(quant_network)
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
        mindspore.save_checkpoint(ascend_network, "test_woq_predict_2stage.ckpt")

    def infer():
        inputx = Tensor(np.ones((5, 5), dtype=np.float32), dtype=dtype.float32)
        network = SimpleNet()
        ptq = RTN()
        ptq.set_weight_only_quant(True)
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
        ptq.set_weight_only_quant(True)
        quant_network = ptq.apply(network)
        quant_network = ptq.calibrate(quant_network)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        mindspore.save_checkpoint(ascend_network, "test_linears_woq_predict_2stage.ckpt")
        return fp_outputs

    def infer(inputs):
        network = LinearsNet()
        ptq = RTN()
        ptq.set_weight_only_quant(True)
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
