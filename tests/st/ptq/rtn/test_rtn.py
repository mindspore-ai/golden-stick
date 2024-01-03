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
from collections import OrderedDict
import pytest
from mindspore import nn
from mindspore.common.dtype import QuantDtype
from mindspore.ops.operations import _quant_ops as Q
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.ptq import RTNConfig
from mindspore_gs.ptq.linear import Linear
from mindspore_gs.ptq.quant_cells import LinearQuant
from mindspore_gs.ptq.fake_quantizer import MinMaxPerLayer, MinMaxPerChannel, MinMaxHolder

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.test_utils import qat_config_compare


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
        self.linear = Linear(5, 6, weight_init="ones")

    def construct(self, x):
        return self.linear(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply():
    """
    Feature: RoundToNearestPTQ algorithm set functions.
    Description: Apply RoundToNearestPTQ on SimpleNet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = SimpleNet()
    ptq = RTN()
    new_network = ptq.apply(network)
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

    act_fake_quant = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, MinMaxPerLayer)
    assert act_fake_quant.symmetric()
    assert act_fake_quant.quant_dtype() == QuantDtype.INT8
    assert not act_fake_quant.is_per_channel()
    assert not act_fake_quant.narrow_range()
    assert act_fake_quant.num_bits() == 8

    assert quant_cell.output_quantizer() is None


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert():
    """
    Feature: simulated quantization convert function.
    Description: convert a compressed network to a standard network before exporting to MindIR.
    Expectation: convert success and structure of network as expect.
    """
    network = SimpleNet()
    ptq = RTN()
    new_network = ptq.apply(network)
    new_network = ptq.convert(new_network)
    cells: OrderedDict = new_network.name_cells()

    quant_cell = cells.get("linear", None)
    assert isinstance(quant_cell, LinearQuant)
    weight_fake_quant = quant_cell.weight_quantizer()
    assert isinstance(weight_fake_quant, MinMaxHolder)
    # pylint: disable=W0212
    assert isinstance(weight_fake_quant._fq, Q.FakeQuantPerChannel)

    act_fake_quant = quant_cell.input_quantizer()
    assert isinstance(act_fake_quant, MinMaxHolder)
    # pylint: disable=W0212
    assert isinstance(act_fake_quant._fq, Q.FakeQuantPerLayer)

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
    min_data = quant_params.get("min")
    max_data = quant_params.get("max")
    assert len(min_data) == 6
    assert len(max_data) == 6
    for min_ in min_data:
        assert min_ == 1.
    for max_ in max_data:
        assert max_ == 1.
