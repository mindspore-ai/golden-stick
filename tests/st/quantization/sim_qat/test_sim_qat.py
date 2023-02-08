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
"""test interfaces of sim_qat."""
import os
import sys
from collections import OrderedDict
import pytest
from mindspore import nn
from mindspore.common.dtype import QuantDtype
from mindspore_gs.quantization.simulated_quantization import SimulatedQuantizationAwareTraining as SimQAT
from mindspore_gs.quantization.simulated_quantization.simulated_fake_quantizers import SimulatedFakeQuantizerPerLayer, \
    SimulatedFakeQuantizerPerChannel
from mindspore_gs.quantization.quantize_wrapper_cell import QuantizeWrapperCell
from mindspore_gs.quantization.simulated_quantization.simulated_quantization_config import SimulatedQuantizationConfig

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))
# pylint: disable=wrong-import-position
from tests.st.test_utils import qat_config_compare


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor():
    """
    Feature: SimQAT algorithm.
    Description: Call constructor of SimulatedQuantizationAwareTraining and check config.
    Expectation: SimulatedQuantizationConfig is updated according to argument `config` of constructor.
    """

    config = {"quant_delay": (100, 200), "quant_dtype": (QuantDtype.INT8, QuantDtype.INT8),
              "per_channel": (False, True), "symmetric": (True, False), "narrow_range": (True, False),
              "enable_fusion": True, "freeze_bn": 100, "bn_fold": True, "one_conv_fold": False}
    qat = SimQAT(config)
    quant_config: SimulatedQuantizationConfig = qat._config
    assert qat_config_compare(quant_config, config)

    config = {"quant_delay": [100, 200], "quant_dtype": [QuantDtype.INT8, QuantDtype.INT8],
              "per_channel": [False, True], "symmetric": [True, False], "narrow_range": [True, False],
              "enable_fusion": True, "freeze_bn": 100, "bn_fold": True, "one_conv_fold": False}
    qat = SimQAT(config)
    quant_config: SimulatedQuantizationConfig = qat._config
    assert qat_config_compare(quant_config, config)

    config = {"quant_delay": 100, "quant_dtype": QuantDtype.INT8, "per_channel": False, "symmetric": True,
              "narrow_range": True, "enable_fusion": True, "freeze_bn": 100, "bn_fold": True, "one_conv_fold": False}
    qat = SimQAT(config)
    quant_config: SimulatedQuantizationConfig = qat._config
    assert qat_config_compare(quant_config, config)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor_error():
    """
    Feature: SimQAT algorithm.
    Description: Feed invalid config to constructor of SimulatedQuantizationAwareTraining and except error.
    Expectation: Except error.
    """

    config = {"quant_delay": (True, True)}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the type of 'act_quant_delay' "
                                        "must be int, but got 'bool'"):
        SimQAT(config)

    config = {"quant_delay": (1, True)}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the type of 'weight_quant_delay' "
                                        "must be int, but got 'bool'"):
        SimQAT(config)

    config = {"per_channel": (1, 1)}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'act_per_channel' must be a "
                                        "bool, but got int."):
        SimQAT(config)

    config = {"per_channel": (False, 1)}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'weight_per_channel' must be a "
                                        "bool, but got int."):
        SimQAT(config)

    config = {"symmetric": (1, 1)}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'act_symmetric' must be a bool, "
                                        "but got int."):
        SimQAT(config)

    config = {"symmetric": (True, 1)}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'weight_symmetric' must be a "
                                        "bool, but got int."):
        SimQAT(config)

    config = {"narrow_range": (1, 1)}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'act_narrow_range' must be a "
                                        "bool, but got int."):
        SimQAT(config)

    config = {"bn_fold": 1}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'bn_fold' must be a bool, but "
                                        "got int."):
        SimQAT(config)

    config = {"one_conv_fold": 1}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'one_conv_fold' must be a bool, "
                                        "but got int."):
        SimQAT(config)

    config = {"quant_dtype": [1, 1]}
    with pytest.raises(TypeError, match="The parameter `act quant dtype` must be isinstance of QuantDtype, but got 1."):
        SimQAT(config)

    config = {"quant_dtype": [QuantDtype.INT8, 1]}
    with pytest.raises(TypeError, match="The parameter `weight quant dtype` must be isinstance of QuantDtype, but got "
                                        "1."):
        SimQAT(config)

    config = {"freeze_bn": True}
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the type of 'freeze_bn' must be "
                                        "int, but got 'bool'"):
        SimQAT(config)

    config = {"freeze_bn": -1}
    with pytest.raises(ValueError, match="For 'SimulatedQuantizationAwareTraining', the 'freeze_bn' must be int and "
                                         "must >= 0, but got '-1' with type 'int'."):
        SimQAT(config)

    config = {"quant_delay": [1, 1, 1]}
    with pytest.raises(ValueError, match="input `quant delay` len should be less than 3"):
        SimQAT(config)

    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.INT8, QuantDtype.INT8]}
    with pytest.raises(ValueError, match="input `quant dtype` len should be less than 3"):
        SimQAT(config)

    config = {"per_channel": [False, False, False]}
    with pytest.raises(ValueError, match="input `per channel` len should be less than 3"):
        SimQAT(config)

    config = {"symmetric": [False, False, False]}
    with pytest.raises(ValueError, match="input `symmetric` len should be less than 3"):
        SimQAT(config)

    config = {"narrow_range": [False, False, False]}
    with pytest.raises(ValueError, match="input `narrow range` len should be less than 3"):
        SimQAT(config)

    config = {"quant_delay": [-1, -1]}
    with pytest.raises(ValueError, match="For 'SimulatedQuantizationAwareTraining', the 'act_quant_delay' must be int "
                                         "and must >= 0, but got '-1' with type 'int'."):
        SimQAT(config)

    config = {"quant_delay": [1, -1]}
    with pytest.raises(ValueError, match="For 'SimulatedQuantizationAwareTraining', the 'weight_quant_delay' must be "
                                         "int and must >= 0, but got '-1' with type 'int'."):
        SimQAT(config)

    config = {"per_channel": [True, True]}
    with pytest.raises(ValueError, match="Only supported if `act_per_channel` is False yet."):
        SimQAT(config)

    config = {"quant_dtype": [QuantDtype.UINT8, QuantDtype.UINT8]}
    with pytest.raises(ValueError, match="Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet."):
        SimQAT(config)

    config = {"quant_dtype": [QuantDtype.INT8, QuantDtype.UINT8]}
    with pytest.raises(ValueError, match="Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet."):
        SimQAT(config)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_bn_fold():
    """
    Feature: set_bn_fold api of SimQAT.
    Description: Check default value of bn_fold and value after called set_bn_fold.
    Expectation: Config success.
    """

    qat = SimQAT()
    config: SimulatedQuantizationConfig = qat._config
    assert not config.bn_fold
    qat.set_bn_fold(True)
    assert config.bn_fold


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_set_bn_fold_type_error():
    """
    Feature: set_bn_fold api of SimQAT.
    Description: Feed invalid type of bn_fold to set_bn_fold function.
    Expectation: Except TypeError.
    """

    qat = SimQAT()
    with pytest.raises(TypeError, match="For 'SimulatedQuantizationAwareTraining', the 'bn_fold' must be a bool, but "
                                        "got int."):
        qat.set_bn_fold(1)


class NetToQuant(nn.Cell):
    """
    Network with single conv2d to be quanted
    """

    def __init__(self):
        super(NetToQuant, self).__init__()
        self.conv = nn.Conv2d(5, 6, 5, pad_mode='valid')

    def construct(self, x):
        x = self.conv(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_apply():
    """
    Feature: SimQAT algorithm set functions.
    Description: Apply DefaultQuantAwareTraining on lenet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = NetToQuant()
    qat = SimQAT()
    qat.set_act_quant_delay(900)
    qat.set_weight_quant_delay(900)
    qat.set_act_per_channel(False)
    qat.set_weight_per_channel(True)
    qat.set_act_narrow_range(False)
    qat.set_weight_narrow_range(False)
    qat.set_one_conv_fold(True)
    qat.set_bn_fold(False)
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()

    assert cells.get("Conv2dQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: SimulatedFakeQuantizerPerChannel = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SimulatedFakeQuantizerPerChannel)
    assert weight_fake_quant._symmetric
    assert weight_fake_quant._quant_delay == 900
    act_fake_quant = conv_quant._output_quantizer
    assert isinstance(act_fake_quant, SimulatedFakeQuantizerPerLayer)
    assert not act_fake_quant._symmetric
    assert act_fake_quant._quant_delay == 900


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert():
    """
    Feature: simulated quantization convert function.
    Description: convert a compressed network to a standard network before exporting to MindIR.
    Expectation: convert success and structure of network as expect.
    """
    network = NetToQuant()
    qat = SimQAT()
    new_network = qat.apply(network)
    new_network = qat.convert(new_network)

    cells: OrderedDict = new_network.name_cells()
    assert cells.get("Conv2dQuant", None) is not None
    conv: QuantizeWrapperCell = cells.get("Conv2dQuant")
    assert isinstance(conv, QuantizeWrapperCell)
    act_fake_quant = conv._output_quantizer
    assert not isinstance(act_fake_quant, SimulatedFakeQuantizerPerLayer)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_error():
    """
    Feature: simulated quantization convert function.
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: Except TypeError.
    """
    network = NetToQuant()
    qat = SimQAT()
    new_network = qat.apply(network)
    with pytest.raises(TypeError, match="The parameter `net_opt` must be isinstance of Cell"):
        qat.convert(100)

    with pytest.raises(TypeError, match="The parameter `ckpt_path` must be isinstance of str"):
        qat.convert(new_network, 100)

    with pytest.raises(ValueError, match="The parameter `ckpt_path` can only be empty or a valid file"):
        qat.convert(new_network, "file_path")
