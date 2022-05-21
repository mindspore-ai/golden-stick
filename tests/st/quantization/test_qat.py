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
"""test qat."""
from collections import OrderedDict
import pytest
from golden_stick.quantization.default_qat import DefaultQuantAwareTraining
from golden_stick.quantization.default_qat.default_fake_quantizer import DefaultFakeQuantizerPerLayer, \
    DefaultFakeQuantizerPerChannel
from golden_stick.quantization.quantize_wrapper_cell import QuantizeWrapperCell
from mindspore import nn


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
def test_set_config():
    """
    Feature: DefaultQuantAwareTraining algorithm set functions.
    Description: Apply DefaultQuantAwareTraining on lenet.
    Expectation: Apply success and coordinate attributes are same as config.
    """

    network = NetToQuant()
    qat = DefaultQuantAwareTraining()
    qat.set_act_quant_delay(900)
    qat.set_weight_quant_delay(900)
    qat.set_act_per_channel(False)
    qat.set_weight_per_channel(True)
    qat.set_act_narrow_range(False)
    qat.set_weight_narrow_range(False)
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()

    assert cells.get("Conv2dQuant", None) is not None
    conv_quant: QuantizeWrapperCell = cells.get("Conv2dQuant")
    assert isinstance(conv_quant, QuantizeWrapperCell)
    conv_handler = conv_quant._handler
    weight_fake_quant: DefaultFakeQuantizerPerChannel = conv_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, DefaultFakeQuantizerPerChannel)
    assert weight_fake_quant._symmetric
    assert weight_fake_quant._quant_delay == 900
    act_fake_quant = conv_quant._output_quantizer
    assert isinstance(act_fake_quant, DefaultFakeQuantizerPerLayer)
    assert not act_fake_quant._symmetric
    assert act_fake_quant._quant_delay == 900


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_config_enable_fusion():
    """
    Feature: set_enable_fusion api of DefaultQuantAwareTraining.
    Description: Check default value of enable_fusion and value after called set_enable_fusion.
    Expectation: Config success.
    """
    qat = DefaultQuantAwareTraining()
    assert not qat._config.enable_fusion
    qat.set_enable_fusion(True)
    assert qat._config.enable_fusion
