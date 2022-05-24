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
"""test lenet."""

from collections import OrderedDict
import pytest
from golden_stick.quantization.simulated_quantization import SimulatedQuantizationAwareTraining
from golden_stick.quantization.simulated_quantization.simulated_fake_quantizers import SimulatedFakeQuantizerPerLayer, \
    SimulatedFakeQuantizerPerChannel
from golden_stick.quantization.quantize_wrapper_cell import QuantizeWrapperCell
from mindspore import nn
from mindspore.common.initializer import Normal


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet5(num_class=10)

    """

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_lenet():
    """
    Feature: DefaultQuantAwareTraining algorithm.
    Description: Apply DefaultQuantAwareTraining on lenet.
    Expectation: Apply success.
    """

    network = LeNet5(10)
    qat = SimulatedQuantizationAwareTraining({"per_channel": [False, True], "symmetric": [False, True],
                                              "quant_delay": [900, 900]})
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

    assert cells.get("DenseQuant", None) is not None
    dense_quant: QuantizeWrapperCell = cells.get("DenseQuant")
    assert isinstance(dense_quant, QuantizeWrapperCell)
    dense_handler = dense_quant._handler
    weight_fake_quant: SimulatedFakeQuantizerPerChannel = dense_handler.fake_quant_weight
    assert isinstance(weight_fake_quant, SimulatedFakeQuantizerPerChannel)
    assert weight_fake_quant._symmetric
    assert weight_fake_quant._quant_delay == 900
    act_fake_quant = dense_quant._output_quantizer
    assert isinstance(act_fake_quant, SimulatedFakeQuantizerPerLayer)
    assert not act_fake_quant._symmetric
    assert act_fake_quant._quant_delay == 900
