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
from golden_stick.quantization.default_qat import DefaultQuantAwareTraining
from golden_stick.quantization.quantize_wrapper_cell import QuantizeWrapperCell
from mindspore.nn import Conv2dBnAct
from mindspore import nn
from mindspore.common.initializer import Normal


class LeNet5(nn.Cell):
    """define LeNet5"""
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        """net structure"""
        x = self.conv1(x)
        x = self.bn(x)
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


def test_lenet():
    """
    Feature: DefaultQuantAwareTraining algorithm.
    Description: Apply DefaultQuantAwareTraining on lenet.
    Expectation: Apply success.
    """

    network = LeNet5(10)
    qat = DefaultQuantAwareTraining()
    new_network = qat.apply(network)
    cells: OrderedDict = new_network.name_cells()
    assert cells.get("conv1_1", None) is not None
    assert isinstance(cells.get("conv1_1"), Conv2dBnAct)

    assert cells.get("conv1_2", None) is not None
    assert isinstance(cells.get("conv1_2"), QuantizeWrapperCell)

    assert cells.get("conv2_1", None) is not None
    assert isinstance(cells.get("conv2_1"), QuantizeWrapperCell)

    assert cells.get("fc1_1", None) is not None
    assert isinstance(cells.get("fc1_1"), QuantizeWrapperCell)
