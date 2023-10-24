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
"""test unfold node func in transformer."""

import pytest

from mindspore import nn
from mindspore.ops import operations as P
from mindspore_gs.net_transform import NetTransformer


def _conv3x3(in_channel, out_channel, stride=1):
    """_conv3x3"""
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init="ones")


def _conv1x1(in_channel, out_channel, stride=1):
    """_conv1x1"""
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init="ones")


def _bn(channel):
    """_bn"""
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


class ResidualBlock(nn.Cell):
    """ResidualBlock"""
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        """init"""
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(channel, out_channel, stride=1)
        self.bn1 = _bn(out_channel)
        self.relu = nn.ReLU()
        self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), _bn(out_channel)])

    def construct(self, x):
        """construct"""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        identity = self.down_sample_layer(identity)
        out = out + identity
        out = self.relu(out)

        return out


class ResNetSimple(nn.Cell):
    """ResNetSimple"""
    def __init__(self):
        """init"""
        super(ResNetSimple, self).__init__(auto_prefix=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode='pad', weight_init="ones")
        self.bn1 = _bn(16)
        self.relu = P.ReLU()
        self.layer1 = self._make_layer(ResidualBlock, 2, in_channel=63, out_channel=256, stride=1)
        self.layer1.append(self.conv1)
        self.layer1.append(self.bn1)
        self.reshape = P.Reshape()
        self.out_channels = 10

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        return x

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """make_layer"""
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)
        for _ in range(1, layer_num):
            resnet_block = ResidualBlock(out_channel, out_channel, stride=1)
            layers.append(resnet_block)
        return nn.SequentialCell(layers)


@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unfold_nodes():
    """
    Feature: parse CellContainer Node and return unfolded nodes.
    Description: parse a network with SequentialCell object, and return unfolded nodes.
    Expectation: Net transformer can unfold nodes successfully.

    Unfolded nodes are support to be as follows:
    # 0  NodeType.Input
    # 1  conv
    # 2  bn
    # 3  relu
    # 4  NodeType.Input
    # 5  NodeType.CallMethod
    # 6  conv
    # 7  bn
    # 8  conv
    # 9  bn
    # 10 NodeType.MathOps
    # 11 relu
    # 12 NodeType.Output
    # 13 NodeType.Input
    # 14 NodeType.CallMethod
    # 15 conv
    # 16 bn
    # 17 conv
    # 18 bn
    # 19 NodeType.MathOps
    # 20 relu
    # 21 NodeType.Output
    # 22 conv
    # 23 bn
    # 24 NodeType.Output
    """
    net_trans = NetTransformer(ResNetSimple())
    node_count = 0
    conv_count = 0
    bn_count = 0
    relu_count = 0
    for node in net_trans.unfolded_nodes():
        if node_count in [1, 6, 8, 15, 17, 22]:
            assert node.get_handler().get_instance_type() is nn.Conv2d
            conv_count += 1
        elif node_count in [2, 7, 9, 16, 18, 23]:
            assert node.get_handler().get_instance_type() is nn.BatchNorm2d
            bn_count += 1
        elif node_count in [3, 11, 20]:
            assert node.get_handler().get_instance_type() is nn.ReLU or \
                   node.get_handler().get_instance_type() is P.ReLU
            relu_count += 1
        node_count += 1
    assert node_count == 25
    assert conv_count == 6
    assert bn_count == 6
    assert relu_count == 3
