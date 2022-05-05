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
"""DefaultTrasforms."""
from collections import OrderedDict
from mindspore.rewrite import Replacement, PatternNode, Node
from mindspore.nn import BatchNorm2d, Conv2dBnAct, DenseBnAct
from mindspore import nn

act_list = [nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.LeakyReLU, nn.HSigmoid, nn.HSwish]


class Conv2dBnActFuse(Replacement):
    """
    Derived class of Replacement. Define the fuse function of conv2d+bn+act.
    """

    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        """
        Derived from Replacement. Define how to fuse dense+bn+act.
        """
        act_pattern = None
        bn_pattern = None
        if pattern.type() in act_list:
            act_pattern = pattern
            bn_pattern = act_pattern.get_inputs()[0]
        elif pattern.type() == BatchNorm2d:
            bn_pattern = pattern
        conv_pattern = bn_pattern.get_inputs()[0]
        bn_node: Node = matched.get(bn_pattern.name())
        conv_node: Node = matched.get(conv_pattern.name())
        conv2dbnact_cell = Conv2dBnAct(conv_node.get_attribute("in_channels"),
                                       conv_node.get_attribute("out_channels"),
                                       conv_node.get_attribute("kernel_size"),
                                       conv_node.get_attribute("stride"),
                                       conv_node.get_attribute("pad_mode"),
                                       conv_node.get_attribute("padding"),
                                       conv_node.get_attribute("dilation"),
                                       conv_node.get_attribute("group"),
                                       conv_node.get_attribute("has_bias"),
                                       conv_node.get_attribute("weight_init"),
                                       conv_node.get_attribute("bias_init"),
                                       True,
                                       bn_node.get_attribute("momentum"),
                                       bn_node.get_attribute("eps"),
                                       "relu" if act_pattern is not None else None
                                       )
        conv2d_bn_act_node = Node.create_call_cell(conv2dbnact_cell, conv_node.get_targets(), conv_node.get_args(),
                                                   conv_node.get_kwargs(), conv_node.get_name())
        return [conv2d_bn_act_node]


class DenseBnActFuse(Replacement):
    """
    Derived class of Replacement. Define the fuse function of dense+bn+act.
    """

    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        """
        Derived from Replacement. Define how to fuse dense+bn+act.
        """
        act_pattern = None
        bn_pattern = None
        if pattern.type() in act_list:
            act_pattern = pattern
            bn_pattern = act_pattern.get_inputs()[0]
        elif pattern.type() == BatchNorm2d:
            bn_pattern = pattern
        dense_pattern = bn_pattern.get_inputs()[0]
        bn_node: Node = matched.get(bn_pattern.name())
        dense_node: Node = matched.get(dense_pattern.name())
        dense_bn_act_cell = DenseBnAct(
            dense_node.get_attribute("in_channels"),
            dense_node.get_attribute("out_channels"),
            dense_node.get_attribute("weight_init"),
            dense_node.get_attribute("bias_init"),
            dense_node.get_attribute("has_bias"),
            True,
            bn_node.get_attribute("momentum"),
            bn_node.get_attribute("eps"),
            "relu" if act_pattern is not None else None
        )
        dense_bn_act_node = Node.create_call_cell(dense_bn_act_cell, dense_node.get_targets(), dense_node.get_args(),
                                                  dense_node.get_kwargs(), dense_node.get_name())
        return [dense_bn_act_node]
