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
from mindspore import nn
from .combined import Conv2dBn


class Conv2dBnFuse(Replacement):
    """
    Derived class of Replacement. Define how to build a replacement from a Conv2d-BatchNorm pattern match.
    """

    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        """
        Derived from Replacement. Define how to fuse conv+bn.
        """

        bn_pattern = None
        conv_pattern = None

        cur_pattern = pattern
        while cur_pattern:
            if cur_pattern.type() == nn.BatchNorm2d:
                if bn_pattern:
                    raise RuntimeError("Error match, multi-bn!")
                bn_pattern = cur_pattern
            if cur_pattern.type() == nn.Conv2d:
                if conv_pattern:
                    raise RuntimeError("Error match, multi-conv!")
                conv_pattern = cur_pattern
            cur_pattern = cur_pattern.get_inputs()[0] if cur_pattern.get_inputs() else None
        if conv_pattern is None:
            raise RuntimeError("Error match, no-conv!")

        bn_node: Node = matched.get(bn_pattern.name()) if bn_pattern else None
        conv_node: Node = matched.get(conv_pattern.name())
        kwargs = {'in_channels': conv_node.get_attribute("in_channels"),
                  'out_channels': conv_node.get_attribute("out_channels"),
                  'kernel_size': conv_node.get_attribute("kernel_size"),
                  'stride': conv_node.get_attribute("stride"),
                  'pad_mode': conv_node.get_attribute("pad_mode"),
                  'padding': conv_node.get_attribute("padding"),
                  'dilation': conv_node.get_attribute("dilation"),
                  'group': conv_node.get_attribute("group"),
                  'has_bias': conv_node.get_attribute("has_bias"),
                  }
        if hasattr(conv_node, "weight_init"):
            kwargs["weight_init"] = conv_node.get_attribute("weight_init")
        if hasattr(conv_node, "bias_init"):
            kwargs["bias_init"] = conv_node.get_attribute("bias_init")
        if bn_node:
            kwargs['has_bn'] = True
            kwargs['eps'] = bn_node.get_attribute("eps")
            kwargs['momentum'] = bn_node.get_attribute("momentum")
        conv2d_bn_node = Node.create_call_cell(Conv2dBn(**kwargs), conv_node.get_targets(),
                                               conv_node.get_args(), conv_node.get_kwargs(), "Conv2dBn")
        return [conv2d_bn_node]


class Conv2dBnActFuse(Replacement):
    """
    Derived class of Replacement. Define how to build a replacement from a Conv2d-BatchNorm-Activation pattern match.
    """

    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        """
        Derived from Replacement. Define how to fuse conv+bn+act.
        """

        act_pattern = None
        bn_pattern = None
        conv_pattern = None

        cur_pattern = pattern
        while cur_pattern:
            if cur_pattern.type() == nn.ReLU:
                if act_pattern:
                    raise RuntimeError("Error match, multi-act!")
                act_pattern = cur_pattern
            if cur_pattern.type() == nn.BatchNorm2d:
                if bn_pattern:
                    raise RuntimeError("Error match, multi-bn!")
                bn_pattern = cur_pattern
            if cur_pattern.type() == nn.Conv2d:
                if conv_pattern:
                    raise RuntimeError("Error match, multi-conv!")
                conv_pattern = cur_pattern
            cur_pattern = cur_pattern.get_inputs()[0] if cur_pattern.get_inputs() else None
        if conv_pattern is None:
            raise RuntimeError("Error match, no-conv!")

        bn_node: Node = matched.get(bn_pattern.name()) if bn_pattern else None
        conv_node: Node = matched.get(conv_pattern.name())
        kwargs = {'in_channels': conv_node.get_attribute("in_channels"),
                  'out_channels': conv_node.get_attribute("out_channels"),
                  'kernel_size': conv_node.get_attribute("kernel_size"),
                  'stride': conv_node.get_attribute("stride"),
                  'pad_mode': conv_node.get_attribute("pad_mode"),
                  'padding': conv_node.get_attribute("padding"),
                  'dilation': conv_node.get_attribute("dilation"),
                  'group': conv_node.get_attribute("group"),
                  'has_bias': conv_node.get_attribute("has_bias"),
                  }
        if hasattr(conv_node, "weight_init"):
            kwargs["weight_init"] = conv_node.get_attribute("weight_init")
        if hasattr(conv_node, "bias_init"):
            kwargs["bias_init"] = conv_node.get_attribute("bias_init")
        if bn_node:
            kwargs['eps'] = bn_node.get_attribute("eps")
            kwargs['momentum'] = bn_node.get_attribute("momentum")
        if act_pattern:
            kwargs['activation'] = "relu"
        conv2d_bn_act_node = Node.create_call_cell(nn.Conv2dBnAct(**kwargs), conv_node.get_targets(),
                                                   conv_node.get_args(), conv_node.get_kwargs(), "Conv2dBnAct")
        return [conv2d_bn_act_node]


class DenseActFuse(Replacement):
    """
    Derived class of Replacement. Define how to build a replacement from a Conv2d-Activation pattern match.
    """

    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        """
        Derived from Replacement. Define how to fuse dense+act.
        """
        act_pattern = None
        dense_pattern = None

        cur_pattern = pattern
        while cur_pattern:
            if cur_pattern.type() == nn.ReLU:
                if act_pattern:
                    raise RuntimeError("Error match, multi-act!")
                act_pattern = cur_pattern
            if cur_pattern.type() == nn.Dense:
                if dense_pattern:
                    raise RuntimeError("Error match, multi-dense!")
                dense_pattern = cur_pattern
            cur_pattern = cur_pattern.get_inputs()[0] if cur_pattern.get_inputs() else None
        if dense_pattern is None:
            raise RuntimeError("Error match, no-dense!")

        dense_node: Node = matched.get(dense_pattern.name())
        kwargs = {'in_channels': dense_node.get_attribute("in_channels"),
                  'out_channels': dense_node.get_attribute("out_channels"),
                  'has_bias': dense_node.get_attribute("has_bias"),
                  }
        if hasattr(dense_node, "weight_init"):
            kwargs["weight_init"] = dense_node.get_attribute("weight_init")
        if hasattr(dense_node, "bias_init"):
            kwargs["bias_init"] = dense_node.get_attribute("bias_init")
        if act_pattern:
            kwargs['activation'] = "relu"
        dense_act_node = Node.create_call_cell(nn.Dense(**kwargs), dense_node.get_targets(), dense_node.get_args(),
                                               dense_node.get_kwargs(), "DenseAct")
        return [dense_act_node]


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
        if pattern.type() == nn.ReLU:
            act_pattern = pattern
            bn_pattern = act_pattern.get_inputs()[0]
        elif pattern.type() == nn.BatchNorm2d:
            bn_pattern = pattern
        dense_pattern = bn_pattern.get_inputs()[0]
        bn_node: Node = matched.get(bn_pattern.name())
        dense_node: Node = matched.get(dense_pattern.name())
        dense_bn_act_cell = nn.DenseBnAct(
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
                                                  dense_node.get_kwargs(), "DenseBnAct")
        return [dense_bn_act_node]
