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
"""GhostNet model define"""
import math

import mindspore.nn as nn
from mindspore.ops import operations as P
from ..comp_algo import CompAlgo


class MyHSigmoid(nn.Cell):
    """
    Hard Sigmoid definition.
    """

    def __init__(self):
        super(MyHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        """ construct """
        return self.relu6(x + 3.) * 0.16666667


class Activation(nn.Cell):
    """
    Activation definition.
    """

    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = MyHSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.HSwish()
        else:
            raise ValueError(
                "Not implemented.Please set act_func in "
                "['relu', 'relu6', 'hsigmoid', 'hard_sigmoid', 'hswish', 'hard_swish'].")

    def construct(self, x):
        """ construct """
        return self.act(x)


class ConvUnit(nn.Cell):
    """
    ConvUnit warpper definition.
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu', pad_mode='pad'):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              group=num_groups,
                              has_bias=False,
                              pad_mode=pad_mode)
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = Activation(act_type) if use_act else None

    def construct(self, x):
        """ construct of conv unit """
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class GhostModule(nn.Cell):
    """
    GhostModule warpper definition.
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu', prex=None, pad_mode='pad'):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                     num_groups=1, use_act=use_act, act_type=act_type, pad_mode=pad_mode)
        self.primary_conv.conv.weight.name = prex + self.primary_conv.conv.weight.name
        self.primary_conv.bn.gamma.name = prex + self.primary_conv.bn.gamma.name
        self.primary_conv.bn.beta.name = prex + self.primary_conv.bn.beta.name
        self.primary_conv.bn.moving_mean.name = prex + self.primary_conv.bn.moving_mean.name
        self.primary_conv.bn.moving_variance.name = prex + self.primary_conv.bn.moving_variance.name
        self.cheap_operation = ConvUnit(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                        padding=dw_size // 2, num_groups=init_channels,
                                        use_act=use_act, act_type=act_type, pad_mode='pad')
        self.cheap_operation.conv.weight.name = prex + self.cheap_operation.conv.weight.name
        self.cheap_operation.bn.gamma.name = prex + self.cheap_operation.bn.gamma.name
        self.cheap_operation.bn.beta.name = prex + self.cheap_operation.bn.beta.name
        self.cheap_operation.bn.moving_mean.name = prex + self.cheap_operation.bn.moving_mean.name
        self.cheap_operation.bn.moving_variance.name = prex + self.cheap_operation.bn.moving_variance.name
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        """ ghost module construct """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return self.concat((x1, x2))


class GhostAlgo(CompAlgo):
    """
    `GhostAlgo` is a subclass of CompAlgo, which implements the replacement of the original Block with the cheap
     operation GhostModule.

    Args:
        config (Dict): Configuration of `GhostAlgo`. There are no configurable options for
            `GhostAlgo` currently, but for compatibility, the config parameter in the constructor of class A
            is retained.

    Examples:
        >>> from mindspore_gs import GhostAlgo
        >>> from mindspore import nn
        >>> class Net(nn.Cell):
        ...     def __init__(self, num_channel=1):
        ...         super(Net, self).__init__()
        ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.bn1 = nn.BatchNorm2d(6)
        ...
        ...     def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.bn(x)
        ...         return x
        >>> ## 1) Define network to be transformed
        >>> net = Net()
        >>> ## 2) Define Ghost Algorithm
        >>> algo = GhostAlgo()
        >>> ## 3) Apply Ghost-algorithm to origin network
        >>> new_net = algo.apply(net)
        >>> ## 4) Print check the result. Conv2d should be transformed to cheap_operation GhostModule.
        >>> names = []
        >>> for name, module in new_net.cells_and_names():
        ...    names.append(name)
        >>> print('layer.conv1.cheap_operation' in names)
        True
    """

    def _get_pad(self, kernel_size):
        """set the padding number"""
        pad = 0
        if kernel_size == 1:
            pad = 0
        elif kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        elif kernel_size == 7:
            pad = 3
        else:
            raise ValueError("Not implemented.Please set kernel_size in [1, 3, 5, 7].")
        return pad

    def _tranform(self, net):
        """Transform conv."""

        def _inject(modules):
            keys = list(modules.keys())
            for _, k in enumerate(keys):
                for value, param in modules[k].parameters_and_names():
                    prex = param.name.strip(value)
                if isinstance(modules[k], nn.Conv2d):
                    try:
                        kernel_size_0 = modules[k].kernel_size[0]
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"For GoldenStick Ghost algo, get kernel_size of modules[{k}] failed.") from e
                    if kernel_size_0 == 1:
                        try:
                            stride_0 = modules[k].stride[0]
                        except RuntimeError as e:
                            raise RuntimeError(
                                f"For GoldenStick Ghost algo, get stride of modules[{k}] failed.") from e
                        if modules[k].in_channels == modules[k].out_channels and stride_0 == 1:
                            modules[k] = GhostModule(modules[k].in_channels, modules[k].out_channels,
                                                     kernel_size=modules[k].kernel_size,
                                                     stride=modules[k].stride, padding=modules[k].padding,
                                                     pad_mode=modules[k].pad_mode, act_type='relu', prex=prex)
                    else:
                        if modules[k].in_channels != 3:
                            modules[k] = ConvUnit(modules[k].in_channels, modules[k].out_channels,
                                                  kernel_size=modules[k].kernel_size,
                                                  stride=modules[k].stride,
                                                  padding=self._get_pad(kernel_size_0), act_type='relu',
                                                  num_groups=modules[k].out_channels, use_act=False)
                            modules[k].conv.weight.name = prex + modules[k].conv.weight.name
                            modules[k].bn.gamma.name = prex + modules[k].bn.gamma.name
                            modules[k].bn.beta.name = prex + modules[k].bn.beta.name
                            modules[k].bn.moving_mean.name = prex + modules[k].bn.moving_mean.name
                            modules[k].bn.moving_variance.name = prex + modules[k].bn.moving_variance.name
                elif (not isinstance(modules[k], GhostModule)) and (not isinstance(modules[k], ConvUnit)) \
                        and modules[k]._cells:
                    _inject(modules[k]._cells)

        _inject(net._cells)
        return net

    def apply(self, network, **kwargs):
        """
        Transform input `network` to a Ghost network.

        Args:
            network (Cell): Network to be transformed.

        Returns:
            Ghost network.
        """
        return self._tranform(network)
