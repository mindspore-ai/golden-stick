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
"""ScopPruner."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import constexpr
from mindspore import Parameter
from ...comp_algo import CompAlgo


@constexpr
def generate_int(shape):
    """Generate int."""
    return int(shape // 2)


class KfConv2d(nn.Cell):
    """KF Conv2d."""

    def __init__(self, conv_ori, bn_ori, prex):
        super(KfConv2d, self).__init__()
        self.conv = conv_ori
        self.bn = bn_ori
        self.out_channels = self.conv.out_channels
        self.kfscale = Parameter(ops.Ones()((1, self.out_channels, 1, 1), mindspore.float32), requires_grad=True,
                                 name=prex + '.kfscale')
        self.kfscale.data.fill(0.5)
        self.concat_op = ops.Concat(axis=0)

    def construct(self, x):
        """Calculate."""
        x = self.conv(x)
        if self.training:
            num_ori = generate_int(x.shape[0])
            x = self.concat_op((self.kfscale * x[:num_ori] + (1 - self.kfscale) * x[num_ori:], x[num_ori:]))
        x = self.bn(x)
        return x


@constexpr
def generate_tensor(shape, mask_list):
    """Generate tensor."""
    mask = ops.Ones()((shape), mstype.float16).asnumpy()
    for i in mask_list:
        mask[:, i, :, :] = 0.0
    new_mask = Tensor(mask)
    new_mask.set_dtype(mstype.bool_)
    return new_mask


class MaskedConv2dbn(nn.Cell):
    """Mask Conv2d and bn."""

    def __init__(self, kf_conv2d_ori, prex):
        super(MaskedConv2dbn, self).__init__()
        self.target = context.get_context("device_target").upper()
        self.conv = kf_conv2d_ori.conv
        self.bn = kf_conv2d_ori.bn
        self.zeros = ops.Zeros()
        self.one = ops.Ones()
        self.out_index = Parameter(kf_conv2d_ori.out_index, requires_grad=False, name=prex + '.out_index')
        self.cast = ops.Cast()
        self.mask = self.out_index.asnumpy().tolist()

    def construct(self, x):
        """Calculate."""
        x = self.conv(x)
        x = self.bn(x)
        if self.target == 'ASCEND':
            new_mask = generate_tensor(x.shape, self.mask)
            output = ops.MaskedFill()(x, new_mask, 0.0)
            return output
        mask = self.zeros((x.shape), mstype.float32)
        mask[:, self.mask, :, :] = 1.0
        x = x * mask
        return x


class PrunedConv2dbn1(nn.Cell):
    """Prune Conv2d and bn."""

    def __init__(self, masked_module):
        super(PrunedConv2dbn1, self).__init__()

        newconv = nn.Conv2d(in_channels=masked_module.conv.in_channels, out_channels=len(masked_module.out_index),
                            kernel_size=masked_module.conv.kernel_size, stride=masked_module.conv.stride,
                            has_bias=False, padding=masked_module.conv.padding, pad_mode='pad')
        self.conv = newconv
        weight_data = masked_module.conv.weight.data.clone()
        self.conv.weight = Parameter(ops.Gather()(weight_data, masked_module.out_index, 0), requires_grad=True,
                                     name=masked_module.conv.weight.name)

        newbn = nn.BatchNorm2d(len(masked_module.out_index))
        self.bn = newbn
        self.bn.gamma = Parameter(ops.Gather()(masked_module.bn.gamma.data.clone(), masked_module.out_index, 0),
                                  requires_grad=True, name=masked_module.bn.gamma.name)
        self.bn.beta = Parameter(ops.Gather()(masked_module.bn.beta.data.clone(), masked_module.out_index, 0),
                                 requires_grad=True, name=masked_module.bn.beta.name)
        self.bn.moving_mean = Parameter(
            ops.Gather()(masked_module.bn.moving_mean.data.clone(), masked_module.out_index, 0), requires_grad=False,
            name=masked_module.bn.moving_mean.name)
        self.bn.moving_variance = Parameter(
            ops.Gather()(masked_module.bn.moving_variance.data.clone(), masked_module.out_index, 0),
            requires_grad=False, name=masked_module.bn.moving_variance.name)

        self.oriout_channels = masked_module.conv.out_channels
        self.out_index = masked_module.out_index

    def construct(self, x):
        """Calculate."""
        x = self.conv(x)
        x = self.bn(x)
        return x


class PrunedConv2dbnmiddle(nn.Cell):
    """Prune Conv2d and bn."""

    def __init__(self, masked_module):
        super(PrunedConv2dbnmiddle, self).__init__()

        newconv = nn.Conv2d(in_channels=len(masked_module.in_index), out_channels=len(masked_module.out_index),
                            kernel_size=masked_module.conv.kernel_size, stride=masked_module.conv.stride,
                            has_bias=False, padding=masked_module.conv.padding, pad_mode=masked_module.conv.pad_mode)
        self.conv = newconv

        weight_data = masked_module.conv.weight.data.clone()
        weight_data = ops.Gather()(ops.Gather()(weight_data, masked_module.out_index, 0), masked_module.in_index, 1)
        self.conv.weight = Parameter(weight_data, requires_grad=True, name=masked_module.conv.weight.name)

        newbn = nn.BatchNorm2d(len(masked_module.out_index))
        self.bn = newbn
        self.bn.gamma = Parameter(ops.Gather()(masked_module.bn.gamma.data.clone(), masked_module.out_index, 0),
                                  requires_grad=True, name=masked_module.bn.gamma.name)
        self.bn.beta = Parameter(ops.Gather()(masked_module.bn.beta.data.clone(), masked_module.out_index, 0),
                                 requires_grad=True, name=masked_module.bn.beta.name)
        self.bn.moving_mean = Parameter(
            ops.Gather()(masked_module.bn.moving_mean.data.clone(), masked_module.out_index, 0), requires_grad=False,
            name=masked_module.bn.moving_mean.name)
        self.bn.moving_variance = Parameter(
            ops.Gather()(masked_module.bn.moving_variance.data.clone(), masked_module.out_index, 0),
            requires_grad=False, name=masked_module.bn.moving_variance.name)

        self.oriout_channels = masked_module.conv.out_channels
        self.out_index = masked_module.out_index

    def construct(self, x):
        """Calculate."""
        x = self.conv(x)
        x = self.bn(x)
        return x


class PrunedConv2dbn2(nn.Cell):
    """Prune Conv2d and bn."""

    def __init__(self, masked_module):
        super(PrunedConv2dbn2, self).__init__()

        newconv = nn.Conv2d(in_channels=len(masked_module.in_index), out_channels=len(masked_module.out_index),
                            kernel_size=masked_module.conv.kernel_size, stride=masked_module.conv.stride,
                            has_bias=False, padding=masked_module.conv.padding, pad_mode='pad')
        self.conv = newconv

        weight_data = masked_module.conv.weight.data.clone()
        weight_data = ops.Gather()(ops.Gather()(weight_data, masked_module.out_index, 0), masked_module.in_index, 1)
        self.conv.weight = Parameter(weight_data, requires_grad=True, name=masked_module.conv.weight.name)

        newbn = nn.BatchNorm2d(len(masked_module.out_index))
        self.bn = newbn
        self.bn.gamma = Parameter(ops.Gather()(masked_module.bn.gamma.data.clone(), masked_module.out_index, 0),
                                  requires_grad=True, name=masked_module.bn.gamma.name)
        self.bn.beta = Parameter(ops.Gather()(masked_module.bn.beta.data.clone(), masked_module.out_index, 0),
                                 requires_grad=True, name=masked_module.bn.beta.name)
        self.bn.moving_mean = Parameter(
            ops.Gather()(masked_module.bn.moving_mean.data.clone(), masked_module.out_index, 0), requires_grad=False,
            name=masked_module.bn.moving_mean.name)
        self.bn.moving_variance = Parameter(
            ops.Gather()(masked_module.bn.moving_variance.data.clone(), masked_module.out_index, 0),
            requires_grad=False, name=masked_module.bn.moving_variance.name)

        self.oriout_channels = masked_module.conv.out_channels
        self.out_index = masked_module.out_index
        self.zeros = ops.Zeros()

    def construct(self, x):
        """Calculate."""
        x = self.conv(x)
        x = self.bn(x)
        output = self.zeros((x.shape[0], self.oriout_channels, x.shape[2], x.shape[3]), mstype.float32)
        output[:, self.out_index, :, :] = x
        return output


class PrunerKfCompressAlgo(CompAlgo):
    """
    Derived class of GoldenStick. Scop-algorithm. Construct effective knockoff counterparts.


    Args:
        config (Dict): Configuration of `PrunerKfCompressAlgo`. There are no configurable options for
            `PrunerKfCompressAlgo` currently, but for compatibility, the config parameter in the constructor of class A
            is retained.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gs import PrunerKfCompressAlgo
        >>> from models.resnet import resnet50
        >>> class NetToPrune(nn.Cell):
        ...     def __init__(self, num_channel=1):
        ...         super(NetToPrune, self).__init__()
        ...         self.conv = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.bn = nn.BatchNorm2d(6)
        ...
        ...     def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.bn(x)
        ...         return x
        ...
        >>> ## 1) Define network to be quantized
        >>> net = NetToPrune()
        >>> ## 2) Define Knockoff Algorithm
        >>> kf_pruning = PrunerKfCompressAlgo()
        >>> ## 3) Apply Konckoff-algorithm to origin network
        >>> net_pruning = kf_pruning.apply(net)
        >>> ## 4) Print network and check the result. Conv2d and bn should be transformed to KfConv2d.
        >>> print(net_pruning)
        NetToPrune<
          (conv): KfConv2d<
            (conv): Conv2d<input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid,
              padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter
              (name=conv.bn.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter
              (name=conv.bn.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter
              (name=conv.bn.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter
              (name=conv.bn.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
            >
          (bn): SequentialCell<>
          >
    """

    def _tranform(self, net):
        """Transform net."""
        module = net._cells
        keys = list(module.keys())
        for _, k in enumerate(keys):
            if 'layer' in k:
                module[k] = self._tranform_conv(module[k])
        return net

    def _tranform_conv(self, net):
        """Transform conv."""

        def _inject(modules):
            keys = list(modules.keys())
            for ik, k in enumerate(keys):
                if isinstance(modules[k], nn.Conv2d):
                    if k not in ('0', 'conv1_3x3', 'conv1_7x7'):
                        for value, param in modules[k].parameters_and_names():
                            prex = param.name.strip(value)
                        modules[k] = KfConv2d(modules[k], modules[keys[ik + 1]], prex)
                        for params in modules[k].conv.get_parameters():
                            params.name = prex + params.name
                        for params in modules[k].bn.get_parameters():
                            params.name = prex + params.name
                        modules[keys[ik + 1]] = nn.SequentialCell()
                elif (not isinstance(modules[k], KfConv2d)) and modules[k]._cells:
                    _inject(modules[k]._cells)
        _inject(net._cells)
        return net

    def apply(self, network):
        """
        Transform input `network` to a knockoff network.

        Args:
            network (Cell): Network to be pruned.

        Returns:
            Knockoff network.
        """
        return self._tranform(network)


class PrunerFtCompressAlgo(CompAlgo):
    """
    Derived class of GoldenStick. Scop-algorithm.
    FineTune for recover net.

    Args:
        config (Dict): Configuration of `PrunerFtCompressAlgo`. There are no configurable options for
            `PrunerFtCompressAlgo` currently, but for compatibility, the config parameter in the constructor of class A
            is retained.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gs import PrunerKfCompressAlgo, PrunerFtCompressAlgo
        >>> from models.resnet import resnet50
        >>> class NetToPrune(nn.Cell):
        ...     def __init__(self, num_channel=1):
        ...         super(NetToPrune, self).__init__()
        ...         self.conv = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.bn = nn.BatchNorm2d(6)
        ...
        ...     def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.bn(x)
        ...         return x
        ...
        >>> net = NetToPrune()
        >>> kf_pruning = PrunerKfCompressAlgo()
        >>> net_pruning_kf = kf_pruning.apply(net)
        >>> ## 1) Define FineTune Algorithm
        >>> ft_pruning = PrunerFtCompressAlgo()
        >>> ## 2) Apply FineTune-algorithm to origin network
        >>> net_pruning_ft = ft_pruning.apply(net_pruning_kf)
        >>> ## 3) Print network and check the result. Conv2d and bn should be transformed to KfConv2d.
        >>> print(net_pruning_ft)
        NetToPrune<
          (conv): MaskedConv2dbn<
            (conv): Conv2d<input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), pad_mode=valid,
              padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
            (bn): BatchNorm2d<num_features=6, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter
              (name=conv.bn.bn.gamma, shape=(6,), dtype=Float32, requires_grad=True), beta=Parameter
              (name=conv.bn.bn.beta, shape=(6,), dtype=Float32, requires_grad=True), moving_mean=Parameter
              (name=conv.bn.bn.moving_mean, shape=(6,), dtype=Float32, requires_grad=False), moving_variance=Parameter
              (name=conv.bn.bn.moving_variance, shape=(6,), dtype=Float32, requires_grad=False)>
            >
          (bn): SequentialCell<>
          >
    """

    def _recover_conv(self, net):
        """Recover conv."""

        def _inject(modules):
            keys = list(modules.keys())

            for _, k in enumerate(keys):
                if isinstance(modules[k], KfConv2d):
                    for value, param in modules[k].parameters_and_names():
                        prex = param.name.strip(value.split('.')[-1])
                    modules[k] = MaskedConv2dbn(modules[k], prex)
                    for params in modules[k].conv.get_parameters():
                        params.name = prex + params.name
                    for params in modules[k].bn.get_parameters():
                        params.name = prex + params.name
                elif (not isinstance(modules[k], KfConv2d)) and modules[k]._cells:
                    _inject(modules[k]._cells)

        _inject(net._cells)
        return net

    def _pruning_conv(self, net):
        """Prune conv."""

        def _inject(modules):
            keys = list(modules.keys())

            for _, k in enumerate(keys):
                if isinstance(modules[k], MaskedConv2dbn):
                    if 'conv1' in k:
                        modules[k] = PrunedConv2dbn1(modules[k])
                    elif 'conv2' in k:
                        modules[k] = PrunedConv2dbnmiddle(modules[k])
                    elif 'conv3' in k:
                        modules[k] = PrunedConv2dbn2(modules[k])
                elif (not isinstance(modules[k], KfConv2d)) and modules[k]._cells:
                    _inject(modules[k]._cells)

        _inject(net._cells)
        return net

    def apply(self, network):
        """
        Transform a knockoff `network` to a normal and pruned network.

        Args:
            network (Cell): Knockoff network.

        Returns:
            Pruned network.
        """
        return self._recover_conv(network)
