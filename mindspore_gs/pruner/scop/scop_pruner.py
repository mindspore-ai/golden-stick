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
from mindspore._checkparam import Validator, Rel
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore.train.callback import Callback
from mindspore import Tensor
from mindspore.ops import constexpr
from mindspore import Parameter
from ...comp_algo import CompAlgo, CompAlgoConfig


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


class KfCallback(Callback):
    """
    Define konockoff data callback for scop algorithm.
    """

    def step_begin(self, run_context):
        """
        Step_begin.
        """
        cb_params = run_context.original_args()
        cur_data = cb_params.train_dataset_element
        kf = cur_data[0]
        kf_label = cur_data[1]
        idx = ops.Randperm(max_length=kf.shape[0])(mindspore.Tensor([kf.shape[0]], dtype=mstype.int32))
        kf_input = kf[idx, :].view(kf.shape)
        kf_input_label = kf_label[idx].view(kf_label.shape)
        cur_data[0] = ops.Concat(axis=0)((cur_data[0], kf_input))
        cur_data[1] = ops.Concat(axis=0)((cur_data[1], kf_input_label))
        cb_params.train_dataset_element = cur_data


class PrunerKfCompressAlgo(CompAlgo):
    """
    `PrunerKfCompressAlgo` is a subclass of CompAlgo, which implements the use of high imitation data to learn and
    discover redundant convolution kernels in the SCOP algorithm.

    Note:
        For the input parameter `config`, there is currently no optional configuration item for `PrunerKfCompressAlgo`,
        but for compatibility, `config` is reserved and replaced with an empty dictionary during initialization.
        Such as `kf_pruning = PrunerKfCompressAlgo({})`.

    Args:
        config (dict): Configuration of `PrunerKfCompressAlgo`. There are no configurable options for
            `PrunerKfCompressAlgo` currently, but for compatibility, the config parameter in the constructor of class A
            is retained.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gs.pruner import PrunerKfCompressAlgo
        >>> from mindspore import nn
        >>> class Net(nn.Cell):
        ...     def __init__(self, num_channel=1):
        ...         super(Net, self).__init__()
        ...         self.conv = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.bn = nn.BatchNorm2d(6)
        ...
        ...     def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.bn(x)
        ...         return x
        ...
        ... class NetToPrune(nn.Cell):
        ...     def __init__(self):
        ...        super(NetToPrune, self).__init__()
        ...        self.layer = Net()
        ...
        ...     def construct(self, x):
        ...         x = self.layer(x)
        ...         return x
        ...
        >>> ## 1) Define network to be quantized
        >>> net = NetToPrune()
        >>> ## 2) Define Knockoff Algorithm
        >>> kf_pruning = PrunerKfCompressAlgo({})
        >>> ## 3) Apply Konckoff-algorithm to origin network
        >>> net_pruning = kf_pruning.apply(net)
        >>> ## 4) Print network and check the result. Conv2d and bn should be transformed to KfConv2d.
        >>> print(net_pruning)
        NetToPrune<
          (layer): Net<
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
         >
    """

    def callbacks(self, *args, **kwargs):
        """
        Define the callbacks for SCOP algorithm，the callback that generates konockoff data.

        Returns:
            List of instance of SCOP Callbacks.
        """
        cb = []
        cb.append(KfCallback())
        cb.extend(super(PrunerKfCompressAlgo, self).callbacks())
        return cb

    def _tranform(self, net):
        """Transform net."""
        module = net._cells
        keys = list(module.keys())
        for _, k in enumerate(keys):
            if 'layer' in k:
                module[k] = self._tranform_conv(module[k])
        for param in net.get_parameters():
            param.requires_grad = False
        for _, (_, module) in enumerate(net.cells_and_names()):
            if isinstance(module, KfConv2d):
                module.kfscale.requires_grad = True
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

        Raises:
            TypeError: If `network` is not Cell.
        """
        if not isinstance(network, nn.Cell):
            raise TypeError(f'The parameter `network` must be isinstance of Cell, but got {type(network)}.')
        return self._tranform(network)


class PrunerFtCompressAlgo(CompAlgo):
    """
    `PrunerFtCompressAlgo` is a subclass of CompAlgo that implements the ability to remove redundant convolution kernels
    and fully train the network.

    Args:
        config (dict): Configuration of `PrunerFtCompressAlgo`, keys are attribute names,
            values are attribute values. Supported attribute are listed below:

            - prune_rate (float): number in [0.0, 1.0)

    Raises:
        TypeError: If `prune_rate` is not float.
        ValueError: If `epoch_size` is less than 0 or greater than or equal to 1.


    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore_gs.pruner import PrunerKfCompressAlgo, PrunerFtCompressAlgo
        >>> from mindspore import nn
        >>> class Net(nn.Cell):
        ...     def __init__(self, num_channel=1):
        ...         super(Net, self).__init__()
        ...         self.conv = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.bn = nn.BatchNorm2d(6)
        ...
        ...     def construct(self, x):
        ...         x = self.conv(x)
        ...         x = self.bn(x)
        ...         return x
        ...
        ... class NetToPrune(nn.Cell):
        ...     def __init__(self):
        ...        super(NetToPrune, self).__init__()
        ...        self.layer = Net()
        ...
        ...     def construct(self, x):
        ...         x = self.layer(x)
        ...         return x
        ...
        >>> net = NetToPrune()
        >>> kf_pruning = PrunerKfCompressAlgo({})
        >>> net_pruning_kf = kf_pruning.apply(net)
        >>> ## 1) Define FineTune Algorithm
        >>> ft_pruning = PrunerFtCompressAlgo({'prune_rate': 0.5})
        >>> ## 2) Apply FineTune-algorithm to origin network
        >>> net_pruning_ft = ft_pruning.apply(net_pruning_kf)
        >>> ## 3) Print network and check the result. Conv2d and bn should be transformed to KfConv2d.
        >>> print(net_pruning_ft)
        NetToPrune<
         (layer): Net<
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
        >
    """

    def _create_config(self):
        """Create PrunerFtCompressConfig."""
        self._config = PrunerFtCompressConfig()

    def _update_config_from_dict(self, config: dict):
        """Update prune `config` from a dict"""
        self.set_prune_rate(config.get("prune_rate", 0.0))

    def set_prune_rate(self, prune_rate: float):
        """
        Set value of prune_rate of `_config`

        Args:
            prune_rate (float): the size of network needs to be pruned.

        Raises:
            TypeError: If `prune_rate` is not float.
            ValueError: If `prune_rate` is less than 0. or greater than 1.
        """
        prune_rate = Validator.check_float_range(prune_rate, 0.0, 1.0, Rel.INC_LEFT,
                                                 "prune_rate", self.__class__.__name__)
        self._config.prune_rate = prune_rate

    def _recover(self, net):
        """Recover."""
        kfconv_list = []
        for _, (_, module) in enumerate(net.cells_and_names()):
            if isinstance(module, KfConv2d):
                kfconv_list.append(module)
        for param in net.get_parameters():
            param.requires_grad = True
        for _, (_, module) in enumerate(net.cells_and_names()):
            if isinstance(module, KfConv2d):
                module.score = module.bn.gamma.data.abs() * ops.Squeeze()(
                    module.kfscale.data - (1 - module.kfscale.data))
        for kfconv in kfconv_list:
            kfconv.prune_rate = self._config.prune_rate
        for _, (_, module) in enumerate(net.cells_and_names()):
            if isinstance(module, KfConv2d):
                _, index = ops.Sort()(module.score)
                num_pruned_channel = int(module.prune_rate * module.score.shape[0])
                module.out_index = index[num_pruned_channel:]
        return self._recover_conv(net)

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

        Raises:
            TypeError: If `network` is not Cell.
        """
        if not isinstance(network, nn.Cell):
            raise TypeError(f'The parameter `network` must be isinstance of Cell, but got {type(network)}.')
        return self._recover(network)


class PrunerFtCompressConfig(CompAlgoConfig):
    """Config for PrunerFtCompress."""

    def __init__(self):
        super(PrunerFtCompressConfig, self).__init__()
        self.prune_rate = 0.0
