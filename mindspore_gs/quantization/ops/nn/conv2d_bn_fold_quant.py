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
"""Conv2dBnFoldQuant."""
from __future__ import absolute_import

import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.common.dtype import QuantDtype
from mindspore.ops.operations import _quant_ops as Q
from mindspore_gs import Backend
from mindspore_gs.validator import Validator
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.quantization.simulated_quantization.combined import Conv2dBn
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.quant_utils import fold_batchnorm
from .batchnorm_fold_cell import BatchNormFoldCell


class Conv2dBnFoldQuant(QuantCell):
    r"""
    2D convolution with Batch Normalization operation folded construct.

    This part is a more detailed overview of Conv2d operation. For more details about Quantization,
    please refer to the implementation of class of `FakeQuantWithMinMaxObserver`,
    :class:`FakeQuantWithMinMaxObserver`.

    .. math::
        y = x\times w+  b

        w_{q}=quant(\frac{w}{\sqrt{Var[y]+\epsilon}}*\gamma )

        y_{out}= w_{q}\times x+\frac{b-E[y]}{\sqrt{Var[y]+\epsilon}}*\gamma +\beta

    where :math:`quant` is the continuous execution of quant and dequant. Two convolution
    and Batch Normalization operation are used here, the purpose of the first convolution and Batch Normalization
    is to count the mean `E[y]` and variance `Var[y]` of current batch output for quantization.

    Args:
        fake (bool): Whether Conv2dBnFoldQuant Cell adds FakeQuantWithMinMaxObserver. Default: True.
        quant_dtype (QuantDtype): Specifies the FakeQuant datatype. Default: QuantDtype.INT8.
        freeze_bn (int): The quantization freeze Batch Normalization op is according to the global step.
            Default: 100000.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: If `has_bias` or `fake` is not a bool.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> conv2d_bnfold = nn.Conv2dBnFoldQuant(1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                                      weight_init="ones")
        >>> x = Tensor(np.array([[[[1, 0, 3], [1, 4, 7], [2, 5, 2]]]]), mindspore.float32)
        >>> result = conv2d_bnfold(x)
        >>> print(result)
        [[[[5.9296875 13.8359375]
           [11.859375 17.78125]]]]
    """

    def __init__(self, handler: Conv2dBn, policy: LayerPolicy, fake=True, quant_dtype=QuantDtype.INT8,
                 freeze_bn=100000):
        """Initialize Conv2dBnFoldQuant layer"""
        super(Conv2dBnFoldQuant, self).__init__(handler, policy)
        if not handler.has_bn:
            raise ValueError(f"For '{self.cls_name}', input Conv2dBn should has batchnorm.")
        if context.get_context('device_target') == "CPU":
            raise ValueError(f"For '{self.cls_name}', only the 'Ascend' and 'GPU' platforms"
                             f" are supported, but got {context.get_context('device_target')}.")
        self.in_channels = handler.in_channels
        self.out_channels = handler.out_channels
        self.kernel_size = handler.kernel_size
        self.stride = handler.stride
        self.dilation = handler.dilation
        self.pad_mode = handler.pad_mode
        self.padding = handler.padding
        self.group = handler.group
        self.has_bias = handler.has_bias
        self.freeze_bn = freeze_bn
        self.fake = Validator.check_bool(fake, "fake", self.cls_name)
        self.quant_dtype = quant_dtype
        self.is_gpu = context.get_context('device_target') == "GPU"

        # initialize convolution op and Parameter
        self.conv = P.Conv2D(out_channel=self.out_channels,
                             kernel_size=self.kernel_size,
                             pad_mode=self.pad_mode,
                             pad=self.padding,
                             stride=self.stride,
                             dilation=self.dilation,
                             group=self.group)
        channel_axis = 0
        self.weight = handler.weight
        self.bias_add = P.BiasAdd()
        if self.has_bias:
            self.bias = handler.bias

        # initialize BatchNorm Parameter
        self.gamma = handler.batchnorm.gamma
        self.beta = handler.batchnorm.beta
        self.moving_mean = handler.batchnorm.moving_mean
        self.moving_variance = handler.batchnorm.moving_variance

        # initialize fake ops
        weight_perchannel_args = PerChannelArgs(self.out_channels, channel_axis)
        self._weight_quantizer = policy.get_weight_quantizer(self.weight.name, weight_perchannel_args)
        self.eps = handler.batchnorm.eps
        self.batchnorm_fold = BatchNormFoldCell(epsilon=self.eps, momentum=handler.batchnorm.momentum,
                                                freeze_bn=freeze_bn)
        self.correct_mul = Q.CorrectionMul(channel_axis)
        if context.get_context('device_target') == "Ascend":
            self.batchnorm_fold2_train = Q.BatchNormFold2D(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = Q.BatchNormFold2D(freeze_bn=0)
        elif context.get_context('device_target') == "GPU":
            self.batchnorm_fold2_train = Q.BatchNormFold2(freeze_bn=freeze_bn)
            self.batchnorm_fold2_infer = Q.BatchNormFold2(freeze_bn=0)
        self.step = Parameter(initializer('normal', [1], dtype=mstype.int32), name='step', requires_grad=False)
        self.one = Tensor(1, mstype.int32)
        self.assignadd = P.AssignAdd()

    def weight_quantizer(self):
        return self._weight_quantizer

    def extend_repr(self):
        """Display instance object as string."""
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, fake={}, freeze_bn={}'\
            .format(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.pad_mode, self.padding,
                    self.dilation, self.group, self.fake, self.freeze_bn)
        return s

    def convert(self, backend: Backend = Backend.MS, is_deploy=False):
        if self._converted:
            return
        if backend is not Backend.MS:
            raise ValueError("Only support convert to MS Backend now, got: ", backend)
        if self.has_bias and self.bias:
            raise ValueError("Only support conv2d with out bias.")
        super(Conv2dBnFoldQuant, self).convert(backend, is_deploy)
        self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
        weight, bias = fold_batchnorm(self.weight.data.asnumpy(), self)
        weight_tensor = Tensor(weight)
        bias_tensor = Tensor(bias, mstype.float32)
        self.weight = Parameter(weight_tensor, name=f"{self.weight.name}_bnfold")
        bias_name = f"{self.weight.name}_bias_bnfold"
        self.bias = Parameter(bias_tensor, name=bias_name)
        self.has_bias = True

    # pylint: disable=arguments-differ
    def core_construct(self, x):
        """construct."""
        out_conv = self.conv(x, self.weight)
        if self.has_bias:
            out_conv = self.bias_add(out_conv, self.bias)
        # BN fold1
        batch_mean, batch_std, running_mean, running_std = self.batchnorm_fold(out_conv,
                                                                               self.moving_mean,
                                                                               self.moving_variance,
                                                                               self.step)
        # fake weight
        weight = self.correct_mul(self.weight, self.gamma, running_std)
        if self.fake:
            weight = self._weight_quantizer(weight)
        out = self.conv(x, weight)
        if self.has_bias:
            out = self.bias_add(out, self.bias)
        # BN fold2
        if self.is_gpu:
            if self.training:
                out = self.batchnorm_fold2_train(out, self.beta, self.gamma,
                                                 batch_std, batch_mean, running_std, running_mean, self.step)
                self.assignadd(self.step, self.one)
            else:
                out = self.batchnorm_fold2_infer(out, self.beta, self.gamma,
                                                 batch_std, batch_mean, running_std, running_mean, self.step)
        else:
            if self.training:
                out = self.batchnorm_fold2_train(out, self.beta, self.gamma, batch_std, batch_mean, running_std)
                self.assignadd(self.step, self.one)
            else:
                out = self.batchnorm_fold2_infer(out, self.beta, self.gamma, running_std, running_mean, running_std)
        return out
