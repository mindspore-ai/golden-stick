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
"""Combined layer."""

from mindspore import nn
from mindspore.nn.layer.normalization import BatchNorm2d
from mindspore.nn.cell import Cell
from mindspore_gs.validator import Validator


class Conv2dBn(Cell):
    r"""
    A combination of convolution and Batchnorm layer.

    This part is a more detailed overview of Conv2d operation.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple]): The data type is int or a tuple of 2 integers. Specifies the height
            and width of the 2D convolution window. Single int means the value is for both height and width of
            the kernel. A tuple of 2 ints means the first value is for the height and the other is for the
            width of the kernel.
        stride (int): Specifies stride for all spatial dimensions with the same value. The value of stride must be
            greater than or equal to 1 and lower than any one of the height and width of the `x`. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (int): Implicit paddings on both sides of the `x`. Default: 0.
        dilation (int): Specifies the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location. Its value must be greater than
            or equal to 1 and lower than any one of the height and width of the `x`. Default: 1.
        group (int): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            It can be a Tensor, a string, an Initializer or a number. When a string is specified,
            values from 'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' and 'XavierUniform' distributions as well
            as constant 'One' and 'Zero' distributions are possible. Alias 'xavier_uniform', 'he_uniform', 'ones'
            and 'zeros' are acceptable. Uppercase and lowercase are both acceptable. Refer to the values of
            Initializer for more details. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Possible
            Initializer and string are the same as 'weight_init'. Refer to the values of
            Initializer for more details. Default: 'zeros'.
        has_bn (bool): Specifies to used batchnorm or not. Default: False.
        momentum (float): Momentum for moving average for batchnorm, must be [0, 1]. Default:0.997
        eps (float): Term added to the denominator to improve numerical stability for batchnorm, should be greater
            than 0. Default: 1e-5.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`. The data type is float32.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`. The data type is float32.

    Raises:
        TypeError: If `in_channels`, `out_channels`, `stride`, `padding` or `dilation` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels` or `out_channels` `stride`, `padding` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.Conv2dBn(120, 240, 4, has_bn=True, activation='relu')
        >>> x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
        >>> result = net(x)
        >>> output = result.shape
        >>> print(output)
        (1, 240, 1024, 640)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bn=False,
                 momentum=0.997,
                 eps=1e-5):
        """Initialize Conv2dBn."""
        super(Conv2dBn, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              pad_mode=pad_mode,
                              padding=padding,
                              dilation=dilation,
                              group=group,
                              has_bias=has_bias,
                              weight_init=weight_init,
                              bias_init=bias_init)
        self.has_bn = Validator.check_bool(has_bn, "has_bn", self.cls_name)
        if has_bn:
            self.batchnorm = BatchNorm2d(out_channels, eps, momentum)

    def construct(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.batchnorm(x)
        return x
