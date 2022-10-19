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
"""Simulated Quantization Convert."""

from mindspore.nn.layer.quant import Conv2dQuant, DenseQuant
from mindspore.nn.layer import Conv2d, Dense
from mindspore import ops
from mindspore import Tensor
from mindspore._extends import cell_attr_register


class Conv2dWithFQWeight(Conv2d):
    """
    2D convolution layer with fake quantization weight.
    """
    @cell_attr_register
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 weight_init='normal',
                 weight_name='',
                 has_bias=False,
                 bias_name='',
                 bias_init='zeros'):
        super(Conv2dWithFQWeight, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init)
        self.identity = ops.Identity()
        self.weight.name = weight_name
        if self.bias is not None:
            self.bias.name = bias_name

    def construct(self, x):
        """construct function"""
        weight = self.identity(self.weight)
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


def create_conv2d_from_conv2dquant(conv2dquant: Conv2dQuant, **quant_params):
    """
    A method to create `Conv2d` from a `Conv2dQuant` with quant_params.
    """
    if conv2dquant.bias is None:
        bias = None
    else:
        bias = conv2dquant.bias.value()
    conv = Conv2dWithFQWeight(
        conv2dquant.in_channels,
        conv2dquant.out_channels,
        kernel_size=conv2dquant.kernel_size,
        stride=conv2dquant.stride,
        pad_mode=conv2dquant.pad_mode,
        padding=conv2dquant.padding,
        dilation=conv2dquant.dilation,
        group=conv2dquant.group,
        weight_init=conv2dquant.weight.value(),
        weight_name=conv2dquant.weight.name,
        has_bias=conv2dquant.has_bias,
        bias_init=bias,
        bias_name=conv2dquant)
    for key, value in quant_params.items():
        conv.conv2d.add_prim_attr(key, Tensor(value))
    return conv


class DenseWithFQWeight(Dense):
    """
    The dense connected layer layer with fake quantization weight.
    """
    @cell_attr_register
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 weight_name='',
                 has_bias=False,
                 bias_name='',
                 bias_init='zeros',
                 activation=None):
        super(DenseWithFQWeight, self).__init__(
            in_channels,
            out_channels,
            weight_init,
            bias_init,
            has_bias,
            activation)
        self.identity = ops.Identity()
        self.weight.name = weight_name
        if self.bias is not None:
            self.bias.name = bias_name

    def construct(self, x):
        """construct function"""
        weight = self.identity(self.weight)
        x_shape = self.shape_op(x)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)
        return x


def create_dense_from_densequant(densequant: DenseQuant, **quant_params):
    """
    A method to create `Dense` from a `DenseQuant` with quant_params.
    """
    if densequant.bias is None:
        bias = None
        bias_name = ''
    else:
        bias = densequant.bias.value()
        bias_name = densequant.bias.name
    dense = DenseWithFQWeight(
        in_channels=densequant.in_channels,
        out_channels=densequant.out_channels,
        weight_init=densequant.weight.value(),
        weight_name=densequant.weight.name,
        has_bias=densequant.has_bias,
        bias_init=bias,
        bias_name=bias_name,
        activation=densequant.activation)
    for key, value in quant_params.items():
        dense.matmul.add_prim_attr(key, Tensor(value))
    return dense
