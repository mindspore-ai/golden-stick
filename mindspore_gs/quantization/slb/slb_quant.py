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
"""
Note:
    Constant module for compression. This is interface that is subject to change or deletion.
"""

from functools import partial
import mindspore
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.nn.layer.conv import Conv2d
from mindspore.nn.layer.quant import QuantConfig as OpQuantConfig
from mindspore.common.initializer import initializer
from mindspore.common import initializer as init
from mindspore._checkparam import Validator, twice
from ..constant import QuantDtype
from .slb_fake_quantizer import QBNNFakeQuantizerPerLayer, QBNNACTQuantizer


quant_config_slb_default = OpQuantConfig(weight=partial(QBNNFakeQuantizerPerLayer, num_bits=1),
                                         activation=QBNNACTQuantizer(num_bits=8))


class Conv2dQBNNQuant(nn.Cell):
    r"""
    2D convolution with fake quantized operation layer.

    This part is a more detailed overview of Conv2d operation. For more details about Quantization,
    please refer to the implementation of class of `QBNNFakeQuantizerPerLayer`,
    :class:`mindspore_gs.quantization.slb.slb_fake_quantizer.QBNNFakeQuantizerPerLayer`.

    Args:
        in_channels (int): The number of input channel :math:`C_{in}`.
        out_channels (int): The number of output channel :math:`C_{out}`.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution window.
        stride (Union[int, tuple[int]]): Specifies stride for all spatial dimensions with the same value. Default: 1.
        pad_mode (str): Specifies padding mode. The optional values are "same", "valid", "pad". Default: "same".
        padding (Union[int, tuple[int]]): Implicit paddings on both sides of the `x`. Default: 0.
        dilation (Union[int, tuple[int]]): Specifies the dilation rate to use for dilated convolution. Default: 1.
        group (int): Splits filter into groups, `in_ channels` and `out_channels` must be
            divisible by the number of groups. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the convolution kernel.
            Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the bias vector. Default: 'zeros'.
        quant_config (QuantConfig): Configures the types of quant observer and quant settings of weight and
            activation. Note that, QuantConfig is a special namedtuple, which is designed for quantization
            and can be generated by :func:`mindspore.compression.quant.create_quant_config` method.
            Default: QuantConfig with both items set to default :class:`FakeQuantWithMinMaxObserver`.
        quant_dtype (QuantDtype): Datatype used to quantize weights, weights quantization support int4|int2|int1 now.
            Default: QuantDtype.INT1.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
          The input dimension is preferably 2D or 4D.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `in_channels`, `out_channels` or `group` is not an int.
        TypeError: If `kernel_size`, `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore.compression import quant
        >>> from mindspore_gs.quantization.slb.slb_quant import Conv2dQBNNQuant, quant_config_slb_default
        >>> from mindspore import Tensor
        >>> conv2d_quant = Conv2dQBNNQuant(1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
        ...                               weight_init='ones', quant_config=quant_config_slb_default)
        >>> x = Tensor(np.array([[[[1, 0, 3], [1, 4, 7], [2, 5, 2]]]]), mindspore.float32)
        >>> result = conv2d_quant(x)
        >>> print(result)
        [[[[5.9296875  13.8359375]
           [11.859375  17.78125]]]]
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
                 quant_config=quant_config_slb_default,
                 quant_dtype=QuantDtype.INT1):
        """Initialize Conv2dQBNNQuant."""
        super(Conv2dQBNNQuant, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = has_bias
        self.kernel_size = twice(kernel_size)
        self.stride = twice(stride)
        self.dilation = twice(dilation)
        for kernel_size_elem in self.kernel_size:
            Validator.check_positive_int(kernel_size_elem, 'kernel_size item', self.cls_name)
        for stride_elem in self.stride:
            Validator.check_positive_int(stride_elem, 'stride item', self.cls_name)
        for dilation_elem in self.dilation:
            Validator.check_positive_int(dilation_elem, 'dilation item', self.cls_name)
        if pad_mode not in ('valid', 'same', 'pad'):
            raise ValueError(f"For '{self.cls_name}', the 'pad_mode' must be one of values "
                             f"in ('valid', 'same', 'pad'), but got {pad_mode}.")
        self.pad_mode = pad_mode
        if isinstance(padding, int):
            Validator.check_non_negative_int(padding, 'padding', self.cls_name)
            self.padding = padding
        elif isinstance(padding, tuple):
            for pad in padding:
                Validator.check_non_negative_int(pad, 'padding item', self.cls_name)
            self.padding = padding
        else:
            raise TypeError(f"For '{self.cls_name}', the type of 'padding' must be int/tuple(int), "
                            f"but got {type(padding).__name__}!")
        self.group = Validator.check_positive_int(group, "group", self.cls_name)


        if quant_dtype == QuantDtype.INT4:
            num_bits = 4
        elif quant_dtype == QuantDtype.INT2:
            num_bits = 2
        elif quant_dtype == QuantDtype.INT1:
            num_bits = 1

        self._weight_num = 2**num_bits

        weight_init = init.HeNormal(mode='fan_out', nonlinearity='relu')
        weight_shape = [out_channels, in_channels // group, *self.kernel_size, self._weight_num]
        self.weight = Parameter(initializer(weight_init, weight_shape, mindspore.float32),
                                name='weight', requires_grad=True)


        self.bias_add = P.BiasAdd()
        if Validator.check_bool(has_bias, "has_bias", self.cls_name):
            self.bias = Parameter(initializer(bias_init, [out_channels]), name='bias')
        else:
            self.bias = None

        self.conv = P.Conv2D(out_channel=self.out_channels,
                             kernel_size=self.kernel_size,
                             mode=1,
                             pad_mode=self.pad_mode,
                             pad=self.padding,
                             stride=self.stride,
                             dilation=self.dilation,
                             group=self.group)

        self.fake_quant_weight = quant_config.weight()

    @classmethod
    def from_float(cls, conv: Conv2d, quant_config: OpQuantConfig, weight_quant_dtype: QuantDtype):
        """
        A class method to create `Conv2dQBNNQuant` from a `Conv2d`

        Examples:
            >>> from mindspore import nn
            >>> from mindspore.nn.layer.quant import QuantConfig as OpQuantConfig
            >>> from mindspore_gs.quantization.slb.slb_quant import Conv2dQBNNQuant
            >>> from mindspore_gs.quantization.slb.slb_fake_quantizer import QBNNFakeQuantizerPerLayer, QBNNACTQuantizer
            >>> from mindspore_gs.quantization.constant import QuantDtype
            >>> ic = 10
            >>> oc = 100
            >>> kernel_size = 3
            >>> conv_op = nn.Conv2d(ic, oc, kernel_size)
            >>> # when apply QAT on `conv_op`, QAT need to create a quant conv2d whose weight is fake-quanted
            >>> quant_config: OpQuantConfig = OpQuantConfig(weight=partial(QBNNFakeQuantizerPerLayer, num_bits=1),
            >>>                                             activation=QBNNACTQuantizer(num_bits=8))
            >>> weight_quant_dtype: QuantDtype = QuantDtype.INT1
            >>> conv_quant = Conv2dQBNNQuant.from_float(conv_op, quant_config, weight_quant_dtype)
        """
        conv_quant = cls(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            pad_mode=conv.pad_mode,
            padding=conv.padding,
            dilation=conv.dilation,
            group=conv.group,
            has_bias=conv.has_bias,
            bias_init=conv.bias_init,
            weight_init=conv.weight_init,
            quant_config=quant_config,
            quant_dtype=weight_quant_dtype)
        return conv_quant

    def construct(self, x):
        weight = self.fake_quant_weight(self.weight)
        out = self.conv(x, weight)
        if self.has_bias:
            return self.bias_add(out, self.bias)
        return out

    def extend_repr(self):
        """Display instance object as string."""
        s = 'in_channels={}, out_channels={}, kernel_size={}, stride={}, ' \
            'pad_mode={}, padding={}, dilation={}, group={}, ' \
            'has_bias={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.pad_mode,
                                 self.padding, self.dilation, self.group, self.has_bias)
        return s