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
"""FakeQuantWithMinMaxObserver."""
from __future__ import absolute_import

from functools import partial
from collections import namedtuple
import numpy as np

from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.dtype import QuantDtype
import mindspore.context as context
from mindspore.nn.cell import Cell
from mindspore.ops.operations import _quant_ops as Q
from mindspore_gs.validator import Validator
from mindspore_gs.ops.common.quant_op_utils import get_quant_dtype_num_bits


def _partial_init(cls_or_self, **kwargs):
    """
    Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Examples:
        >>> class Foo:
        ...     def __init__(self, a, b, answer):
        ...         pass
        >>> Foo.partial_init = classmethod(_partial_init)
        >>> foo_builder = Foo.partial_init(a=3, b=4).partial_init(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> result = (id(foo_instance1) == id(foo_instance2))
        >>> print(result)
        False
    """

    class _PartialWrapper:
        r"""
        class of wrapper that allows creation of class factories.
        """

        partial_init = _partial_init

        def __init__(self, p):
            """__init__"""
            self.p = p

        def __call__(self, *args, **keywords):
            """__call__"""
            return self.p(*args, **keywords)

        def __repr__(self):
            """__repr__"""
            return self.p.__repr__()

    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


class _Observer(Cell):
    """
    Base class of Observer. Observer is used to calculate the statistics of specific layer.

    Notes:
        This class is an abstract class.

    Args:
        quant_dtype (QuantDtype): The type of FakeQuant data.
    """

    partial_init = classmethod(_partial_init)

    def __init__(self, quant_dtype):
        """Initialize _Observer."""
        super(_Observer, self).__init__()
        self.quant_dtype = quant_dtype

    def extend_repr(self):
        """extend_repr"""
        s = f"quant_dtype={self.quant_dtype}"
        return s

    def construct(self, x):
        """construct"""


class UniformQuantObserver(_Observer):
    """
    The base class of Uniform Quantization Observer.

    Args:
        quant_dtype (QuantDtype): The type of FakeQuant data. Default: QuantDtype.INT8.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        num_channels (int): declarate the min and max channel size, Default: 1.

    Returns:
        Tensor.
    """

    min_max_map = {
        QuantDtype.INT2: (-2, 1),
        QuantDtype.INT3: (-4, 3),
        QuantDtype.INT4: (-8, 7),
        QuantDtype.INT5: (-16, 15),
        QuantDtype.INT6: (-32, 31),
        QuantDtype.INT7: (-64, 63),
        QuantDtype.INT8: (-128, 127),

        QuantDtype.UINT2: (0, 3),
        QuantDtype.UINT3: (0, 7),
        QuantDtype.UINT4: (0, 15),
        QuantDtype.UINT5: (0, 31),
        QuantDtype.UINT6: (0, 63),
        QuantDtype.UINT7: (0, 127),
        QuantDtype.UINT8: (0, 255)
    }

    def __init__(self, quant_dtype=QuantDtype.INT8, per_channel=False, symmetric=False, narrow_range=False,
                 num_channels=1):
        """Initialize UniformQuantObserver."""
        super(UniformQuantObserver, self).__init__(quant_dtype)
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.num_channels = num_channels


class FakeQuantWithMinMaxObserver(UniformQuantObserver):
    r"""
    Quantization aware operation which provides the fake quantization observer function on data with min and max.

    The detail of the quantization mode `DEFAULT` is described as below:

    The running min/max :math:`x_{min}` and :math:`x_{max}` are computed as:

    .. math::

        \begin{array}{ll} \\
            x_{min} =
            \begin{cases}
                \min(\min(X), 0)
                  & \text{ if } ema = \text{False} \\
                \min((1 - c) \min(X) + \text{c } x_{min}, 0)
                  & \text{ if } \text{otherwise}
            \end{cases}\\
            x_{max} =
            \begin{cases}
                \max(\max(X), 0)
                  & \text{ if } ema = \text{False} \\
                \max((1 - c) \max(X) + \text{c } x_{max}, 0)
                  & \text{ if } \text{otherwise}
            \end{cases}
        \end{array}

    where X is the input tensor, and :math:`c` is the `ema_decay`.

    The scale and zero point zp is computed as:

    .. math::

        \begin{array}{ll} \\
            scale =
            \begin{cases}
                \frac{x_{max} - x_{min}}{Q_{max} - Q_{min}}
                  & \text{ if } symmetric = \text{False} \\
                \frac{2\max(x_{max}, \left | x_{min} \right |) }{Q_{max} - Q_{min}}
                  & \text{ if } \text{otherwise}
            \end{cases}\\
            zp\_min = Q_{min} - \frac{x_{min}}{scale} \\
            zp = \left \lfloor \min(Q_{max}, \max(Q_{min}, zp\_min)) + 0.5 \right \rfloor
        \end{array}

    where :math:`Q_{max}` and :math:`Q_{min}` is decided by quant_dtype, for example, if quant_dtype=INT8,
    then :math:`Q_{max} = 127` and :math:`Q_{min} = -128`.

    The fake quant output is computed as:

    .. math::

        \begin{array}{ll} \\
            u_{min} = (Q_{min} - zp) * scale \\
            u_{max} = (Q_{max} - zp) * scale \\
            u_X = \left \lfloor \frac{\min(u_{max}, \max(u_{min}, X)) - u_{min}}{scale}
            + 0.5 \right \rfloor \\
            output = u_X * scale + u_{min}
        \end{array}

    The detail of the quantization mode `LEARNED_SCALE` is described as below:

    The fake quant output is computed as:

    .. math::

        \bar{X}=\left\{\begin{matrix}
        clip\left ( \frac{X}{maxq},0,1\right ) \qquad \quad if\quad neg\_trunc\\
        clip\left ( \frac{X}{maxq},-1,1\right )\qquad \ if\quad otherwise
        \end{matrix}\right. \\

        output=\frac{floor\left ( \bar{X}\ast  Q_{max}+0.5  \right ) \ast scale }{Q_{max}}

    where X is the input tensor.
    where :math:`Q_{max}` (quant_max) is decided by quant_dtype and neg_trunc, for example, if quant_dtype=INT8
    and neg_trunc works, :math:`Q_{max} = 256` , otherwise :math:`Q_{max} = 127`.

    The maxq is updated by training, and its gradient is calculated as follows:

    .. math::

        \frac{\partial \ output}{\partial \ maxq} = \left\{\begin{matrix}
        -\frac{X}{maxq}+\left \lfloor \frac{X}{maxq} \right \rceil \qquad if\quad bound_{lower}< \frac{X}{maxq}< 1\\
        -1 \qquad \quad \qquad \quad if\quad \frac{X}{maxq}\le bound_{lower}\\
         1  \qquad \quad \qquad \quad if\quad \frac{X}{maxq}\ge  1 \qquad \quad
        \end{matrix}\right. \\

        bound_{lower}=
        \left\{\begin{matrix}
         0\qquad \quad if\quad neg\_trunc\\
        -1\qquad if\quad otherwise
        \end{matrix}\right.

    Then minq is computed as:

    .. math::

        minq=\left\{\begin{matrix}
        0  \qquad \qquad \quad if\quad neg\_trunc\\
        -maxq\qquad if\quad otherwise
        \end{matrix}\right.

    When exporting, the scale and zero point zp is computed as:

    .. math::

        scale=\frac{maxq}{quant\_max} ,\quad zp=0 \\

    zp is equal to 0 consistently, due to the LEARNED_SCALE`s symmetric nature.

    Args:
        min_init (int, float, list): The initialized min value. Default: -6.
        max_init (int, float, list): The initialized max value. Default: 6.
        ema (bool): The exponential Moving Average algorithm updates min and max. Default: False.
        ema_decay (float): Exponential Moving Average algorithm parameter. Default: 0.999.
        per_channel (bool):  Quantization granularity based on layer or on channel. Default: False.
        channel_axis (int): Quantization by channel axis. Default: 1.
        num_channels (int): declarate the min and max channel size, Default: 1.
        quant_dtype (QuantDtype): The datatype of quantization, supporting 4 and 8bits. Default: QuantDtype.INT8.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        narrow_range (bool): Whether the quantization algorithm uses narrow range or not. Default: False.
        quant_delay (int): Quantization delay parameters according to the global step. Default: 0.
        neg_trunc (bool): Whether the quantization algorithm uses negative truncation or not. Default: False.
        mode (str): Optional quantization mode, currently only `DEFAULT`(QAT) and `LEARNED_SCALE` are supported.
            Default: ("DEFAULT")
    Inputs:
        - **x** (Tensor) - The input of FakeQuantWithMinMaxObserver. The input dimension is preferably 2D or 4D.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `min_init` or `max_init` is not int, float or list.
        TypeError: If `quant_delay` is not an int.
        ValueError: If `quant_delay` is less than 0.
        ValueError: If `min_init` is not less than `max_init`.
        ValueError: If `mode` is neither `DEFAULT` nor `LEARNED_SCALE`.
        ValueError: If `mode` is `LEARNED_SCALE` and `symmetric` is not `True`.
        ValueError: If `mode` is `LEARNED_SCALE`, and `narrow_range` is not `True` unless when `neg_trunc` is `True`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> fake_quant = nn.FakeQuantWithMinMaxObserver()
        >>> x = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
        >>> result = fake_quant(x)
        >>> print(result)
        [[ 0.9882355  1.9764705  0.9882355]
         [-1.9764705  0.        -0.9882355]]
    """

    def __init__(self,
                 min_init=-6,
                 max_init=6,
                 ema=False,
                 ema_decay=0.999,
                 per_channel=False,
                 channel_axis=1,
                 num_channels=1,
                 quant_dtype=QuantDtype.INT8,
                 symmetric=False,
                 narrow_range=False,
                 quant_delay=0,
                 neg_trunc=False,
                 mode="DEFAULT"):
        """Initialize FakeQuantWithMinMaxObserver"""
        super(FakeQuantWithMinMaxObserver, self).__init__(quant_dtype=quant_dtype, per_channel=per_channel,
                                                          symmetric=symmetric, narrow_range=narrow_range,
                                                          num_channels=num_channels)
        Validator.check_value_type("min_init", min_init, [int, float, list], type(self).__name__)
        Validator.check_value_type("max_init", max_init, [int, float, list], type(self).__name__)
        Validator.check_non_negative_int(quant_delay, 'quant_delay', self.cls_name)
        self.min_init = min_init
        self.max_init = max_init
        self.quant_dtype = quant_dtype
        self.num_bits = get_quant_dtype_num_bits(quant_dtype)
        self.ema = ema
        self.ema_decay = ema_decay
        self.per_channel = per_channel
        self.num_channels = num_channels
        self.channel_axis = channel_axis
        self.quant_delay = quant_delay
        self.symmetric = symmetric
        self.narrow_range = narrow_range
        self.neg_trunc = neg_trunc
        self.mode = mode
        self.is_ascend = context.get_context('device_target') == "Ascend"
        self.neg = P.Neg()

        min_array = self._get_init_array(self.min_init)
        max_array = self._get_init_array(self.max_init)
        if not np.greater(max_array, min_array).all():
            raise ValueError(f"For '{self.cls_name}', the 'max_init' must be greater than 'min_init', "
                             f"but got 'max_init': {max_init}, 'min_init': {min_init}.")
        if self.mode == "DEFAULT":
            self._default_init(min_array, max_array)
        elif self.mode == "LEARNED_SCALE":
            self._learned_scale_init(min_array, max_array)
        else:
            raise ValueError(f"For '{self.cls_name}', only `DEFAULT` and `LEARNED_SCALE` mode are valid, but got "
                             f"'mode': {self.mode}.")

    def reset(self, quant_dtype=QuantDtype.INT8, min_init=-6, max_init=6):
        r"""
        Reset the quant max parameter (eg. 256) and the initial value of the minq parameter and maxq parameter,
        this function is currently only valid for `LEARNED_SCALE` mode.

        Args:
            quant_dtype (QuantDtype): The datatype of quantization, supporting 4 and 8bits. Default: QuantDtype.INT8.
            min_init (int, float, list): The initialized min value. Default: -6.
            max_init (int, float, list): The initialized max value. Default: 6.
        """
        if self.mode == "LEARNED_SCALE":
            self.quant_dtype = quant_dtype
            self.num_bits = get_quant_dtype_num_bits(quant_dtype)
            self._calculate_quant_max()
            if self.neg_trunc:
                min_init = 0

            self.min_init = min_init
            self.max_init = max_init
            min_array = self._get_init_array(self.min_init)
            max_array = self._get_init_array(self.max_init)
            if not np.greater(max_array, min_array).all():
                raise ValueError(f"For '{self.cls_name}', the 'max_init' must be greater than 'min_init', "
                                 f"but got 'max_init': {max_init}, 'min_init': {min_init}.")

            self.minq.set_data(Tensor(min_array))
            self.maxq.set_data(Tensor(max_array))
            self.quant_max.set_data(Tensor(np.array([self._quant_max]).astype(np.float32)))
        else:
            raise ValueError(f"For '{self.cls_name}', only `LEARNED_SCALE` mode is valid, but got 'mode': {self.mode}.")

    def _default_init(self, min_array, max_array):
        """
        Initialization of `DEFAULT`(QAT) mode.
        """
        # init tensor min and max for fake quantized operation
        self.minq = Parameter(Tensor(min_array), name='quant_min', requires_grad=False)
        self.maxq = Parameter(Tensor(max_array), name='quant_max', requires_grad=False)

        # init fake quant relative op
        if self.per_channel:
            quant_fun = partial(Q.FakeQuantPerChannel, channel_axis=self.channel_axis)
            ema_fun = partial(Q.MinMaxUpdatePerChannel, channel_axis=self.channel_axis)
        else:
            quant_fun = Q.FakeQuantPerLayer
            ema_fun = Q.MinMaxUpdatePerLayer

        self.ema_update = ema_fun(ema=self.ema, ema_decay=self.ema_decay)
        if self.is_ascend:
            self.fake_quant_train = quant_fun(num_bits=get_quant_dtype_num_bits(self.quant_dtype),
                                              symmetric=self.symmetric,
                                              narrow_range=self.narrow_range,
                                              quant_delay=self.quant_delay)
            self.fake_quant_infer = self.fake_quant_train
        else:
            quant_fun = partial(quant_fun,
                                ema=self.ema,
                                ema_decay=self.ema_decay,
                                num_bits=get_quant_dtype_num_bits(self.quant_dtype),
                                symmetric=self.symmetric,
                                narrow_range=self.narrow_range,
                                quant_delay=self.quant_delay)
            self.fake_quant_train = quant_fun(training=True)
            self.fake_quant_infer = quant_fun(training=False)

    def _learned_scale_init(self, min_array, max_array):
        """
        Initialization of `LEARNED_SCALE` mode.
        """
        if not self.symmetric:
            raise ValueError(f"For '{self.cls_name}', the 'LEARNED_SCALE' mode only support 'symmetric' quant, "
                             f"but got 'symmetric': {self.symmetric}. Please set 'symmetric' to True.")
        if self.neg_trunc:
            min_array = self._get_init_array(0)
            if self.narrow_range:
                raise ValueError(f"For '{self.cls_name}', the 'LEARNED_SCALE' mode only support the combination of "
                                 f"'neg_trunc=True and narrow_range=False' config scenario, but got 'narrow_range': "
                                 f"{self.narrow_range}.")
        elif not self.narrow_range:
            raise ValueError(f"For '{self.cls_name}', the 'LEARNED_SCALE' mode only support 'narrow_range=True' "
                             f"config, except for 'neg_trunc=True' scenario. But got 'narrow_range': "
                             f"{self.narrow_range}.")

        self._calculate_quant_max()

        self.minq = Parameter(Tensor(min_array), name='minq')
        self.maxq = Parameter(Tensor(max_array), name='maxq')
        self.quant_max = Parameter(Tensor(np.array([self._quant_max]).astype(np.float32)),
                                   name="quant_max", requires_grad=False)

        # init fake quant relative op
        if self.per_channel:
            quant_fun = partial(Q.FakeLearnedScaleQuantPerChannel, channel_axis=self.channel_axis)
        else:
            quant_fun = Q.FakeLearnedScaleQuantPerLayer

        quant_fun = partial(quant_fun,
                            quant_delay=self.quant_delay,
                            neg_trunc=self.neg_trunc)
        self.fake_quant_train = quant_fun(training=True)
        self.fake_quant_infer = quant_fun(training=False)

    def _get_init_array(self, init_date):
        """
        Convert the initial value to array.
        """
        if isinstance(init_date, list) and self.per_channel and len(init_date) != self.num_channels:
            raise ValueError(f"For '{self.cls_name}', the length of 'min_init/max_init' list must be equal to "
                             f"'num_channels' for perchannel quant scenario, but got 'min_init/max_init': {init_date} "
                             f"and num_channels: {self.num_channels}.")
        if isinstance(init_date, list) and not self.per_channel and len(init_date) != 1:
            raise ValueError(f"For '{self.cls_name}', the length of the 'min_init/max_init' list must be 1 for "
                             f"perlayer quant scenario, but got {len(init_date)}.")

        if isinstance(init_date, list):
            min_max_array = np.array(init_date).astype(np.float32)
        elif self.per_channel and not isinstance(init_date, list):
            min_max_array = np.array([init_date] * self.num_channels).astype(np.float32)
        else:
            min_max_array = np.array([init_date]).astype(np.float32)
        return min_max_array

    def _calculate_quant_max(self):
        """
        The quantization range is calculated according to num_bits.
        """
        if not self.neg_trunc:
            self._quant_max = (1 << (self.num_bits - 1)) - 1
        else:
            self._quant_max = (1 << self.num_bits) - 1

    def extend_repr(self):
        """Display instance object as string."""
        s = 'quant_dtype={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}({}, {}), ' \
            'quant_delay={}, min_init={}, max_init={}'.format(self.quant_dtype, self.symmetric, self.narrow_range,
                                                              self.ema, self.ema_decay, self.per_channel,
                                                              self.channel_axis, self.num_channels, self.quant_delay,
                                                              self.min_init, self.max_init)
        return s

    def construct(self, x):
        """construct."""
        if self.mode == "LEARNED_SCALE":
            if self.training:
                out = self.fake_quant_train(x, self.maxq, self.quant_max)
                if not self.neg_trunc:
                    self.minq = self.neg(self.maxq)
            else:
                out = self.fake_quant_infer(x, self.maxq, self.quant_max)
        else:
            if self.training:
                min_up, max_up = self.ema_update(x, self.minq, self.maxq)
                self.minq = min_up
                self.maxq = max_up
                out = self.fake_quant_train(x, self.minq, self.maxq)
            else:
                out = self.fake_quant_infer(x, self.minq, self.maxq)
        return out


QuantConfig = namedtuple("QuantConfig", ['weight', 'activation'])
quant_config_default = QuantConfig(weight=FakeQuantWithMinMaxObserver.partial_init(),
                                   activation=FakeQuantWithMinMaxObserver.partial_init())
