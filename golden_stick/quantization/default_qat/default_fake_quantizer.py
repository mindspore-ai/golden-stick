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
"""DefaultQuantizeOp."""

from functools import partial
import numpy as np
import mindspore
from mindspore.ops.operations import _quant_ops as Q
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.context as context
from ..fake_quantizer import FakeQuantizer
from ..quant_utils import compute_kl_threshold


def _calculate_quant_max(num_bits, neg_trunc=False):
    """
    Define how to calculate the max value of quant data by bit num.

    Args:
        num_bits: the bits num after data be quantized.
        neg_trunc: whether the quantization algorithm uses negative truncation or not. Default: False

    Returns:
        max quantized data.
    """
    if neg_trunc:
        quant_max = (1 << num_bits) - 1
    else:
        quant_max = (1 << (num_bits - 1)) - 1
    return quant_max


class DefaultFakeQuantizerPerLayer(FakeQuantizer):
    """
    Default implement of MinMaxFakeQuantizer.
    1. statistic the min max value passing through this op
    2. run fake quant execution to simulate the quantize loss
    """

    def __init__(self, ema=False, ema_decay=0.999, symmetric=False, narrow_range=False, num_bits=8, quant_delay=0):
        super(DefaultFakeQuantizerPerLayer, self).__init__()
        self._ema = ema
        self._ema_decay = ema_decay
        self._symmetric = symmetric
        self._num_bits = num_bits
        self._quant_delay = quant_delay
        self._narrow_range = narrow_range
        self._min_max_update_func = Q.MinMaxUpdatePerLayer(ema=self._ema, ema_decay=self._ema_decay)
        self._is_ascend = context.get_context("device_target") == "Ascend"
        quant_func = Q.FakeQuantPerLayer
        self._init_fake_quant_func(quant_func)
        self._float_min = Parameter(Tensor(np.array([-6]).astype(np.float32), mindspore.float32),
                                    name="float_min", requires_grad=False)
        self._float_max = Parameter(Tensor(np.array([6]).astype(np.float32), mindspore.float32),
                                    name="float_max", requires_grad=False)

    def _init_fake_quant_func(self, quant_func):
        """
        Define fake quant function according to device
        """
        if self._is_ascend:
            self._fake_quant_train = quant_func(num_bits=self._num_bits,
                                                symmetric=self._symmetric,
                                                narrow_range=self._narrow_range,
                                                quant_delay=self._quant_delay)
            self._fake_quant_infer = self.fake_quant_train
        else:
            quant_func = partial(quant_func,
                                 ema=self._ema,
                                 ema_decay=self._ema_decay,
                                 num_bits=self._num_bits,
                                 symmetric=self._symmetric,
                                 narrow_range=self._narrow_range,
                                 quant_delay=self._quant_delay)
            self._fake_quant_train = quant_func(training=True)
            self._fake_quant_infer = quant_func(training=False)

    def extend_repr(self):
        """Display instance object as string."""
        s = 'bit_num={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}, ' \
            'quant_delay={}'.format(self._num_bits, self._symmetric, self._narrow_range,
                                    self._ema, self._ema_decay, False, self._quant_delay)
        return s

    def construct(self, x):
        if self.training:
            self._float_min, self._float_max = \
                self._min_max_update_func(x, self._float_min, self._float_max)
            out = self._fake_quant_train(x, self._float_min, self._float_max)
        else:
            out = self._fake_quant_infer(x, self._float_min, self._float_max)
        return out


class DefaultFakeQuantizerPerChannel(DefaultFakeQuantizerPerLayer):
    """
    Derived from DefaultFakeQuantizerPerLayer, perchannel version of default fake quantizer
    """

    def __init__(self, num_channels=1, channel_axis=1, ema=False, ema_decay=0.999, symmetric=False, narrow_range=False,
                 num_bits=8, quant_delay=0):
        super(DefaultFakeQuantizerPerChannel, self).__init__(ema=ema, ema_decay=ema_decay, symmetric=symmetric,
                                                             narrow_range=narrow_range, num_bits=num_bits,
                                                             quant_delay=quant_delay)
        self._float_min = Parameter(Tensor(np.array([-6] * num_channels).astype(np.float32), mindspore.float32),
                                    name="float_min", requires_grad=False)
        self._float_max = Parameter(Tensor(np.array([6] * num_channels).astype(np.float32), mindspore.float32),
                                    name="float_max", requires_grad=False)
        quant_func = partial(Q.FakeQuantPerChannel, channel_axis=channel_axis)
        self._init_fake_quant_func(quant_func)
        self._min_max_update_func = Q.MinMaxUpdatePerChannel(channel_axis=channel_axis, ema=ema, ema_decay=ema_decay)

    def extend_repr(self):
        """Display instance object as string."""
        s = 'bit_num={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}({}, {}), ' \
            'quant_delay={}'.format(self._num_bits, self._symmetric, self._narrow_range,
                                    self._ema, self._ema_decay, True,
                                    self._channel_axis, self._num_channels, self._quant_delay)
        return s


class LearnedFakeQuantizerPerLayer(FakeQuantizer):
    """
    Derived class of FakeQuantizer. Use learning-rate from each epoch to compute scale and zero-point.
    """

    def __init__(self, num_bits=8, quant_delay=0, min_init=-6, max_init=6, neg_trunc=False):
        super(LearnedFakeQuantizerPerLayer, self).__init__()
        self._num_bits = num_bits
        self.neg_trunc = neg_trunc
        self._quant_max = _calculate_quant_max(self._num_bits, self.neg_trunc)
        self.quant_max = Parameter(Tensor(np.array([self._quant_max]).astype(np.float32)))
        quant_func = partial(Q.FakeLearnedScaleQuantPerLayer, quant_delay=quant_delay, neg_trunc=self.neg_trunc)
        self.fake_quant_train = quant_func(training=True)
        self.fake_quant_infer = quant_func(training=False)
        self._float_min = Parameter(Tensor([min_init], mindspore.float32), name="float_min")
        self._float_max = Parameter(Tensor([max_init], mindspore.float32), name="float_max")

    def compute_quant_param(self, weight_param):
        max_init = [compute_kl_threshold(weight_param, self._num_bits)]
        min_init = [-x for x in max_init]
        self._float_min.set_data(Tensor(self._get_init_array(max_init)))
        self._float_max.set_data(Tensor(self._get_init_array(min_init)))

    def construct(self, x):
        if self.training:
            out = self.fake_quant_train(x, self._float_max, self.quant_max)
        else:
            out = self.fake_quant_infer(x, self._float_max, self.quant_max)
        return out


class LearnedFakeQuantizePerChannel(FakeQuantizer):
    """
    Derived class of FakeQuantizer. perchannel version of LearnedFakeQuantizerPerLayer.
    """

    def __init__(self, num_bits=8, num_channels=1, channel_axis=1, quant_delay=0,
                 float_min=-6, float_max=6, neg_trunc=False):
        super(LearnedFakeQuantizePerChannel, self).__init__()
        self._num_bits = num_bits
        self._quant_max = _calculate_quant_max(self._num_bits, neg_trunc)
        self.quant_max = Parameter(Tensor(np.array([self._quant_max]).astype(np.float32)))
        quant_func = partial(Q.FakeLearnedScaleQuantPerChannel, quant_delay=quant_delay, neg_trunc=neg_trunc,
                             channel_axis=channel_axis)
        self.fake_quant_train = quant_func(training=True)
        self.fake_quant_infer = quant_func(training=False)
        self._num_channels = num_channels
        self._float_min = Parameter(Tensor(self._get_init_array(float_min), mindspore.float32), name="float_min")
        self._float_max = Parameter(Tensor(self._get_init_array(float_max), mindspore.float32), name="float_max")

    def compute_quant_param(self, weight_param):
        max_init = [compute_kl_threshold(weight_para_each.asnumpy(), self._num_bits)
                    for weight_para_each in weight_param]
        min_init = [-x for x in max_init]
        self._float_min.set_data(Tensor(self._get_init_array(max_init)))
        self._float_max.set_data(Tensor(self._get_init_array(min_init)))

    def _get_init_array(self, init_data):
        """
        Convert the initial value to array.
        """
        if isinstance(init_data, list) and len(init_data) != self._num_channels:
            raise ValueError(f"For '{self.cls_name}', the length of 'min_init/max_init' list should be equal to "
                             f"'num_channels' for perchannel quant scenario, but got 'min_init/max_init': {init_data} "
                             f"and num_channels: {self._num_channels}.")

        if isinstance(init_data, list):
            min_max_array = np.array(init_data).astype(np.float32)
        else:
            min_max_array = np.array([init_data] * self._num_channels).astype(np.float32)
        return min_max_array

    def construct(self, x):
        if self.training:
            out = self.fake_quant_train(x, self._float_max, self.quant_max)
        else:
            out = self.fake_quant_infer(x, self._float_max, self.quant_max)
        return out
