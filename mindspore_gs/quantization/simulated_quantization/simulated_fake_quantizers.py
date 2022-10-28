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
"""Simulated fake quantizers."""

from functools import partial
import numpy as np
import mindspore
from mindspore.ops.operations import _quant_ops as Q
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.dtype import QuantDtype
import mindspore.context as context
from ..fake_quantizer import FakeQuantizer
from ..quant_utils import get_quant_min_max, cal_quantization_params, LinearFakeQuantCell


class SimulatedFakeQuantizerPerLayer(FakeQuantizer):
    """
    Implement of SimFakeQuantizer.
    1. statistic the min max value passing through this op
    2. run fake quant execution to simulate the quantize loss
    """

    def __init__(self, ema=False, ema_decay=0.999, symmetric=False, narrow_range=False, num_bits=8, quant_delay=0):
        super(SimulatedFakeQuantizerPerLayer, self).__init__()
        self._ema = ema
        self._ema_decay = ema_decay
        self._symmetric = symmetric
        self._num_bits = num_bits
        self._quant_delay = quant_delay
        self._narrow_range = narrow_range
        self._min_max_update_func = Q.MinMaxUpdatePerLayer(ema=self._ema, ema_decay=self._ema_decay)
        self._is_ascend = context.get_context("device_target") == "Ascend"
        quant_func = Q.FakeQuantPerLayer
        if context.get_context("device_target") == "GPU":
            self._min_max_update_func = Q.MinMaxUpdatePerLayer(ema=self._ema, ema_decay=self._ema_decay)
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
            self._fake_quant_infer = self._fake_quant_train
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

    def extract_quant_param(self):
        quant_min, quant_max = get_quant_min_max(num_bits=self._num_bits, signed=self._symmetric,
                                                 narrow_range=self._narrow_range)
        input_min = self._float_min.data.asnumpy()
        input_max = self._float_max.data.asnumpy()
        scale, zp = cal_quantization_params(input_min, input_max, quant_min, quant_max, symmetric=self._symmetric)
        return input_min, input_max, scale, zp

    def construct(self, x):
        if self.training:
            self._float_min, self._float_max = \
                self._min_max_update_func(x, self._float_min, self._float_max)
            out = self._fake_quant_train(x, self._float_min, self._float_max)
        else:
            out = self._fake_quant_infer(x, self._float_min, self._float_max)
        return out

    def convert_to_fakequantparam(self):
        if self._num_bits != 8:
            raise TypeError("Only support int8 simulate quantization now!"
                            "Please set quant_dtype=QuantDtype.INT8 for SimulatedQuantizationAwareTraining.")
        quant_dtype = QuantDtype.INT8
        _, _, scale, zero_point = self.extract_quant_param()
        new_subcell = LinearFakeQuantCell(quant_dtype=quant_dtype, scale=tuple(scale), zero_point=tuple(zero_point),
                                          is_per_channel=False)
        return new_subcell

class SimulatedFakeQuantizerPerChannel(SimulatedFakeQuantizerPerLayer):
    """
    Derived from SimFakeQuantizerPerLayer, perchannel version of sim fake quantizer
    """

    def __init__(self, num_channels=1, channel_axis=1, ema=False, ema_decay=0.999, symmetric=False, narrow_range=False,
                 num_bits=8, quant_delay=0):
        super(SimulatedFakeQuantizerPerChannel, self).__init__(ema=ema, ema_decay=ema_decay, symmetric=symmetric,
                                                               narrow_range=narrow_range, num_bits=num_bits,
                                                               quant_delay=quant_delay)
        self._channel_axis = channel_axis
        self._num_channels = num_channels
        self._float_min = Parameter(Tensor(np.array([-6] * self._num_channels).astype(np.float32), mindspore.float32),
                                    name="float_min", requires_grad=False)
        self._float_max = Parameter(Tensor(np.array([6] * self._num_channels).astype(np.float32), mindspore.float32),
                                    name="float_max", requires_grad=False)
        self._min_max_update_func = Q.MinMaxUpdatePerChannel(channel_axis=self._channel_axis, ema=self._ema,
                                                             ema_decay=self._ema_decay)
        quant_func = partial(Q.FakeQuantPerChannel, channel_axis=self._channel_axis)
        self._is_ascend = context.get_context("device_target") == "Ascend"
        if context.get_context("device_target") == "GPU":
            self._min_max_update_func = Q.MinMaxUpdatePerChannel(channel_axis=self._channel_axis,
                                                                 ema=self._ema,
                                                                 ema_decay=self._ema_decay)
            quant_func = partial(Q.FakeQuantPerChannel, channel_axis=self._channel_axis)
        self._init_fake_quant_func(quant_func)

    def extract_quant_param(self):
        quant_min, quant_max = get_quant_min_max(num_bits=self._num_bits, signed=self._symmetric,
                                                 narrow_range=self._narrow_range)
        input_min = self._float_min.data.asnumpy()
        input_max = self._float_max.data.asnumpy()
        scale, zp = cal_quantization_params(input_min, input_max, quant_min, quant_max, symmetric=self._symmetric)
        return input_min, input_max, scale, zp

    def extend_repr(self):
        """Display instance object as string."""
        s = 'bit_num={}, symmetric={}, narrow_range={}, ema={}({}), per_channel={}({}, {}), ' \
            'quant_delay={}'.format(self._num_bits, self._symmetric, self._narrow_range,
                                    self._ema, self._ema_decay, True,
                                    self._channel_axis, self._num_channels, self._quant_delay)
        return s

    def convert_to_fakequantparam(self):
        if self._num_bits != 8:
            raise TypeError("Only support int8 simulate quantization now!"
                            "Please set quant_dtype=QuantDtype.INT8 for SimulatedQuantizationAwareTraining.")
        quant_dtype = QuantDtype.INT8
        _, _, scale, zero_point = self.extract_quant_param()
        new_subcell = LinearFakeQuantCell(quant_dtype=quant_dtype, scale=tuple(scale), zero_point=tuple(zero_point),
                                          is_per_channel=True)
        return new_subcell
