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
"""SlbFakeQuantizer."""

from functools import partial
from typing import Union

import numpy as np
import mindspore
import mindspore.context as context
from mindspore.ops.operations import _quant_ops as Q
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops.operations._quant_ops import FakeQuantParam
from mindspore.common.dtype import QuantDtype
from mindspore_gs.common.validator import Validator
from mindspore_gs.quantization.fake_quantizer import FakeQuantizer, LinearFakeQuantizer
from mindspore_gs.quantization.quant_utils import get_quant_min_max, cal_quantization_params


class SlbFakeQuantizerPerLayer(FakeQuantizer):
    """
    Implement of SlbFakeQuantizer.
    1. Define weight_list and auxiliary coefficient matrix.
    2. Optimize auxiliary coefficient matrix.
    3. Select quantized weight with the highest probability.

    Args:
        num_bits (int): The quant bit of weight, Default: 1.

    Raises:
        TypeError: If `num_bits` is not an int.
    """
    def __init__(self, num_bits=1):
        super(SlbFakeQuantizerPerLayer, self).__init__()
        self.num_bits = Validator.check_positive_int(num_bits, "num_bits")
        self.argmax = P.Argmax()
        self.onehot = P.OneHot()
        self.softmax = P.Softmax()
        self.sum = P.ReduceSum()
        self.assign = P.Assign()
        self.true_tensor = Tensor(1, mindspore.float32)
        self.false_tensor = Tensor(0, mindspore.float32)

        if self.num_bits == 1:
            self.w_list_init = np.array([-1, 1])
            self.w_list = Parameter(Tensor(self.w_list_init, mindspore.float32), name='w_list', requires_grad=False)
        else:
            self.w_list_init = np.linspace(-1, 1, 2**self.num_bits)
            self.w_list = Parameter(Tensor(self.w_list_init, mindspore.float32), name='w_list', requires_grad=False)

        self.temperature = Parameter(Tensor([1], mindspore.float32),
                                     name="temperature", requires_grad=False)
        self.flag_temperature_end_changing = Parameter(self.false_tensor,
                                                       name="flag_temperature_end_changing", requires_grad=False)

    def set_temperature(self, t):
        """
        Change the temperature in training

        Args:
            t (float): the current temperature. Default: 1.

        Raises:
            TypeError: If `t` is not a float.
        """
        t = Validator.check_positive_float(t, "temperature")
        self.assign(self.temperature, Tensor([t], mindspore.float32))

    def set_temperature_end_flag(self):
        """
        Set flag_temperature_end_changing True when temperature stop increasing
        """
        self.assign(self.flag_temperature_end_changing, self.true_tensor)

    def construct(self, x):
        """
        SlbFakeQuantizer apply method.
        """
        is_training = self.training
        if not is_training:
            # Compute one-hot representation of matrix A's argmax
            weights = self.onehot(self.argmax(x), x.shape[-1], self.true_tensor, self.false_tensor)
        else:
            is_temperature_end_changing = self.flag_temperature_end_changing
            if is_temperature_end_changing == 0:
                # Compute matrix P of probabilities (as softmax of A*T)
                weights = self.softmax(x * self.temperature)
            else:
                # Compute one-hot representation of matrix A's argmax
                weights = self.onehot(self.argmax(x), x.shape[-1], self.true_tensor, self.false_tensor)
        # Compute continuous weights
        weights = weights * self.w_list
        out = self.sum(weights, -1)
        return out

    def extend_repr(self):
        """Display instance object as string."""
        s = 'bit_num={}'.format(self.num_bits)
        return s

    def name(self) -> str:
        return "SLBQuant"

    def quant_dtype(self) -> QuantDtype:
        if self.num_bits == 1:
            return QuantDtype.INT1
        if self.num_bits == 2:
            return QuantDtype.INT2
        if self.num_bits == 4:
            return QuantDtype.INT4
        raise ValueError("Only support 1,2,4 bit weight quantize for slb quantization now!"
                         "Please set weight_quant_dtype in [QuantDtype.INT1, QuantDtype.INT2, QuantDtype.INT4] for "
                         "SlbQuantAwareTraining.")

    def is_per_channel(self) -> bool:
        return False

    def quant_params(self) -> dict:
        scale_w = np.ones(1) * 2. ** (self.num_bits - 1)
        zp_w = np.zeros(1)
        return {FakeQuantParam.attr_key_linear_quant_scale: scale_w.tolist(),
                FakeQuantParam.attr_key_linear_quant_zero_point: zp_w.tolist(),
                "num_bits": self.num_bits}


class SlbActQuantizer(LinearFakeQuantizer):
    """
    Implement of SlbActQuantizer.
    1. statistic the min max value passing through this op
    2. run fake quant execution to simulate the quantize loss
    """

    def __init__(self, ema=False, ema_decay=0.999, symmetric=False, narrow_range=False, num_bits=8, quant_delay=900):
        super(SlbActQuantizer, self).__init__()
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

    def foo_init(self):
        self.float_min = Parameter(initializer('ones', self.float_min.shape, self.float_min.dtype),
                                   name=self.float_min.name)
        self.float_max = Parameter(initializer('ones', self.float_max.shape, self.float_max.dtype),
                                   name=self.float_max.name)

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
        """
        Extracts quantization parameters from the current min and max values.
        """
        quant_min, quant_max = get_quant_min_max(num_bits=self._num_bits, signed=self._symmetric,
                                                 narrow_range=self._narrow_range)
        input_min = self._float_min.data.asnumpy()
        input_max = self._float_max.data.asnumpy()
        scale, zp = cal_quantization_params(input_min, input_max, quant_min, quant_max, symmetric=self._symmetric)
        return input_min, input_max, scale, zp

    def mins(self) -> Union[list, tuple]:
        return self._float_min.data.asnumpy().tolist()

    def maxs(self) -> Union[list, tuple]:
        return self._float_max.data.asnumpy().tolist()

    def num_bits(self) -> int:
        return self._num_bits

    def narrow_range(self) -> bool:
        return self._narrow_range

    def symmetric(self) -> bool:
        return self._symmetric

    def quant_dtype(self) -> QuantDtype:
        if self._num_bits != 8:
            raise TypeError("Only support int8 feature quantize for slb quantization now!"
                            "Please set act_quant_dtype=QuantDtype.INT8 for SlbQuantAwareTraining.")
        return QuantDtype.INT8

    def is_per_channel(self) -> bool:
        return False

    def construct(self, x):
        """
        Forward pass for the SLB fake quantizer.
        """
        if self.training:
            self._float_min, self._float_max = \
                self._min_max_update_func(x, self._float_min, self._float_max)
            out = self._fake_quant_train(x, self._float_min, self._float_max)
        else:
            out = self._fake_quant_infer(x, self._float_min, self._float_max)
        return out
