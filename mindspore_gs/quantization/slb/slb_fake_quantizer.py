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

import numpy as np
import mindspore
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from ..fake_quantizer import FakeQuantizer


class QBNNFakeQuantizerPerLayer(FakeQuantizer):
    """
    Implement of SlbFakeQuantizer.
    1. Define weight_list and auxiliary coefficient matrix.
    2. Optimize auxiliary coefficient matrix.
    3. Select quantized weight with the highest probability.
    """
    def __init__(self, num_bits=1):
        super(QBNNFakeQuantizerPerLayer, self).__init__()
        self._num_bits = num_bits
        self.argmax = P.Argmax()
        self.onehot = P.OneHot()
        self.softmax = P.Softmax()
        self.sum = P.ReduceSum()
        self.assign = P.Assign()
        self.true_tensor = Tensor(1, mindspore.float32)
        self.false_tensor = Tensor(0, mindspore.float32)

        if self._num_bits == 1:
            self.w_list = Parameter(Tensor([-1, 1], mindspore.float32).view(1, 1, 1, 1, -1),
                                    name='w_list', requires_grad=False)
        else:
            self.w_list_init = np.linspace(-1, 1, 2**self._num_bits)
            self.w_list = Parameter(Tensor(self.w_list_init, mindspore.float32).view(1, 1, 1, 1, -1),
                                    name='w_list', requires_grad=False)

        self.temperature = Parameter(Tensor([1], mindspore.float32),
                                     name="temperature", requires_grad=False)
        self.flag_temperature_end_changing = Parameter(self.false_tensor,
                                                       name="flag_temperature_end_changing", requires_grad=False)

    def set_temperature(self, t):
        """
        Change the temperature in training
        """
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
        if not self.training:
            # Compute one-hot representation of matrix A's argmax
            weights = self.onehot(self.argmax(x), x.shape[-1], self.true_tensor, self.false_tensor)
        else:
            is_temperature_end_changing = self.flag_temperature_end_changing
            if not is_temperature_end_changing:
                # Compute matrix P of probabilities (as softmax of A*T)
                weights = self.softmax(x * self.temperature)
            else:
                # Compute one-hot representation of matrix A's argmax
                weights = self.onehot(self.argmax(x), x.shape[-1], self.true_tensor, self.false_tensor)
        # Compute continuous weights
        weights = weights * self.w_list
        out = self.sum(weights, -1)
        return out


class SlbQuantizer(nn.Cell):
    """
    Quantizer for QBNNACTQuantizer.
    """
    def __init__(self, num_bits=8):
        super(SlbQuantizer, self).__init__()
        self.reshape = P.Reshape()
        self.round = P.Floor()
        self._num_bits = num_bits

        scale = 2 ** self._num_bits - 1
        self.scale = Tensor(scale, mindspore.float32)

    def construct(self, input_):
        scale = self.reshape(self.scale, (1, 1, 1, 1))
        return self.round(input_ * scale) / scale

    def bprop(self, input_, out, dout):
        """
        Do backpropogation
        """
        return (dout,)


class QBNNACTQuantizer(FakeQuantizer):
    """
    Activation quantization method.
    """
    def __init__(self, num_bits=8):
        super(QBNNACTQuantizer, self).__init__()
        self._num_bits = num_bits
        self.min_val = Tensor(0, mindspore.float32)
        self.max_val = Tensor(1000, mindspore.float32)
        self.max_val2 = Tensor(1.49999, mindspore.float32)
        self.relu_upper_bound = Parameter(Tensor([1], mindspore.float32),
                                          name='relu_upper_bound', requires_grad=True)
        self.quantize = SlbQuantizer(num_bits)

    def construct(self, x):
        """
        Activation quantization apply method.
        """
        # Clip from above
        x = x - (x >= self.relu_upper_bound).astype(mindspore.float32)*(x - self.relu_upper_bound)
        # Normalize
        x = x / self.relu_upper_bound
        # Clip negative values;
        # obtain float "coordinates" in the bit space
        x = C.clip_by_value(x, self.min_val, self.max_val)
        x = self.quantize(x)
        # Scale back to original magnitude
        x = x * self.relu_upper_bound
        return x
