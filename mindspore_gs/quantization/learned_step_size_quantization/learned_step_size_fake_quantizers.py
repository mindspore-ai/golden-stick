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
"""learned scale fake quantizers."""

from functools import partial
import numpy as np
import mindspore
from mindspore.ops.operations import _quant_ops as Q
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
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


class LearnedStepSizeFakeQuantizerPerLayer(FakeQuantizer):
    """
    Derived class of FakeQuantizer. Use learning-rate from each epoch to compute scale and zero-point.
    """

    def __init__(self, num_bits=8, quant_delay=0, min_init=-6, max_init=6, neg_trunc=False, symmetric=True,
                 narrow_range=True):
        super(LearnedStepSizeFakeQuantizerPerLayer, self).__init__()
        self._num_bits = num_bits
        self.neg_trunc = neg_trunc
        self._symmetric = symmetric
        self._narrow_range = narrow_range
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


class LearnedStepSizeFakeQuantizePerChannel(FakeQuantizer):
    """
    Derived class of FakeQuantizer. perchannel version of LearnedFakeQuantizerPerLayer.
    """

    def __init__(self, num_bits=8, num_channels=1, channel_axis=1, quant_delay=0,
                 float_min=-6, float_max=6, neg_trunc=False, symmetric=True, narrow_range=True):
        super(LearnedStepSizeFakeQuantizePerChannel, self).__init__()
        self._num_bits = num_bits
        self._symmetric = symmetric
        self._narrow_range = narrow_range
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
