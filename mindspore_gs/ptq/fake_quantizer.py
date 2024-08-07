# Copyright 2023 Huawei Technologies Co., Ltd
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
"""ptq fake quantizer."""
from typing import Union
import abc
import numpy as np
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore import dtype as mstype
from mindspore_gs.quantization.quant_utils import get_quant_min_max, cal_quantization_params


class FakeQuantizer(Cell):
    """
    Abstract class for cell which statistic distribute of input x and return quant param.
    """
    def __init__(self):
        super().__init__()
        self._attrs = {}

    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def quant_dtype(self) -> mstype:
        raise NotImplementedError

    @abc.abstractmethod
    def is_per_channel(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def quant_params(self) -> dict:
        raise NotImplementedError

    def set_attr(self, key, value):
        self._attrs[key] = value

    def get_attr(self, key, default=None):
        return self._attrs.get(key, default)


class LinearFakeQuantizer(FakeQuantizer):
    """
    Abstract class derived from FakeQuantizer, suit for linear quantization.
    """

    attr_key_min = "min"
    attr_key_max = "max"
    attr_key_num_bits = "num_bits"
    attr_key_narrow_range = "narrow_range"
    attr_key_symmetric = "symmetric"
    attr_key_signed = "signed"
    attr_key_channel_axis = "channel_axis"
    attr_key_quant_scale = "quant_scale"
    attr_key_quant_zero_point = "quant_zero_point"
    attr_value_quant_algo_name = "linear_quant_algo"

    def name(self) -> str:
        return LinearFakeQuantizer.attr_value_quant_algo_name

    def foo_init(self):
        raise NotImplementedError

    @abc.abstractmethod
    def mins(self) -> Union[list, tuple]:
        raise NotImplementedError

    @abc.abstractmethod
    def maxs(self) -> Union[list, tuple]:
        raise NotImplementedError

    @abc.abstractmethod
    def num_bits(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def narrow_range(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def symmetric(self) -> bool:
        raise NotImplementedError

    def signed(self) -> bool:
        return self.symmetric()

    def channel_axis(self) -> int:
        return -1

    def get_scale_zp(self):
        quant_min, quant_max = get_quant_min_max(self.num_bits(), self.signed(), self.narrow_range())
        input_mins = np.array(self.mins(), dtype=np.float32)
        input_maxs = np.array(self.maxs(), dtype=np.float32)
        scale, zp = cal_quantization_params(input_mins, input_maxs, quant_min, quant_max, symmetric=self.symmetric())
        scale = scale.tolist()
        zp = zp.tolist()
        return scale, zp

    def quant_params(self) -> dict:
        scale, zp = self.get_scale_zp()
        params = {LinearFakeQuantizer.attr_key_min: self.mins(), LinearFakeQuantizer.attr_key_max: self.maxs(),
                  LinearFakeQuantizer.attr_key_num_bits: self.num_bits(),
                  LinearFakeQuantizer.attr_key_narrow_range: self.narrow_range(),
                  LinearFakeQuantizer.attr_key_symmetric: self.symmetric(),
                  LinearFakeQuantizer.attr_key_signed: self.signed(),
                  LinearFakeQuantizer.attr_key_quant_scale: scale,
                  LinearFakeQuantizer.attr_key_quant_zero_point: zp}
        if self.is_per_channel():
            params[LinearFakeQuantizer.attr_key_channel_axis] = self.channel_axis()
        return params


class MinMaxHolder(nn.Cell):
    """MinMaxHolder for deploy"""
    def __init__(self, fq, min_, max_):
        super().__init__()
        self._fq = fq
        self._min = min_
        self._max = max_

    def construct(self, x):
        return self._fq(x, self._min, self._max)

    # pylint: disable=W0613
    def shard(self, in_strategy):
        self._fq = self._fq.shard(in_strategy)


class MinMaxPerLayer(LinearFakeQuantizer):
    """Static minmax by layer"""

    def __init__(self, symmetric=True, narrow_range=False, quant_dtype=mstype.int8, strategy=None):
        super(MinMaxPerLayer, self).__init__()
        self._symmetric = symmetric
        self._narrow_range = narrow_range
        self._quant_dtype = quant_dtype
        if self._narrow_range:
            raise ValueError("Not support narrow_range now.")
        if self._quant_dtype != mstype.int8:
            raise ValueError(f"Only support quant to int8 now, but got {self._quant_dtype}.")
        self._signed = quant_dtype == mstype.int8
        self.float_min = Parameter(Tensor(np.array([float("inf")]), mstype.float32), name="float_min")
        self.float_max = Parameter(Tensor(np.array([-float("inf")]), mstype.float32), name="float_max")
        self._in_strategy = strategy
        if strategy:
            self.min = P.ReduceMin().shard(strategy)
            self.max = P.ReduceMax().shard(strategy)
        else:
            self.min = P.ReduceMin()
            self.max = P.ReduceMax()

    def construct(self, x):
        """
        Defines the computation of MinMaxPerLayer to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        self.float_min = ops.minimum(self.min(x), self.float_min)
        self.float_max = ops.maximum(self.max(x), self.float_max)
        return x

    def foo_init(self):
        """foo init"""
        self.float_min = Parameter(initializer('ones', self.float_min.shape, self.float_min.dtype),
                                   name=self.float_min.name)
        self.float_max = Parameter(initializer('ones', self.float_max.shape, self.float_max.dtype),
                                   name=self.float_max.name)

    def mins(self) -> Union[list, tuple]:
        """mins"""
        return self.float_min.data.asnumpy().tolist()

    def maxs(self) -> Union[list, tuple]:
        """maxs"""
        return self.float_max.data.asnumpy().tolist()

    def num_bits(self) -> int:
        """num bits"""
        return 8

    def narrow_range(self) -> bool:
        """narrow range"""
        return self._narrow_range

    def symmetric(self) -> bool:
        """symmetric"""
        return self._symmetric

    def signed(self) -> bool:
        # for ascend backend, only support int8
        return self._signed

    def quant_dtype(self) -> mstype:
        """quant dtype"""
        return self._quant_dtype

    def is_per_channel(self) -> bool:
        """is per channel"""
        return False

    def __repr__(self):
        fminrepr = "float_min: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_min.name, self.float_min.shape, self.float_min.dtype, self.float_min.requires_grad)
        fminrepr += str(self.float_min.asnumpy().flatten()[0:2]) + ")"
        fmaxrepr = "float_max: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_max.name, self.float_max.shape, self.float_max.dtype, self.float_max.requires_grad)
        fmaxrepr += str(self.float_max.asnumpy().flatten()[0:2]) + ")"
        res = "MinMaxPerLayer<{}, {}>".format(fminrepr, fmaxrepr)
        return res


class MinMaxPerChannel(LinearFakeQuantizer):
    """Static minmax by channel"""

    def __init__(self, axis, output_channel, data_rank, symmetric=True, narrow_range=False, quant_dtype=mstype.int8,
                 strategy=None):
        super(MinMaxPerChannel, self).__init__()
        self._symmetric = symmetric
        self._narrow_range = narrow_range
        self._quant_dtype = quant_dtype
        self._data_rank = data_rank
        if self._narrow_range:
            raise ValueError("Not support narrow_range now.")
        if self._quant_dtype != mstype.int8:
            raise ValueError(f"Only support quant to int8 now, but got {self._quant_dtype}.")
        self._signed = quant_dtype == mstype.int8
        self.float_min = Parameter(Tensor(np.array([float("inf")] * output_channel), mstype.float32),
                                   name="float_min")
        self.float_max = Parameter(Tensor(np.array([-float("inf")] * output_channel), mstype.float32),
                                   name="float_max")
        self._in_strategy = strategy
        if strategy:
            per_channel_strategy = strategy[0][axis]
            input_strategy = ((per_channel_strategy,), (per_channel_strategy,))
            self.min = P.ReduceMin().shard(strategy)
            self.max = P.ReduceMax().shard(strategy)
            self.transpose = P.Transpose().shard(strategy)
            self.assign = P.Assign().shard(input_strategy)
            self.minimum = P.Minimum().shard(input_strategy)
            self.maximum = P.Maximum().shard(input_strategy)
        else:
            self.min = P.ReduceMin()
            self.max = P.ReduceMax()
            self.transpose = P.Transpose()
            self.assign = P.Assign()
            self.minimum = P.Minimum()
            self.maximum = P.Maximum()
        if axis < 0:
            axis += data_rank
        self.axis = axis
        self.min_max_axis = []
        for i in range(data_rank):
            if i != axis:
                self.min_max_axis.append(i)

        pre_dims = axis
        post_dims = data_rank - axis - 1
        self._param_shape = [1] * pre_dims + [-1] + [1] * post_dims

    def foo_init(self):
        self.float_min = Parameter(initializer('ones', self.float_min.shape, self.float_min.dtype),
                                   name=self.float_min.name)
        self.float_max = Parameter(initializer('ones', self.float_max.shape, self.float_max.dtype),
                                   name=self.float_max.name)

    def construct(self, x):
        """
        Defines the computation of MinMaxPerChannel to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        self.assign(self.float_min, self.minimum(self.min(x, self.min_max_axis), self.float_min))
        self.assign(self.float_max, self.maximum(self.max(x, self.min_max_axis), self.float_max))

        return x

    def mins(self) -> Union[list, tuple]:
        return self.float_min.data.asnumpy().reshape(self._param_shape).tolist()

    def maxs(self) -> Union[list, tuple]:
        return self.float_max.data.asnumpy().reshape(self._param_shape).tolist()

    def num_bits(self) -> int:
        return 8

    def narrow_range(self) -> bool:
        return self._narrow_range

    def signed(self) -> bool:
        # for ascend backend, only support int8
        return self._signed

    def symmetric(self) -> bool:
        return self._symmetric

    def quant_dtype(self) -> mstype:
        return self._quant_dtype

    def is_per_channel(self) -> bool:
        return True

    def channel_axis(self) -> int:
        return self.axis

    def __repr__(self):
        fminrepr = "float_min: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_min.name, self.float_min.shape, self.float_min.dtype, self.float_min.requires_grad)
        fminrepr += str(self.float_min.asnumpy().flatten()[0:2]) + ")"
        fmaxrepr = "float_max: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_max.name, self.float_max.shape, self.float_max.dtype, self.float_max.requires_grad)
        fmaxrepr += str(self.float_max.asnumpy().flatten()[0:2]) + ")"
        res = "MinMaxPerChannel<{}, {}>".format(fminrepr, fmaxrepr)
        return res
