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

import numpy as np
from mindspore import Parameter, Tensor, QuantDtype, nn
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops.operations import _quant_ops as Q
from mindspore import ops
from mindspore import dtype as mstype
from mindspore_gs.quantization.fake_quantizer import LinearFakeQuantizer


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
    def shard(self, in_strategy, out_strategy=None, parameter_plan=None, device="Ascend", level=0):
        self._fq = self._fq.shard(in_strategy)


class MinMaxPerLayer(LinearFakeQuantizer):
    """Static minmax by layer"""

    def __init__(self, symmetric=True, narrow_range=False, quant_dtype=QuantDtype.INT8, strategy=None):
        super(MinMaxPerLayer, self).__init__()
        self._symmetric = symmetric
        self._narrow_range = narrow_range
        self._quant_dtype = quant_dtype
        if not self.symmetric:
            raise ValueError("Not support un-symmetric now.")
        if self._narrow_range:
            raise ValueError("Not support narrow_range now.")
        if self._quant_dtype != QuantDtype.INT8:
            raise ValueError("Only support quant to int8 now.")
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
        self.float_min = Parameter(initializer('ones', self.float_min.shape, self.float_min.dtype),
                                   name=self.float_min.name)
        self.float_max = Parameter(initializer('ones', self.float_max.shape, self.float_max.dtype),
                                   name=self.float_max.name)

    def mins(self) -> Union[list, tuple]:
        return self.float_min.data.asnumpy().tolist()

    def maxs(self) -> Union[list, tuple]:
        return self.float_max.data.asnumpy().tolist()

    def num_bits(self) -> int:
        return 8

    def narrow_range(self) -> bool:
        return self._narrow_range

    def symmetric(self) -> bool:
        return self._symmetric

    def quant_dtype(self) -> QuantDtype:
        return self._quant_dtype

    def is_per_channel(self) -> bool:
        return False

    def convert_to_ascend(self):
        fake_quant = Q.FakeQuantPerLayer(num_bits=self.num_bits(), symmetric=self.symmetric())
        fq_cell = MinMaxHolder(fake_quant, self.float_min, self.float_max)
        if self._in_strategy:
            fq_cell.shard((self._in_strategy, (1,), (1,)))
        return fq_cell

    def __repr__(self):
        fminrepr = "float_min: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_min.name, self.float_min.shape, self.float_min.dtype, self.float_min.requires_grad)
        fminrepr += str(self.float_min.data.flatten()[0:2]) + ")"
        fmaxrepr = "float_max: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_max.name, self.float_max.shape, self.float_max.dtype, self.float_max.requires_grad)
        fmaxrepr += str(self.float_max.data.flatten()[0:2]) + ")"
        res = "MinMaxPerLayer<{}, {}>".format(fminrepr, fmaxrepr)
        return res


class MinMaxPerChannel(LinearFakeQuantizer):
    """Static minmax by channel"""

    def __init__(self, axis, output_channel, data_rank, symmetric=True, narrow_range=False, quant_dtype=QuantDtype.INT8,
                 strategy=None):
        super(MinMaxPerChannel, self).__init__()
        self._symmetric = symmetric
        self._narrow_range = narrow_range
        self._quant_dtype = quant_dtype
        self._data_rank = data_rank
        if not self.symmetric:
            raise ValueError("Not support un-symmetric now.")
        if self._narrow_range:
            raise ValueError("Not support narrow_range now.")
        if self._quant_dtype != QuantDtype.INT8:
            raise ValueError("Only support quant to int8 now.")
        self.float_min = Parameter(Tensor(np.array([float("inf")] * output_channel), mstype.float32),
                                   name="float_min")
        self.float_max = Parameter(Tensor(np.array([-float("inf")] * output_channel), mstype.float32),
                                   name="float_max")
        self._in_strategy = strategy
        if strategy:
            self.min = P.ReduceMin().shard(strategy)
            self.max = P.ReduceMax().shard(strategy)
        else:
            self.min = P.ReduceMin()
            self.max = P.ReduceMax()
        if axis < 0:
            axis += data_rank
        self.axis = axis
        perm_shape = [axis]
        for i in range(data_rank):
            if i != axis:
                perm_shape.append(i)
        self._perm_shape = tuple(perm_shape)

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
        tmp = ops.Transpose()(x, self._perm_shape)
        tmp = tmp.reshape((-1, tmp.shape[-1]))
        self.float_min = ops.minimum(self.min(tmp, 1), self.float_min)
        self.float_max = ops.maximum(self.max(tmp, 1), self.float_max)

        return x

    def mins(self) -> Union[list, tuple]:
        return self.float_min.data.asnumpy().reshape(self._param_shape).tolist()

    def maxs(self) -> Union[list, tuple]:
        return self.float_max.data.asnumpy().reshape(self._param_shape).tolist()

    def num_bits(self) -> int:
        return 8

    def narrow_range(self) -> bool:
        return self._narrow_range

    def symmetric(self) -> bool:
        return self._symmetric

    def quant_dtype(self) -> QuantDtype:
        return self._quant_dtype

    def is_per_channel(self) -> bool:
        return True

    def channel_axis(self) -> int:
        return self.axis

    def convert_to_ascend(self):
        fake_quant = Q.FakeQuantPerChannel(num_bits=self.num_bits(), symmetric=self.symmetric(), channel_axis=self.axis)
        fq_cell = MinMaxHolder(fake_quant, self.float_min, self.float_max)
        if self._in_strategy:
            minmax_strategy = self._in_strategy[self.axis]
            fq_cell.shard((self._in_strategy, (minmax_strategy,), (minmax_strategy,)))
        return fq_cell

    def __repr__(self):
        fminrepr = "float_min: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_min.name, self.float_min.shape, self.float_min.dtype, self.float_min.requires_grad)
        fminrepr += str(self.float_min.data.flatten()[0:2]) + ")"
        fmaxrepr = "float_max: (name={}, shape={}, dtype={}, requires_grad={}, first_el="\
            .format(self.float_max.name, self.float_max.shape, self.float_max.dtype, self.float_max.requires_grad)
        fmaxrepr += str(self.float_max.data.flatten()[0:2]) + ")"
        res = "MinMaxPerChannel<{}, {}>".format(fminrepr, fmaxrepr)
        return res
