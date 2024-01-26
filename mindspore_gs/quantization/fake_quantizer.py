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
FakeQuantizer, statistic distribute of input x and return quant param.
"""
import abc
from typing import Union

import numpy as np
from mindspore.nn.cell import Cell
from mindspore import QuantDtype
from mindspore.ops.operations._quant_ops import FakeQuantParam
from mindspore_gs.quantization.quant_utils import get_quant_min_max, cal_quantization_params


class FakeQuantParamCell(Cell):
    """FakeQuantParamCell."""
    def __init__(self, op: FakeQuantParam):
        super().__init__()
        if not isinstance(op, FakeQuantParam):
            raise TypeError("Input ops should be a FakeQuantParam, but got: ", type(op))
        self.fq = op

    def construct(self, x):
        return self.fq(x)

    # pylint: disable=W0613
    def shard(self, in_strategy, out_strategy=None, parameter_plan=None, device="Ascend", level=0):
        self.fq = self.fq.shard(in_strategy=in_strategy, out_strategy=out_strategy)


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
    def quant_dtype(self) -> QuantDtype:
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

    def convert_to_fakequantparam(self) -> FakeQuantParamCell:
        fq_param = FakeQuantParam(self.quant_dtype(), self.name(), self.is_per_channel(), **self.quant_params())
        return FakeQuantParamCell(fq_param)


class LinearFakeQuantizer(FakeQuantizer):
    """
    Abstract class derived from FakeQuantizer, suit for linear quantization.
    """

    attr_key_min = "min"
    attr_key_max = "max"
    attr_key_num_bits = "num_bits"
    attr_key_narrow_range = "narrow_range"
    attr_key_symmetric = "symmetric"
    attr_key_channel_axis = "channel_axis"

    def name(self) -> str:
        return FakeQuantParam.attr_value_linear_quant_algo_name

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

    def channel_axis(self) -> int:
        return -1

    def get_scale_zp(self):
        quant_min, quant_max = get_quant_min_max(self.num_bits(), self.symmetric(), self.narrow_range())
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
                  FakeQuantParam.attr_key_linear_quant_scale: scale,
                  FakeQuantParam.attr_key_linear_quant_zero_point: zp}
        if self.is_per_channel():
            params[LinearFakeQuantizer.attr_key_channel_axis] = self.channel_axis()
        return params
