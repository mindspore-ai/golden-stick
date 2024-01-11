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
"""ptq quant cells."""
import abc

import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import log as logger
from mindspore import Parameter, Tensor, dtype
from mindspore_gs import Backend
from mindspore_gs.quantization.fake_quantizer import LinearFakeQuantizer
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.quant_utils import get_quant_min_max, quant_data
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.ptq.convert_utils import convert_to_antiquant, convert_to_quant, convert_to_dequant
from mindformers.modules import Linear


class PTQCell(QuantCell):
    @abc.abstractmethod
    def calibrate(self):
        raise NotImplementedError


class LinearQuant(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy: LayerPolicy):
        super(LinearQuant, self).__init__(linear, policy)
        self._linear = linear
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        input_fq_args = {}
        weight_perchannel_args = PerChannelArgs(self._linear.out_channels, self._weight_axis, rank)
        weight_fq_args = {}
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            input_fq_args["strategy"] = (self._linear.matmul.in_strategy[0],)
            weight_fq_args["strategy"] = (self._linear.matmul.in_strategy[1],)
        self._input_quantizer = self._policy.get_input_quantizer(**input_fq_args)
        self._output_quantizer = None
        self._weight_quantizer = self._policy.get_weight_quantizer(self._linear.weight.name, weight_perchannel_args,
                                                                   **weight_fq_args)

        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        if self._input_quantizer:
            self._input_quantizer.float_min.data.name = prex + "_input_float_min"
            self._input_quantizer.float_max.data.name = prex + "_input_float_max"
        self._weight_quantizer.float_min.data.name = prex + "_weight_float_min"
        self._weight_quantizer.float_max.data.name = prex + "_weight_float_max"

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    def convert(self, backend: Backend = Backend.MS):
        if backend == Backend.MS:
            super(LinearQuant, self).convert(backend)
            if self._weight_quantizer:
                self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            return
        if backend == Backend.FAKE_QUANT:
            if self._input_quantizer:
                self._input_quantizer = self._input_quantizer.convert_to_ascend()
            if self._weight_quantizer:
                self._weight_quantizer = self._weight_quantizer.convert_to_ascend()
            return
        if backend == Backend.GE_ASCEND:
            weight_only = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                          self._weight_quantizer.get_attr("weight_only_quant", False)
            all_quant = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                        isinstance(self._input_quantizer, LinearFakeQuantizer)
            # quant weight to int8
            if all_quant or weight_only:
                super(LinearQuant, self).convert(backend)
                self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
                weight_quantizer: P.FakeQuantParam = self._weight_quantizer.fq
                if hasattr(self._linear, "dtype"):
                    weight = self._linear.cast(self._linear.weight, self._linear.dtype)
                else:
                    weight = self._linear.weight
                weight = weight.asnumpy()
                quant_min, quant_max = get_quant_min_max(weight_quantizer.attrs[LinearFakeQuantizer.attr_key_num_bits],
                                                         weight_quantizer.attrs[
                                                             LinearFakeQuantizer.attr_key_narrow_range])
                scale = weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_scale]
                zp = weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_zero_point]
                weight_quant = quant_data(weight, np.array(scale), np.array(zp), quant_min, quant_max,
                                          self._weight_axis)
                self._linear.weight = Parameter(Tensor(weight_quant, dtype=dtype.int8), name=self._linear.weight.name)
            # convert to ascend quant layer
            if all_quant:
                self._output_quantizer = convert_to_dequant(self._input_quantizer, self._weight_quantizer)
                self._input_quantizer = convert_to_quant(self._input_quantizer)
            elif weight_only:
                self._input_quantizer = None
                self._output_quantizer = None
                self._weight_quantizer = convert_to_antiquant(self._weight_quantizer)
            else:
                logger.info(f"LinearQuant {self} is not quanted.")
                return

    def calibrate(self):
        logger.info(f"calibrating weight of Linear Cell: {self._linear.weight.name}")
        if hasattr(self._linear, "dtype"):
            weight = self._linear.cast(self._linear.weight, self._linear.dtype)
        else:
            weight = self._linear.weight
        self._weight_quantizer(weight)

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = P.Shape()(x)[:-1] + (self._linear.out_channels,)
        x = P.Reshape()(x, (-1, self._linear.in_channels))
        if hasattr(self._linear, "expert_flag") and self._linear.expert_flag:
            if self._linear.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self._linear.expert_num, self._linear.expert_group_size,
                                    self._linear.in_channels))
            else:
                x = P.Reshape()(x, (self._linear.outer_batch, self._linear.expert_num, -1, self._linear.in_channels))
        ori_dtype = F.dtype(x)
        linear_dtype = self._linear.dtype if hasattr(self._linear, "dtype") else None

        weight = self._linear.weight
        if self._input_quantizer:
            x = self._input_quantizer(x)
        else:
            if linear_dtype:
                x = self._linear.cast(x, linear_dtype)
            else:
                x = self._linear.cast(x, self._linear.weight.dtype)
        if self._weight_quantizer:
            weight = self._weight_quantizer(weight)
        elif linear_dtype:
            weight = self._linear.cast(self._linear.weight, linear_dtype)

        x = self._linear.matmul(x, weight)
        if self._linear.has_bias:
            if hasattr(self._linear, "dtype"):
                bias = self._linear.cast(self._linear.bias, self._linear.dtype)
            else:
                bias = self._linear.cast(self._linear.bias, x.dtype)
            x = self._linear.bias_add(x, bias)
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output
