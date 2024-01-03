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

from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from .linear import Linear


class LinearQuant(QuantCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy: LayerPolicy):
        super(LinearQuant, self).__init__(linear, policy)
        self._linear = linear
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        input_fq_args = {}
        weight_perchannel_args = PerChannelArgs(self._linear.out_channels, self._weight_axis)
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
        self._input_quantizer.float_min.data.name = prex + "_input_float_min"
        self._input_quantizer.float_max.data.name = prex + "_input_float_max"
        self._weight_quantizer.float_min.data.name = prex + "_weight_float_min"
        self._weight_quantizer.float_max.data.name = prex + "_weight_float_max"

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    def convert(self):
        self._input_quantizer = self._input_quantizer.convert_to_ascend()
        self._weight_quantizer = self._weight_quantizer.convert_to_ascend()

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
        if hasattr(self._linear, "dtype"):
            weight = self._linear.cast(self._linear.weight, self._linear.dtype)
            x = self._linear.cast(x, self._linear.dtype)
        else:
            weight = self._linear.weight
            x = self._linear.cast(x, self._linear.weight.dtype)

        x = self._input_quantizer(x)
        weight = self._weight_quantizer(weight)

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
