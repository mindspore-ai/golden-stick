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
"""Linear cell from mf. Just for test."""

from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Tensor
import mindspore.common.dtype as mstype
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindspore.context import ParallelMode


class Linear(Cell):
    """Linear of mf, for test"""

    @cell_attr_register
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 expert_group_size=None,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if not (activation is None or issubclass(activation, nn.Cell)):
            raise TypeError(f"For Linear cell, the activation should nn.Cell type, but got {activation}.")
        if isinstance(weight_init, Tensor) and (weight_init.ndim != 2 or weight_init.shape[0] != out_channels or
                                                weight_init.shape[1] != in_channels):
            raise ValueError("The shape of parameter 'weight_init' is error, please check shape of 'weight_init'.")
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.expert_num = expert_num
        self.outer_batch = outer_batch
        self.expert_group_size = expert_group_size
        self.transpose_b = transpose_b
        if self.expert_num > 1:
            self.expert_flag = True
            self.weight = Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                    name="weight")
            self.matmul = P.BatchMatMul(transpose_b=transpose_b)
        else:
            self.expert_flag = False
            self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
            self.matmul = P.MatMul(transpose_b=transpose_b)
        self.use_expert_group_size = _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) \
                                     and not _is_sharding_propagation() and self.expert_flag is True
        if self.use_expert_group_size is True and self.expert_group_size is None:
            raise ValueError("'expert_group_size' should be configured as an integer in MoEConfig.")
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            if isinstance(bias_init, Tensor) and (bias_init.ndim != 1 or bias_init.shape[0] != out_channels):
                raise ValueError("The shape of parameter 'bias_init' is error, please check shape of 'bias_init'.")
            if self.expert_flag:
                self.bias = Parameter(initializer(bias_init,
                                                  [1, self.expert_num, 1, out_channels], param_init_type), name="bias")
            else:
                self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = P.Add()
        self.act_name = activation
        if activation and not callable(activation):
            raise ValueError("Input activation should be callable.")
        if activation:
            self.activation = activation()
        else:
            self.activation = None
        self.activation_flag = self.activation is not None
        self.dtype = compute_dtype
        self.cast = P.Cast()

    def construct(self, x):
        """Forward process, x should be a tensor"""
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        if self.expert_flag:
            if self.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self.expert_num, self.expert_group_size, self.in_channels))
            else:
                x = P.Reshape()(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = F.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        if self.activation_flag:
            x = self.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output
