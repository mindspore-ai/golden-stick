# Copyright 2025 Huawei Technologies Co., Ltd
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
"""mindformers mcore linear wrapper cell."""
from typing import Optional

from mindspore import mint, Parameter
from mindspore.nn import Cell
from mindformers.modules.layers import Linear
from mindformers.parallel_core.inference.tensor_parallel.mappings import (gather_from_model_parallel_region,
                                                                          reduce_from_model_parallel_region,
                                                                          scatter_to_model_parallel_region)
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear
)
from mindspore_gs.ptq.ptq.hal import ParallelType, QuantWithSmooth, DynamicQuantCell


class McoreLinearInferCell(Cell):
    """DeployLinearCell"""

    def __init__(self, linear: Linear, parallel_type: ParallelType):
        super().__init__()
        self._layer = linear
        self.parallel_type = parallel_type

        self.has_act_quant = False
        self.quant_op: Optional[QuantWithSmooth] = None
        self.has_act_dynamic_quant = False
        self.dyn_quant_op: Optional[DynamicQuantCell] = None

        self.is_gmm_mcore = isinstance(linear, (ColumnParallelGroupedLinear,
                                                RowParallelGroupedLinear))

    def _set_act_quant(self, quant_op: QuantWithSmooth):
        self.has_act_quant = True
        self.quant_op = quant_op

    def _set_act_dynamic_quant(self, quant_op: DynamicQuantCell):
        self.has_act_dynamic_quant = True
        self.dyn_quant_op = quant_op

    @property
    def layer(self):
        """layer"""
        return self._layer

    def _transpose_b(self):
        if self.is_gmm_mcore:
            return False
        return self.layer.transpose_b

    def _set_transpose_b_to_false(self):
        if self.is_gmm_mcore:
            return
        self.layer.transpose_b = False

    def col_linear_forward(self, input_, weight=None):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """
        if weight is None:
            if self._layer.weight is None:
                raise RuntimeError(
                    "For ColumnParallelLinear, weight was not supplied to construct(), "
                    "and `skip_weight_param_allocation` is True."
                    )
            weight = self._layer.weight
        else:
            # Check the weight passed in is the correct shape.
            experted_shape = (self._layer.output_size_per_partition, self._layer.input_size)
            if weight.shape != experted_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {experted_shape} as expected."
                )

        origin_dtype = input_.dtype
        output_shape = input_.shape[:-1] + (sum(self._layer.output_size_per_partition),)

        input_ = mint.reshape(input_, (-1, self._layer.input_size))
        input_ = self._layer.cast(input_, self._layer.compute_dtype)

        if self.has_act_quant:
            input_ = self.quant_op(input_)
        if self.has_act_dynamic_quant:
            input_, x_scale = self.dyn_quant_op(input_)
            x_scale = mint.reshape(x_scale, (-1,))

        if self.has_act_dynamic_quant:
            output_parallel = self._layer.quant_method.matmul(input_, weight, None, x_scale)
        else:
            output_parallel = self._layer.quant_method.matmul(input_, weight)

        if self._layer.has_bias and not self._layer.skip_bias_add:
            bias = self._layer.cast(self._layer.bias, self._layer.compute_dtype)
            output_parallel = mint.add(output_parallel, bias)

        output_parallel = mint.reshape(output_parallel, output_shape)
        output_parallel = self._layer.cast(output_parallel, origin_dtype)

        if self._layer.gather_output:
            output = gather_from_model_parallel_region(output_parallel, self._layer.tp_group)
        else:
            output = output_parallel
        return output

    def row_linear_forward(self, input_):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self._layer.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_, self._layer.tp_group)

        origin_dtype = input_parallel.dtype
        input_parallel = self._layer.cast(input_parallel, self._layer.compute_dtype)

        if self.has_act_quant:
            input_parallel = self.quant_op(input_parallel)
        if self.has_act_dynamic_quant:
            input_parallel, x_scale = self.dyn_quant_op(input_parallel)
            x_scale = mint.reshape(x_scale, (-1,))

        output_shape = input_parallel.shape[:-1] + (self._layer.output_size,)
        input_parallel = mint.reshape(input_parallel, (-1, self._layer.input_size_per_partition))

        if self.has_act_dynamic_quant:
            output_parallel = self._layer.quant_method.matmul(input_parallel, self._layer.weight, None, x_scale)
        else:
            output_parallel = self._layer.quant_method.matmul(input_parallel, self._layer.weight)
        output = reduce_from_model_parallel_region(output_parallel, self._layer.tp_group)

        if self._layer.has_bias and not self._layer.skip_bias_add:
            bias = self._layer.cast(self._layer.bias, self._layer.compute_dtype)
            output = mint.add(output, bias)

        output = mint.reshape(output, output_shape)
        output = self._layer.cast(output, origin_dtype)
        return output

    def construct(self, x):
        """linear deploy construct"""
        if self.parallel_type == ParallelType.NO_PARALLEL:
            raise RuntimeError("Normal Linear is not supplied by mcore")
        if self.parallel_type == ParallelType.COL_PARALLEL:
            x = self.col_linear_forward(x)
        if self.parallel_type == ParallelType.ROW_PARALLEL:
            x = self.row_linear_forward(x)
        return x

    def sharded_state_dict(self, **kwargs):
        """provide the sharded state dict based on the config"""
        state_dict = {}
        if self.parallel_type == ParallelType.NO_PARALLEL:
            return {}
        tensor_parallel_num = self.layer.tensor_parallel_group_size

        if self.parallel_type == ParallelType.COL_PARALLEL:
            w_shard = (tensor_parallel_num, 1) if self._transpose_b() else (1, tensor_parallel_num)
            if not self.layer.skip_weight_param_allocation:
                state_dict[self.layer.weight.name] = {'shape': self.layer.weight.shape,
                                                      'shard': w_shard}
            if self.layer.bias:
                state_dict[self.layer.bias.name] = {'shape': self.layer.bias.shape,
                                                    'shard': (tensor_parallel_num,)}
        elif self.parallel_type == ParallelType.ROW_PARALLEL:
            w_shard = (1, tensor_parallel_num) if self._transpose_b() else (tensor_parallel_num, 1)
            if self.is_gmm_mcore:
                w_shard = (1, 1, tensor_parallel_num) if self._transpose_b() \
                    else (1, tensor_parallel_num, 1)
            if self.layer.bias:
                state_dict[self.layer.bias.name] = {'shape': self.layer.bias.shape,
                                                    'shard': (1,)}
            state_dict[self.layer.weight.name] = {'shape': self.layer.weight.shape,
                                                  'shard': w_shard}
        else:
            return {}
        if self.quant_op:
            state_dict.update(self.quant_op.param_shard_state(tensor_parallel_num, **kwargs))
        if hasattr(self.layer.quant_method.matmul, "param_shard_state"):
            state_dict.update(self.layer.quant_method.matmul.param_shard_state(tensor_parallel_num, self.parallel_type))
        return state_dict


class McoreGroupLinearDeployer(Cell):
    """McoreGroupLinearDeployer"""

    def append_param(self, name: str, param: Parameter):
        setattr(self, name, param)
