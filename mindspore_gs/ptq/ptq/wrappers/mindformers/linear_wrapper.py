# Copyright 2024 Huawei Technologies Co., Ltd
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
"""mindformers linear wrapper cell."""
import abc
from typing import Optional

from mindspore import ops as msops
from mindspore.ops import functional as F
from mindspore.ops.auto_generate import GroupedMatmulV4
from mindspore.nn import Cell
from mindformers.modules.layers import Linear
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.ptq.hal import ParallelType, QuantWithSmooth, DynamicQuantCell


class WrapperLinearCell(WrapperCell, abc.ABC):
    """WrapperCell"""
    def add_hook(self):
        class HookMatMul(msops.MatMul):
            def __call__(self, *args, **kwargs):
                x = args[0]
                self.samples.append(msops.squeeze(x))
                return msops.MatMul.__call__(self, *args, **kwargs)

        class HookGroupedMatMul(GroupedMatmulV4):
            def __call__(self, *args, **kwargs):
                x = args[0][0]
                self.samples.append(msops.squeeze(x))
                return GroupedMatmulV4.__call__(self, *args, **kwargs)

        # what Python really does to __call__:
        # type(a).__call__(a)
        # as such, if I want to override the __call__ method, I must override the __call__ of a class
        # but if I don't want to affect behaviour of other instances of the same class,
        # I need to create a new class with the overridden __call__ method.
        if isinstance(self.layer.matmul, msops.MatMul):
            self.layer.matmul.__class__ = HookMatMul
            self.layer.matmul.layer_name = self.layer_name
            self.layer.matmul.samples = self.samples
        elif isinstance(self.layer.matmul, GroupedMatmulV4):
            self.layer.matmul.__class__ = HookGroupedMatMul
            self.layer.matmul.layer_name = self.layer_name
            self.layer.matmul.samples = self.samples
        elif hasattr(self.layer.matmul, 'mm') and isinstance(self.layer.matmul.mm, msops.MatMul):
            self.layer.matmul.mm.__class__ = HookMatMul
            self.layer.matmul.mm.layer_name = self.layer_name
            self.layer.matmul.mm.samples = self.samples
        elif hasattr(self.layer.matmul, 'mm') and isinstance(self.layer.matmul.mm, GroupedMatmulV4):
            self.layer.matmul.mm.__class__ = HookGroupedMatMul
            self.layer.matmul.mm.layer_name = self.layer_name
            self.layer.matmul.mm.samples = self.samples
        else:
            raise RuntimeError(f"Unsupported matmul type for hook: {type(self.layer.matmul)}")

    def remove_hook(self):
        if isinstance(self.layer.matmul, msops.MatMul):
            self.layer.matmul.__class__ = msops.MatMul
        elif isinstance(self.layer.matmul, GroupedMatmulV4):
            self.layer.matmul.__class__ = GroupedMatmulV4
        elif hasattr(self.layer.matmul, 'mm') and isinstance(self.layer.matmul.mm, msops.MatMul):
            self.layer.matmul.mm.__class__ = msops.MatMul
        elif hasattr(self.layer.matmul, 'mm') and isinstance(self.layer.matmul.mm, GroupedMatmulV4):
            self.layer.matmul.mm.__class__ = GroupedMatmulV4
        else:
            raise RuntimeError(f"Unsupported matmul type for hook: {type(self.layer.matmul)}")

    @abc.abstractmethod
    def deploy(self):
        """deploy"""
        raise NotImplementedError


class LinearInferCell(Cell):
    """DeployLinearCell"""

    def __init__(self, linear: Linear, parallel_type: ParallelType):
        super().__init__()
        self._layer = linear
        self.parallel_type = parallel_type

        self.has_act_quant = False
        self.quant_op: Optional[QuantWithSmooth] = None
        self.has_act_dynamic_quant = False
        self.dyn_quant_op: Optional[DynamicQuantCell] = None


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

    def linear_forward(self, x, group_list=None):
        """Forward process, x should be a tensor"""
        ori_dtype = F.dtype(x)
        x = self._layer.cast(x, self._layer.dtype)
        if self.has_act_quant:
            x = self.quant_op(x)
        if self.has_act_dynamic_quant:
            x, x_scale = self.dyn_quant_op(x)
            x_scale = self._layer.reshape(x_scale, (-1,))
        out_shape = self._layer.shape(x)[:-1] + (self._layer.out_channels,)
        x = self._layer.reshape(x, (-1, self._layer.in_channels))
        if self._layer.expert_flag and not self._layer.use_gmm:
            if self._layer.use_expert_group_size is True:
                x = self._layer.reshape(x, (-1, self._layer.expert_num, self._layer.expert_group_size,
                                            self._layer.in_channels))
            else:
                x = self._layer.reshape(x, (self._layer.outer_batch, self._layer.expert_num, -1,
                                            self._layer.in_channels))
        # apply gmm to the inference of moe structural models when use_past=True.
        if self._layer.use_gmm:
            raise RuntimeError(f"{type(Linear)} not support GroupedMatmul.")
        if self.has_act_dynamic_quant:
            x = self._layer.matmul(x, self.layer.weight, None, x_scale)
        else:
            x = self._layer.matmul(x, self.layer.weight)
        if self._layer.has_bias:
            x = self._layer.bias_add(x, self._layer.cast(self._layer.bias, self._layer.dtype))
        if self._layer.activation_flag:
            x = self._layer.activation(x)
        x = F.cast(x, ori_dtype)
        output = self._layer.reshape(x, out_shape)
        return output

    def col_linear_forward(self, input_parallel, weight=None, group_list=None):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if weight is None and self._layer.skip_weight_param_allocation:
            raise ValueError("For ColumnParallelLinear, when skip_weight_param_allocation=True,"
                             " weight should be passed to construct(), but got None.")

        origin_dtype = F.dtype(input_parallel)
        input_parallel = self._layer.cast(input_parallel, self._layer.compute_dtype)

        if self._layer.sequence_parallel:
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
            input_parallel = self._layer.gather_from_sp_region(input_parallel)
            input_parallel = input_parallel.swapaxes(0, 1).contiguous()
        if self.has_act_quant:
            input_parallel = self.quant_op(input_parallel)
        if self.has_act_dynamic_quant:
            input_parallel, x_scale = self.dyn_quant_op(input_parallel)
            x_scale = self._layer.reshape(x_scale, (-1,))
        output_shape = self._layer.shape(input_parallel)[:-1] + (self._layer.output_size_per_partition,)
        input_parallel = self._layer.reshape(input_parallel, (-1, self._layer.input_size))
        if self._layer.is_expert and self._layer.expert_num > 1:
            if self.has_act_dynamic_quant:
                output_parallel = self._layer.matmul(input_parallel, self.layer.weight, group_list, x_scale)
            else:
                output_parallel = self._layer.matmul(input_parallel, self._layer.weight, group_list)
        else:
            if self.has_act_dynamic_quant:
                output_parallel = self._layer.matmul(input_parallel, self.layer.weight, None, x_scale)
            else:
                output_parallel = self._layer.matmul(input_parallel, self._layer.weight)
        if self._layer.has_bias:
            output_parallel = self._layer.bias_add(
                output_parallel, self._layer.cast(self._layer.bias, self._layer.compute_dtype)
            )
        output_parallel = self._layer.cast(output_parallel, origin_dtype)
        output_parallel = self._layer.reshape(output_parallel, output_shape)

        if self._layer.gather_output:
            output = self._layer.gather_from_mp_region(output_parallel)
        else:
            output = output_parallel
        return output

    def row_linear_forward(self, input_, group_list=None):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self._layer.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = self._layer.scatter_to_mp_region(input_)

        origin_dtype = F.dtype(input_parallel)
        input_parallel = self._layer.cast(input_parallel, self._layer.compute_dtype)
        if self.has_act_quant:
            input_parallel = self.quant_op(input_parallel)
        if self.has_act_dynamic_quant:
            input_parallel, x_scale = self.dyn_quant_op(input_parallel)
            x_scale = self._layer.reshape(x_scale, (-1,))
        output_shape = self._layer.shape(input_parallel)[:-1] + (self._layer.output_size,)
        input_parallel = self._layer.reshape(input_parallel, (-1, self._layer.input_size_per_partition))
        if self._layer.is_expert and self._layer.expert_num > 1:
            if self.has_act_dynamic_quant:
                output_parallel = self._layer.matmul(input_parallel, self.layer.weight, group_list, x_scale)
            else:
                output_parallel = self._layer.matmul(input_parallel, self._layer.weight, group_list)
        else:
            if self.has_act_dynamic_quant:
                output_parallel = self._layer.matmul(input_parallel, self.layer.weight, None, x_scale)
            else:
                output_parallel = self._layer.matmul(input_parallel, self._layer.weight)

        if self._layer.sequence_parallel:
            output_parallel = output_parallel.swapaxes(0, 1).contiguous()
            output = self._layer.reduce_scatter_to_sp_region(output_parallel)
            output = output.swapaxes(0, 1).contiguous()
        else:
            if (hasattr(self._layer, 'delay_allreduce') and self._layer.delay_allreduce) or \
                    (hasattr(self._layer, 'moe_delay_allreduce') and self._layer.moe_delay_allreduce):
                output = output_parallel
            else:
                output = self._layer.reduce_from_mp_region(output_parallel)

        if self._layer.has_bias:
            output = self._layer.bias_add(output, self._layer.cast(self._layer.bias, self._layer.compute_dtype))
        output = self._layer.cast(output, origin_dtype)
        output = self._layer.reshape(output, output_shape)
        return output

    def construct(self, x, group_list=None):
        """linear deploy construct"""
        if self.parallel_type == ParallelType.NO_PARALLEL:
            return self.linear_forward(x, group_list=group_list)
        if self.parallel_type == ParallelType.COL_PARALLEL:
            x = self.col_linear_forward(x, group_list=group_list)
        if self.parallel_type == ParallelType.ROW_PARALLEL:
            x = self.row_linear_forward(x, group_list=group_list)
        return x

    def sharded_state_dict(self, **kwargs):
        """provide the sharded state dict based on the config"""
        state_dict = {}
        if self.parallel_type == ParallelType.NO_PARALLEL:
            return {}
        tensor_parallel_num = self.layer.tensor_parallel_group_size
        if self.parallel_type == ParallelType.COL_PARALLEL:
            if self._layer.is_expert and self._layer.expert_num > 1:
                w_shard = (1, tensor_parallel_num, 1) if self.layer.transpose_b else (1, 1, tensor_parallel_num)
            else:
                w_shard = (tensor_parallel_num, 1) if self.layer.transpose_b else (1, tensor_parallel_num)
            if self.layer.has_bias:
                state_dict[self.layer.bias.name] = {'shape': self.layer.bias.shape,
                                                    'shard': (tensor_parallel_num,)}
        elif self.parallel_type == ParallelType.ROW_PARALLEL:
            if self._layer.is_expert and self._layer.expert_num > 1:
                w_shard = (1, 1, tensor_parallel_num) if self.layer.transpose_b else (1, tensor_parallel_num, 1)
            else:
                w_shard = (1, tensor_parallel_num) if self.layer.transpose_b else (tensor_parallel_num, 1)
            if self.layer.has_bias:
                state_dict[self.layer.bias.name] = {'shape': self.layer.bias.shape, 'shard': (1,)}
        else:
            return {}
        state_dict[self.layer.weight.name] = {'shape': self.layer.weight.shape, 'shard': w_shard}
        if self.quant_op:
            state_dict.update(self.quant_op.param_shard_state(tensor_parallel_num, **kwargs))
        if hasattr(self.layer.matmul, "param_shard_state"):
            state_dict.update(self.layer.matmul.param_shard_state(tensor_parallel_num, self.parallel_type))
        return state_dict
