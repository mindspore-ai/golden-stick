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
"""ptq wrapper cells for mindformers."""


from mindspore import nn, dtype as msdtype
from mindspore import ops as msops
from mindformers.modules.layers import Linear
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    LinearBase,
    RowParallelLinear,
    ColumnParallelLinear,
    ReplicatedLinear,
)
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear)
from mindformers.parallel_core.inference.tensor_parallel.mappings import (
    gather_from_model_parallel_region,
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region)
from mindspore_gs.ptq.ptq.hal import ParallelType
from mindspore_gs.ptq.context import InnerPTQConfig


class Quant(nn.Cell):
    """Quant"""
    def __init__(self, dst_type=msdtype.int8):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale, offset):
        x = x / scale
        x = msops.round(x) + offset
        x = msops.clip(x, -128., 127.)
        return msops.cast(x, self.dst_type)


class SmoothQuant(nn.Cell):
    """SmoothQuant"""
    def __init__(self, dst_type=msdtype.int8):
        super().__init__()
        self.dst_type = dst_type
        self.quant = Quant(dst_type)

    def construct(self, x, smooth_scale, scale, offset):
        # FIXME hangangqiang2@huawei.com
        # Theoretically, weights should be multiplied by scale, while activations should be divided by scale.
        # However, during deployment, smooth_scale is inverted (reciprocal taken) after apply_scale_to_weight.
        # Since weights have already been multiplied by scale, activations must still be multiplied by scale here.
        x = x * smooth_scale
        return self.quant(x, scale, offset)


class DeQuant(nn.Cell):
    """DeQuant"""
    def __init__(self, dst_type):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale, offset):
        x = (x - offset) * scale
        return x.astype(self.dst_type)


class FakeQuant(nn.Cell):
    """FakeQuant"""
    def __init__(self, quant_dtype, dst_dtype):
        super().__init__()
        self.quant = Quant(quant_dtype)
        self.de_quant = DeQuant(dst_dtype)

    def construct(self, x, scale, offset):
        x = self.quant(x, scale, offset)
        x = self.de_quant(x, scale, offset)
        return x


class SmoothFakeQuant(nn.Cell):
    """SmoothFakeQuant"""
    def __init__(self, quant_dtype, dst_dtype):
        super().__init__()
        self.quant = SmoothQuant(quant_dtype)
        self.de_quant = DeQuant(dst_dtype)

    def construct(self, x, smooth_scale, scale, offset):
        x = self.quant(x, smooth_scale, scale, offset)
        x = self.de_quant(x, scale, offset)
        x = x / smooth_scale
        return x


class DynamicQuant(nn.Cell):
    """DynamicQuant"""
    def __init__(self, dst_type=msdtype.int8):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x):
        x_max = msops.max(x, axis=1)[0]
        x_min = msops.min(x, axis=1)[0]
        x_max = msops.maximum(msops.abs(x_max), msops.abs(x_min))
        scale = msops.mul(x_max, 2) / (127 + 128)
        scale = msops.reshape(scale, (-1, 1))
        x = x / scale
        x = msops.round(x)
        x = msops.clip(x, -128., 127.)
        return msops.cast(x, self.dst_type), scale


class DynamicDeQuant(nn.Cell):
    """DeQuant"""
    def __init__(self, dst_type):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale):
        x = x * scale
        return x.astype(self.dst_type)

class DynamicFakeQuant(nn.Cell):
    """DynamicFakeQuant"""
    def __init__(self, quant_dtype, dst_dtype=msdtype.int8):
        super().__init__()
        self.quant = DynamicQuant(quant_dtype)
        self.de_quant = DynamicDeQuant(dst_dtype)

    def construct(self, x):
        x, scale = self.quant(x)
        x = self.de_quant(x, scale)
        return x


class GMMDeQuant(nn.Cell):
    """DeQuant"""
    def __init__(self, dst_type):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale, offset):
        scale = scale.expand_dims(1)
        offset = offset.expand_dims(1)
        x = (x - offset) * scale
        return x.astype(self.dst_type)


class FakeQuantLinearCell(LinearBase):
    """FakeQuantLinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: LinearBase, context, cfg: InnerPTQConfig):
        super().__init__(linear.input_size, linear.output_size)
        self.context = context
        self.cfg = cfg
        self.layer_name = layer_name
        if isinstance(linear, ColumnParallelLinear):
            self.input_size = linear.input_size
            self.ic = linear.input_size
            self.gather_output = linear.gather_output
            self.tp_group = linear.tp_group
            self.output_size_per_partition = sum(linear.output_partition_sizes)
            self.bias = linear.bias if linear.has_bias else None
            self.parallel_type = ParallelType.COL_PARALLEL
        elif isinstance(linear, RowParallelLinear):
            self.ic = linear.input_size_per_partition
            self.output_size_per_partition = linear.output_size_per_partition
            self.input_is_parallel = linear.input_is_parallel
            self.tp_group = linear.tp_group
            self.bias = None if self.tp_group.rank > 0 else linear.bias
            self.parallel_type = ParallelType.ROW_PARALLEL
        elif isinstance(linear, ReplicatedLinear):
            self.ic = linear.input_size
            self.bias = linear.bias if linear.has_bias else None
            self.output_size_per_partition = linear.output_size[0]
            self.parallel_type = ParallelType.NO_PARALLEL
        else:
            if isinstance(linear, Linear):
                raise RuntimeError("Normal Linear is not supplied by mcore")
            raise ValueError(f"Not supported linear: {linear}")
        self.compute_dtype = linear.compute_dtype

    # pylint: disable=unused-argument
    def construct(self, x, weight=None):
        """linear deploy construct"""
        if self.parallel_type == ParallelType.NO_PARALLEL:
            x = self.replicate_linear_forward(x, weight)
        if self.parallel_type == ParallelType.COL_PARALLEL:
            x = self.col_linear_forward(x)
        if self.parallel_type == ParallelType.ROW_PARALLEL:
            x = self.row_linear_forward(x)
        return x

    def col_linear_forward(self, input_):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """
        output_parallel = self.quant_method.apply(self, input_, self.weight, self.bias)

        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel, self.tp_group)
        else:
            output = output_parallel
        return output

    def row_linear_forward(self, input_):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_, self.tp_group)
        output_parallel = self.quant_method.apply(self, input_parallel, self.weight, self.bias)
        output = reduce_from_model_parallel_region(output_parallel, self.tp_group)
        return output

    def replicate_linear_forward(self, input_, weight=None):
        """
        Forward of ReplicatedLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "For ReplicatedLinear, weight was not supplied to construct(), "
                    "and `skip_weight_param_allocation` is True."
                    )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape.
            experted_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != experted_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {experted_shape} as expected."
                )

        output = self.quant_method.apply(self, input_, weight, self.bias)
        return output
class FakeQuantGroupLinearCell(LinearBase):
    """FakeQuantLinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: LinearBase, context, cfg: InnerPTQConfig):
        super().__init__(linear.input_size, linear.output_size)
        self.context = context
        self.cfg = cfg
        self.layer_name = layer_name
        if isinstance(linear, Linear):
            self.parallel_type = ParallelType.NO_PARALLEL
        elif isinstance(linear, ColumnParallelGroupedLinear):
            self.input_size = linear.input_size
            self.ic = linear.input_size
            self.gather_output = linear.gather_output
            self.tp_group = linear.tp_group
            self.output_size_per_partition = linear.output_size_per_partition
            self.bias = linear.bias if linear.has_bias else None
            self.parallel_type = ParallelType.COL_PARALLEL
            self.num_local_experts = linear.num_local_experts
        elif isinstance(linear, RowParallelGroupedLinear):
            self.ic = linear.input_size_per_partition
            self.output_size_per_partition = linear.output_size
            self.input_is_parallel = linear.input_is_parallel
            self.tp_group = linear.tp_group
            self.bias = None if self.tp_group.rank > 0 else linear.bias
            self.parallel_type = ParallelType.ROW_PARALLEL
            self.num_local_experts = linear.num_local_experts
        else:
            raise ValueError(f"Not supported linear: {linear}")
        self.compute_dtype = linear.compute_dtype

    # pylint: disable=unused-argument
    def construct(self, x, weight=None, group_list=None):
        """linear deploy construct"""
        if self.parallel_type == ParallelType.NO_PARALLEL:
            raise RuntimeError("Normal Linear is not supplied by mcore")
        if self.parallel_type == ParallelType.COL_PARALLEL:
            x = self.col_group_linear_forward(x, self.weight, group_list)
        if self.parallel_type == ParallelType.ROW_PARALLEL:
            x = self.row_linear_forward(x, self.weight, group_list)
        return x

    def col_group_linear_forward(self, input_parallel, weight=None, group_list=None):
        """
        Forward of ColumnParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """
        output_parallel = self.quant_method.apply(self, input_parallel, weight, self.bias, group_list)
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel, self.tp_group)
        else:
            output = output_parallel
        return output

    def row_linear_forward(self, input_, weight=None, group_list=None):
        """
        Forward of RowParallelLinear.
        Performs a linear transformation considering various parallel modes and data type conversions.
        """

        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_, self.tp_group)

        output = self.quant_method.apply(self, input_parallel, weight, self.bias, group_list)
        return output
