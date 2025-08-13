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

from mindspore import nn, Parameter, dtype, Tensor
from mindspore.common.initializer import initializer
from mindspore import ops as msops
from mindformers.modules.layers import Linear
from mindformers.parallel_core.inference.tensor_parallel.layers import LinearBase, RowParallelLinear, ColumnParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.layers import LinearMethodBase
from mindformers.parallel_core.inference.tensor_parallel.mappings import (gather_from_model_parallel_region,
                                                                          scatter_to_model_parallel_region,
                                                                          reduce_from_model_parallel_region)
from mindspore_gs.ptq.ptq.hal import ParallelType


class Quant(nn.Cell):
    """Quant"""
    def __init__(self, dst_type=dtype.int8):
        super().__init__()
        self.dst_type = dst_type

    def construct(self, x, scale, offset):
        x = x / scale
        x = msops.round(x) + offset
        x = msops.clip(x, -128., 127.)
        return msops.cast(x, self.dst_type)


class SmoothQuant(nn.Cell):
    """SmoothQuant"""
    def __init__(self, dst_type=dtype.int8):
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
        return x


class FakeQuantLinearMethod(LinearMethodBase):
    """Linear method without quantization."""
    def __init__(self, layer_name, quant_method: LinearMethodBase, output_dtype, is_act_quant=True, has_smooth=True):
        super().__init__()
        self.layer_name = layer_name
        self.quant_method = quant_method
        self.is_act_quant = is_act_quant
        self.has_smooth = has_smooth
        if has_smooth:
            self.fake_quant = SmoothFakeQuant(dtype.int8, output_dtype)
        else:
            self.fake_quant = FakeQuant(dtype.int8, output_dtype)
        self.de_quant = DeQuant(output_dtype)

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes, params_dtype, **extra_weight_attrs):
        raise NotImplementedError

    def apply(self, layer: nn.Cell, x: Tensor, weight: Tensor, bias: Parameter = None):
        """apply"""
        if self.is_act_quant:
            if self.has_smooth:
                # FIXME hangangqiang2@huawei.com
                # Obtain the real input_scale because dequantization scale should not include smooth_scale;
                # In subsequent algorithm stages, saved weights should not merge both factors. Related logic in the hal
                # of golden-stick must be removed.
                input_scale = layer.input_scale * layer.smooth_scale
                x = self.fake_quant(x, layer.smooth_scale, input_scale, layer.input_offset)
            else:
                x = self.fake_quant(x, layer.input_scale, layer.input_offset)
        weight_scale = layer.weight_scale.reshape((-1, 1))
        weight_offset = layer.weight_offset.reshape((-1, 1))
        weight = self.de_quant(weight, weight_scale, weight_offset)
        return self.quant_method.apply(layer, x, weight, bias)


class FakeQuantLinearCell(LinearBase):
    """FakeQuantLinearCell"""

    def __init__(self, layer_name, linear: LinearBase, is_act_quant=True, has_smooth=True):
        super().__init__(linear.input_size, linear.output_size)
        self.layer_name = layer_name
        if isinstance(linear, Linear):
            self.parallel_type = ParallelType.NO_PARALLEL
        elif isinstance(linear, ColumnParallelLinear):
            self.input_size = linear.input_size
            ic = linear.input_size
            self.gather_output = linear.gather_output
            self.tp_group = linear.tp_group
            self.output_size_per_partition = sum(linear.output_partition_sizes)
            self.bias = linear.bias if linear.has_bias else None
            self.parallel_type = ParallelType.COL_PARALLEL
        elif isinstance(linear, RowParallelLinear):
            ic = linear.input_size_per_partition
            self.output_size_per_partition = linear.output_size_per_partition
            self.input_is_parallel = linear.input_is_parallel
            self.tp_group = linear.tp_group
            self.bias = None if self.tp_group.rank > 0 else linear.bias
            self.parallel_type = ParallelType.ROW_PARALLEL
        else:
            raise ValueError(f"Not supported linear: {linear}")
        self.compute_dtype = linear.compute_dtype

        self.input_scale = Parameter(initializer("ones", (ic,), self.compute_dtype))
        self.smooth_scale = Parameter(initializer("ones", (ic,), self.compute_dtype))
        self.input_offset = Parameter(initializer("zeros", (ic,), dtype.int32))
        self.weight_scale = Parameter(initializer("ones", (self.output_size_per_partition,), self.compute_dtype))
        self.weight_offset = Parameter(initializer("zeros", (self.output_size_per_partition,), dtype.int32))
        self.weight = Parameter(initializer("zeros", linear.weight.shape, dtype.int8))
        self.quant_method = FakeQuantLinearMethod(layer_name, linear.quant_method, self.compute_dtype, is_act_quant,
                                                  has_smooth)

    # pylint: disable=unused-argument
    def construct(self, x, weight=None):
        """linear deploy construct"""
        if self.parallel_type == ParallelType.NO_PARALLEL:
            raise RuntimeError(f"Normal Linear is not supplied by mcore")
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
