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


from mindspore import nn, Parameter, dtype as msdtype, Tensor
from mindspore.common.initializer import initializer
from mindformers.modules.layers import Linear
from mindformers.parallel_core.inference.tensor_parallel.layers import LinearBase
from mindformers.parallel_core.inference.tensor_parallel.gemm_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear
)
from mindformers.parallel_core.inference.tensor_parallel.layers import LinearMethodBase
from mindformers.parallel_core.inference.tensor_parallel.mappings import (gather_from_model_parallel_region,
                                                                          scatter_to_model_parallel_region)
from mindspore_gs.ptq.ptq.hal import ParallelType
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq_config import QuantGranularity, OutliersSuppressionType
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.ptq.wrappers.mindformers.fake_quant_base import DynamicFakeQuant


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


class FakeQuantLinearMethod(LinearMethodBase):
    """Linear method without quantization."""
    def __init__(self, layer_name, quant_method: LinearMethodBase, output_dtype, is_act_quant=True):
        super().__init__()
        self.layer_name = layer_name
        self.quant_method = quant_method
        self.is_act_quant = is_act_quant
        self.fake_quant = DynamicFakeQuant(msdtype.int8, output_dtype)
        self.de_quant = GMMDeQuant(output_dtype)

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes, params_dtype, **extra_weight_attrs):
        raise NotImplementedError

    def apply(self, layer: nn.Cell, x: Tensor, weight: Tensor, bias: Parameter = None, group_list=None):
        """apply"""
        if self.is_act_quant:
            x = self.fake_quant(x)
        weight = self.de_quant(weight, layer.weight_scale, layer.weight_offset)
        return self.quant_method.apply(layer, x, weight, bias, group_list)


class FakeQuantGroupWrapper(WrapperCell):
    """FakeQuantWrapper"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class FakeQuantChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == msdtype.int8 and config.act_quant_dtype == msdtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TOKEN

        Quantizer.reg_fake_quant_layer_map(ColumnParallelGroupedLinear, FakeQuantGroupWrapper, FakeQuantChecker())
        Quantizer.reg_fake_quant_layer_map(RowParallelGroupedLinear, FakeQuantGroupWrapper, FakeQuantChecker())

    def _quant_info(self) -> str:
        return 'FakeQuant'

    def add_hook(self, experimental=False):
        pass

    def remove_hook(self, experimental=False):
        pass

    def deploy(self):
        return FakeQuantGroupLinearCell(self.layer_name, self.layer, self.context, self.cfg)


class FakeQuantGroupLinearCell(LinearBase):
    """FakeQuantLinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: LinearBase, context, cfg: InnerPTQConfig):
        super().__init__(linear.input_size, linear.output_size)
        self.layer_name = layer_name
        self.is_act_quant = cfg.act_quant_dtype == msdtype.int8
        self.has_smooth = cfg.outliers_suppression in (OutliersSuppressionType.SMOOTH,
                                                       OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE)
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

        self.weight_scale = Parameter(initializer(
            "ones", (self.num_local_experts, self.output_size_per_partition), self.compute_dtype))
        self.weight_offset = Parameter(initializer(
            "zeros", (self.num_local_experts, self.output_size_per_partition), msdtype.int32))
        self.weight = Parameter(initializer("zeros", linear.weight.shape, msdtype.int8))
        self.quant_method = FakeQuantLinearMethod(layer_name, linear.quant_method, self.compute_dtype,
                                                  self.is_act_quant)

    # pylint: disable=unused-argument
    def construct(self, x, weight=None, group_list=None):
        """linear deploy construct"""
        if self.parallel_type == ParallelType.NO_PARALLEL:
            raise RuntimeError(f"Normal Linear is not supplied by mcore")
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
