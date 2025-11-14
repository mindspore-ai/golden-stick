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
from mindspore import ops as msops
from mindspore.common.initializer import initializer
from mindformers.parallel_core.inference.tensor_parallel.layers import LinearBase
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear)
from mindformers.parallel_core.inference.tensor_parallel.layers import LinearMethodBase
from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from mindspore_gs.ptq.algo_modules import Quantizer
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq_config import QuantGranularity, OutliersSuppressionType
from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell
from .fake_quant_base import (
    DynamicFakeQuant,
    GMMDeQuant,
    DeQuant,
    FakeQuantGroupLinearCell)


class FakeQuantW8A8DynamicLinearMethod(LinearMethodBase):
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


class FakeQuantW8A8DynamicGroupWrapper(QuantCell):
    """FakeQuantWrapper"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class FakeQuantW8A8DynamicChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == msdtype.int8 and config.act_quant_dtype == msdtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TOKEN

        Quantizer.reg_fake_quant_layer_map(ColumnParallelGroupedLinear, FakeQuantW8A8DynamicGroupWrapper,
                                           FakeQuantW8A8DynamicChecker())
        Quantizer.reg_fake_quant_layer_map(RowParallelGroupedLinear, FakeQuantW8A8DynamicGroupWrapper,
                                           FakeQuantW8A8DynamicChecker())

    def _quant_info(self) -> str:
        return 'FakeQuant'

    def add_hook(self, experimental=False):
        pass

    def remove_hook(self, experimental=False):
        pass

    def deploy(self):
        return FakeQuantW8A8DynamicGroupLinearCell(self.layer_name, self.layer, self.context, self.cfg)


class FakeQuantW8A8DynamicGroupLinearCell(FakeQuantGroupLinearCell):
    """FakeQuantLinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: LinearBase, context, cfg: InnerPTQConfig):
        super().__init__(layer_name, linear, context, cfg)
        self.is_act_quant = cfg.act_quant_dtype == msdtype.int8
        self.has_smooth = cfg.outliers_suppression in (OutliersSuppressionType.SMOOTH,
                                                       OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE)

        self.weight_scale = Parameter(initializer(
            "ones", (self.num_local_experts, self.output_size_per_partition), self.compute_dtype))
        self.weight_offset = Parameter(initializer(
            "zeros", (self.num_local_experts, self.output_size_per_partition), msdtype.int32))
        self.weight = Parameter(initializer("zeros", linear.weight.shape, msdtype.int8))
        self.quant_method = FakeQuantW8A8DynamicLinearMethod(layer_name, linear.quant_method, self.compute_dtype,
                                                             self.is_act_quant)


class FakeQuantW4A8DynamicLinearMethod(LinearMethodBase):
    """Linear method without quantization."""
    def __init__(self, layer_name, quant_method: LinearMethodBase, output_dtype, is_act_quant=True, group_size=256):
        super().__init__()
        self.layer_name = layer_name
        self.quant_method = quant_method
        self.is_act_quant = is_act_quant
        self.group_size = group_size
        self.fake_quant = DynamicFakeQuant(msdtype.int8, output_dtype)
        self.de_quant = DeQuant(output_dtype)

    def create_weights(self, layer: nn.Cell, input_size_per_partition: int,
                       output_partition_sizes, params_dtype, **extra_weight_attrs):
        raise NotImplementedError

    def apply(self, layer: nn.Cell, x: Tensor, weight: Tensor, bias: Parameter = None, group_list=None):
        """apply"""
        if self.is_act_quant:
            x = self.fake_quant(x)
        weight_scale = msops.repeat_elements(layer.weight_scale, rep=self.group_size, axis=1)
        weight_offset = msops.repeat_elements(layer.weight_offset, rep=self.group_size, axis=1)
        weight = self.de_quant(weight, weight_scale, weight_offset)
        return self.quant_method.apply(layer, x, weight, bias, group_list)


class FakeQuantW4A8DynamicGroupWrapper(QuantCell):
    """FakeQuantWrapper"""
    @staticmethod
    def reg_self():
        """reg_self"""
        class FakeQuantW4A8DynamicChecker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == msdtype.qint4x2 and config.act_quant_dtype == msdtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TOKEN and \
                       config.weight_quant_granularity is QuantGranularity.PER_GROUP

        Quantizer.reg_fake_quant_layer_map(ColumnParallelGroupedLinear, FakeQuantW4A8DynamicGroupWrapper,
                                           FakeQuantW4A8DynamicChecker())
        Quantizer.reg_fake_quant_layer_map(RowParallelGroupedLinear, FakeQuantW4A8DynamicGroupWrapper,
                                           FakeQuantW4A8DynamicChecker())

    def _quant_info(self) -> str:
        return 'FakeQuant'

    def add_hook(self, experimental=False):
        pass

    def remove_hook(self, experimental=False):
        pass

    def deploy(self):
        return FakeQuantW4A8DynamicGroupLinearCell(self.layer_name, self.layer, self.context, self.cfg)


class FakeQuantW4A8DynamicGroupLinearCell(FakeQuantGroupLinearCell):
    """FakeQuantLinearCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: LinearBase, context, cfg: InnerPTQConfig):
        super().__init__(layer_name, linear, context, cfg)
        self.is_act_quant = cfg.act_quant_dtype == msdtype.int8

        self.group_size = self.cfg.group_size
        self.is_pre_group = self.group_size > 0
        if self.is_pre_group:
            scale_offset_shape = (self.num_local_experts,
                                  self.ic // self.group_size,
                                  self.output_size_per_partition)
        else:
            scale_offset_shape = (self.num_local_experts,
                                  self.output_size_per_partition)

        self.weight_scale = Parameter(initializer("ones", scale_offset_shape, self.compute_dtype))
        self.weight_offset = Parameter(initializer("zeros", scale_offset_shape, msdtype.int32))
        self.weight = Parameter(initializer("zeros", linear.weight.shape, msdtype.int8))
        self.quant_method = FakeQuantW4A8DynamicLinearMethod(layer_name, linear.quant_method, self.compute_dtype,
                                                             self.is_act_quant, self.group_size)
