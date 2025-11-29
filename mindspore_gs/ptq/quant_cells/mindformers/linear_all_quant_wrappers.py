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
"""ptq wrapper cells for mindformers."""

from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer
from mindspore import ops as msops
from mindformers.modules.layers import Linear
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    ColumnParallelLinear as McoreColumnParallelLinear, RowParallelLinear as McoreRowParallelLinear)
from mindformers.parallel_core.inference.tensor_parallel.layers import QKVParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.layers import ReplicatedLinear
from mindformers.parallel_core.inference.tensor_parallel.layers import MergedColumnParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear
)

from mindspore_gs.ptq.ptq_config import PTQMode, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_functions.basic_quant_func import quant_tensor
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq.hal import (QuantParam, AllQuantMatmul, ParallelType, KernelType,
                                      OutlierSuppressionPlusSmoothMatmul)
from mindspore_gs.ptq.algo_modules import Quantizer
from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from mindspore_gs.ptq.utils import QuantType
from .parallel_minmax import get_min_max_op
from .linear_weight_quant_wrappers import WeightQuantLinearCell
from .linear_wrapper import LinearInferCell
from .mcore_linear_wrapper import McoreLinearInferCell


class AllQuantLinearCell(WeightQuantLinearCell):
    """QuantLinearCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class A8W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype == dtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TENSOR

        Quantizer.reg_layer_map(Linear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(McoreColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(McoreRowParallelLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(QKVParallelLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(ReplicatedLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(MergedColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(ColumnParallelGroupedLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(RowParallelGroupedLinear, AllQuantLinearCell, A8W8Checker())
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear)
            from research.deepseek3.infer.layers import ColumnParallelLinear as DSColumnParallelLinear
            from research.deepseek3.infer.layers import RowParallelLinear as DSRowParallelLinear
            Quantizer.reg_layer_map(DSColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(DSRowParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(ColumnParallelGroupLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(RowParallelGroupLinear, AllQuantLinearCell, A8W8Checker())
        except ImportError:
            pass
        try:
            from research.llama3_1.infer.layers import ColumnParallelLinear as LlamaColumnParallelLinear
            from research.llama3_1.infer.layers import RowParallelLinear as LlamaRowParallelLinear
            Quantizer.reg_layer_map(LlamaColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(LlamaRowParallelLinear, AllQuantLinearCell, A8W8Checker())
        except ImportError:
            pass
        try:
            from research.telechat2.infer.layers import ColumnParallelLinear as TC2ColumnParallelLinear
            from research.telechat2.infer.layers import RowParallelLinear as TC2RowParallelLinear
            Quantizer.reg_layer_map(TC2ColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(TC2RowParallelLinear, AllQuantLinearCell, A8W8Checker())
        except ImportError:
            pass

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)

        is_rowparallel = self.parallel_type == ParallelType.ROW_PARALLEL
        self.x_quant_max, self.x_quant_min = get_min_max_op(cfg.tp_size, is_rowparallel)

        self.x_scale = Parameter(initializer('ones', (1,), dtype=dtype.float64))
        self.x_zp = Parameter(initializer('zeros', (1,), dtype=dtype.float64))

    def _quant_info(self):
        res = super()._quant_info()
        if self.cfg.act_quant_dtype == dtype.int8:
            return f'{res}-A8-{str(self.cfg.act_quant_granularity)}'
        raise RuntimeError(f"Unexpected act_quant_dtype: {self.cfg.act_quant_dtype}.")

    def quant(self):
        """quant"""
        # quant weight
        super().quant()
        # quant activation
        x_scale, x_zp, _ = quant_tensor(self.cat_samples, self.x_quant_min, self.x_quant_max,
                                        self.cfg.act_narrow_range, self.cfg.act_symmetric,
                                        self.cfg.act_quant_granularity == QuantGranularity.PER_GROUP,
                                        self.cfg.group_size,
                                        self.cfg.act_quant_dtype, -1, False)
        self.x_scale.set_data(Tensor(x_scale, dtype=dtype.float64))
        self.x_zp.set_data(Tensor(x_zp, dtype=dtype.float64))
        self.cfg.dumper.dump_data(self.layer_name, "|activation_params|input0_activation_inputs", self.cat_samples)
        self.cfg.dumper.dump_data(self.layer_name, "|activation_params|output0_activation_scale", self.x_scale)
        self.cfg.dumper.dump_data(self.layer_name, "|activation_params|output1_activation_zp", self.x_zp)

    def deploy(self):
        if self.is_mcorelinear:
            return AllQuantMcoreLinearInferCell(self._layer_name, self.layer, self.context, self.cfg,
                                                self.q_weight, QuantParam(self.x_scale, self.x_zp),
                                                QuantParam(self.w_scale, self.w_zp), self.compute_type,
                                                self.parallel_type)

        return AllQuantLinearInferCell(self._layer_name, self.layer, self.context, self.cfg,
                                       self.q_weight, QuantParam(self.x_scale, self.x_zp),
                                       QuantParam(self.w_scale, self.w_zp), self.compute_type,
                                       self.parallel_type)


class AllQuantLinearInferCell(LinearInferCell):
    """AllQuantLinearInferCell"""
    # pylint: disable=unused-argument
    def __init__(self, layer_name, linear: Linear, context: InnerPTQConfig, cfg: InnerPTQConfig, q_weight,
                 x_qparam: QuantParam, w_qparam: QuantParam, compute_type, parallel_type: ParallelType):
        super().__init__(linear, parallel_type)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        use_aclnn_quant = any(opname in layer_name for opname in cfg.aclnn_quant_list)
        bias_osp = None
        if isinstance(self.layer.matmul, OutlierSuppressionPlusSmoothMatmul):
            origin_weight = msops.mul(self._layer.weight, linear.matmul.smooth_scale)
            bias_osp = msops.matmul(
                msops.expand_dims(-linear.matmul.beta_osp, 0),
                (
                    origin_weight.astype("float32").transpose()
                    if self.layer.transpose_b
                    else self._layer.weight.astype("float32")
                ),
            )
            bias_osp = bias_osp.squeeze()
        quant, qmm = AllQuantMatmul.create(layer_name, linear, linear.transpose_b, parallel_type, q_weight, x_qparam,
                                           w_qparam, is_deploy, cfg.tp_size, compute_type,
                                           KernelType.ACLNN if use_aclnn_quant else KernelType.INTERNAL, bias_osp)
        if not is_deploy:
            logger.debug(f"AllQuantLinearInferCell: x_qparam of Layer({parallel_type}:{layer_name}) is {x_qparam}")
            logger.debug(f"AllQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"AllQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}}}")
        self._set_act_quant(quant)
        self.layer.matmul = qmm
        self.layer.weight = q_weight


class AllQuantMcoreLinearInferCell(McoreLinearInferCell):
    """AllQuantLinearInferCell"""

    def __init__(self, layer_name, linear: Linear, context: InnerPTQConfig, cfg: InnerPTQConfig, q_weight,
                 x_qparam: QuantParam, w_qparam: QuantParam, compute_type, parallel_type: ParallelType):
        super().__init__(linear, parallel_type)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        use_aclnn_quant = any(opname in layer_name for opname in cfg.aclnn_quant_list)
        bias_osp = None
        if isinstance(self.layer.quant_method.matmul, OutlierSuppressionPlusSmoothMatmul):
            origin_weight = msops.mul(self._layer.weight, linear.quant_method.matmul.smooth_scale)
            bias_osp = msops.matmul(
                msops.expand_dims(-linear.quant_method.matmul.beta_osp, 0),
                (
                    origin_weight.astype("float32").transpose()
                    if self._transpose_b()
                    else self._layer.weight.astype("float32")
                ),
            )
            bias_osp = bias_osp.squeeze()
        quant, qmm = AllQuantMatmul.create(layer_name, linear, self._transpose_b(), parallel_type, q_weight,
                                           x_qparam, w_qparam, is_deploy, cfg.tp_size, compute_type,
                                           KernelType.ACLNN if use_aclnn_quant else KernelType.INTERNAL, bias_osp,
                                           context.experimental)
        if not is_deploy:
            logger.debug(f"AllQuantLinearInferCell: x_qparam of Layer({parallel_type}:{layer_name}) is {x_qparam}")
            logger.debug(f"AllQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"AllQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}}}")
        self._set_act_quant(quant)
        del self.layer.weight
        self.layer.weight = None
        self.weight = q_weight
        self.smooth_scale = Parameter(Tensor(quant.smooth_scale.asnumpy()))
        self.weight_scale = Parameter(w_qparam.scale.astype(compute_type))
        self.weight_offset = Parameter(w_qparam.zero_point.astype(dtype.int32))
        self.deq_scale = qmm.dequant_scale
        self.quant_bias = qmm.quant_bias
        self.input_scale = self.quant_op.input_scale
        self.input_offset = Parameter(self.quant_op.input_zp)
        self.quant_op = None
        self.has_bias = self.layer.has_bias
        if self.has_bias:
            self.bias = self.layer.bias
            self.layer.bias = None

    def quant_type_dict(self):
        """quant_type_dict"""
        quant_type = {
            self.smooth_scale.name: QuantType.W8A8.value,
            self.weight_scale.name: QuantType.W8A8.value,
            self.weight_offset.name: QuantType.W8A8.value,
            self.weight.name: QuantType.W8A8.value,
            self.input_scale.name: QuantType.W8A8.value,
            self.input_offset.name: QuantType.W8A8.value,
            self.deq_scale.name: QuantType.W8A8.value,
            self.quant_bias.name: QuantType.W8A8.value,
        }
        if self.has_bias:
            quant_type.update({self.bias.name: QuantType.W8A8.value})
        return quant_type
