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

from mindspore_gs.ptq.ptq_config import PTQMode, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_quant_func import quant_tensor
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq.hal import (QuantParam, AllQuantMatmul, ParallelType, KernelType,
                                      OutlierSuppressionPlusSmoothMatmul)
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .parallel_minmax import get_min_max_op
from .linear_weight_quant_wrappers import WeightQuantLinearCell
from .linear_wrapper import LinearInferCell


class AllQuantLinearCell(WeightQuantLinearCell):
    """QuantLinearCell"""

    @staticmethod
    def reg_self():
        class A8W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype == dtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TENSOR

        Quantizer.reg_layer_map(Linear, AllQuantLinearCell, A8W8Checker())
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear,
                                                ColumnParallelLinearWorldRegion, RowParallelLinearWorldRegion)
            from research.deepseek3.infer.layers import ColumnParallelLinear as DSColumnParallelLinear
            from research.deepseek3.infer.layers import RowParallelLinear as DSRowParallelLinear
            from research.llama3_1.infer.layers import ColumnParallelLinear as LlamaColumnParallelLinear
            from research.llama3_1.infer.layers import RowParallelLinear as LlamaRowParallelLinear
            from research.telechat2.infer.layers import ColumnParallelLinear as TC2ColumnParallelLinear
            from research.telechat2.infer.layers import RowParallelLinear as TC2RowParallelLinear
            Quantizer.reg_layer_map(TC2ColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(TC2RowParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(LlamaColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(LlamaRowParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(DSColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(DSRowParallelLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(ColumnParallelGroupLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(RowParallelGroupLinear, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(ColumnParallelLinearWorldRegion, AllQuantLinearCell, A8W8Checker())
            Quantizer.reg_layer_map(RowParallelLinearWorldRegion, AllQuantLinearCell, A8W8Checker())
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
        return AllQuantLinearInferCell(self._layer_name, self.layer, self.cfg, self.q_weight,
                                       QuantParam(self.x_scale, self.x_zp), QuantParam(self.w_scale, self.w_zp),
                                       self.compute_type, self.parallel_type)


class AllQuantLinearInferCell(LinearInferCell):
    """AllQuantLinearInferCell"""

    def __init__(self, layer_name, linear: Linear, cfg: InnerPTQConfig, q_weight, x_qparam: QuantParam,
                 w_qparam: QuantParam, compute_type, parallel_type: ParallelType):
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
                    if self._layer.transpose_b
                    else self._layer.weight.astype("float32")
                ),
            )
            bias_osp = bias_osp.squeeze()
        quant, qmm = AllQuantMatmul.create(layer_name, linear, parallel_type, q_weight, x_qparam, w_qparam, is_deploy,
                                           cfg.tp_size, compute_type,
                                           KernelType.ACLNN if use_aclnn_quant else KernelType.INTERNAL, bias_osp)
        if not is_deploy:
            logger.debug(f"AllQuantLinearInferCell: x_qparam of Layer({parallel_type}:{layer_name}) is {x_qparam}")
            logger.debug(f"AllQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"AllQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}}}")
        self._set_act_quant(quant)
        self.layer.matmul = qmm
        self.layer.weight = q_weight
