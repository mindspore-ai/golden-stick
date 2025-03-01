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

from mindspore import ops as msops
from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer

from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import RowParallelLinear, ColumnParallelLinear

from mindspore_gs.ptq.ptq_config import PTQMode, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_quant_func import quant_tensor
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq.hal import (QuantParam, AllQuantMatmul, ParallelType, KernelType,
                                      OutlierSuppressionPlusMatmulForDeploy)
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
        Quantizer.reg_layer_map(ColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(RowParallelLinear, AllQuantLinearCell, A8W8Checker())

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, network_helper, **kwargs):
        super().__init__(linear_name, linear, context, cfg, network_helper, **kwargs)

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
        quant, qmm, bias = AllQuantMatmul.create(layer_name, linear, parallel_type, q_weight, x_qparam, w_qparam,
                                                 is_deploy, cfg.tp_size, compute_type, KernelType.INTERNAL)
        if not is_deploy:
            logger.debug(f"AllQuantLinearInferCell: x_qparam of Layer({parallel_type}:{layer_name}) is {x_qparam}")
            logger.debug(f"AllQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"AllQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}, {q_weight.asnumpy()}}}")
            logger.debug(f"AllQuantLinearInferCell: bias with correction of Layer("
                         f"{parallel_type}:{layer_name}) is {{{bias.shape}, {bias.dtype}, {bias.asnumpy()}}}")
        self._set_act_quant(quant)
        if linear.has_bias is False:
            self.layer.has_bias = True
            self.layer.has_quant_bias = False
            self.layer.bias_add = msops.Add()
        if isinstance(self.layer.matmul, OutlierSuppressionPlusMatmulForDeploy):
            self.layer.has_bias = False
            self.layer.has_quant_bias = True
            qmm.quant_bias = bias
        else:
            self.layer.bias = bias
        self.layer.matmul = qmm
        self.layer.weight = q_weight
