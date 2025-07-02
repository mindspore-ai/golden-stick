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

from mindformers.modules.layers import Linear
from mindspore import dtype, ops
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import PTQMode, QuantGranularity, PrecisionRecovery
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq.hal import QuantParam, GptqDynamicQuantMatmul, ParallelType
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .linear_gptq_quant_wrappers import GptqWeightQuantLinearCell
from .linear_wrapper import LinearInferCell


class GptqDynamicQuantLinearCell(GptqWeightQuantLinearCell):
    """GptqDynamicQuantLinearCell"""

    @staticmethod
    def reg_self():
        class GptqDynamicA8W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.qint4x2 and config.act_quant_dtype == dtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TOKEN and \
                       config.precision_recovery is PrecisionRecovery.GPTQ

        Quantizer.reg_layer_map(Linear, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear,
                                                ColumnParallelLinearWorldRegion, RowParallelLinearWorldRegion)
            Quantizer.reg_layer_map(ColumnParallelGroupLinear, GptqDynamicQuantLinearCell,
                                    GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(RowParallelGroupLinear, GptqDynamicQuantLinearCell,
                                    GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(ColumnParallelLinearWorldRegion, GptqDynamicQuantLinearCell,
                                    GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(RowParallelLinearWorldRegion, GptqDynamicQuantLinearCell,
                                    GptqDynamicA8W8Checker())
        except ImportError:
            pass
        try:
            from mindformers.experimental.infer.core.layers import RowParallelLinear, ColumnParallelLinear
            Quantizer.reg_layer_map(ColumnParallelLinear, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(RowParallelLinear, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
        except ImportError:
            pass
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear,
                                                ColumnParallelLinearWorldRegion, RowParallelLinearWorldRegion)
            from research.deepseek3.infer.layers import ColumnParallelLinear as DSColumnParallelLinear
            from research.deepseek3.infer.layers import RowParallelLinear as DSRowParallelLinear
            Quantizer.reg_layer_map(DSColumnParallelLinear, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(DSRowParallelLinear, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(ColumnParallelGroupLinear, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(RowParallelGroupLinear, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(ColumnParallelLinearWorldRegion, GptqDynamicQuantLinearCell,
                                    GptqDynamicA8W8Checker())
            Quantizer.reg_layer_map(RowParallelLinearWorldRegion, GptqDynamicQuantLinearCell, GptqDynamicA8W8Checker())
        except ImportError:
            pass

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.weight_need_allgather = False
        self.h = ops.zeros((self.ic, self.ic), dtype=dtype.float32)
        if self.cfg.mode == PTQMode.QUANTIZE and not self.cfg.algo_args["desc_act"]:
            raise ValueError(f"When use GPTQ Dynamic quant algorithm, desc_act in GPTQConfig must be True.")

    def _quant_info(self):
        res = super()._quant_info()
        if self.cfg.act_quant_dtype == dtype.int8:
            return f'{res}-A8-{str(self.cfg.act_quant_granularity)}'
        raise RuntimeError(f"Unexpected act_quant_dtype: {self.cfg.act_quant_dtype}.")

    def deploy(self):
        w_qparam = QuantParam(self.w_scale, self.w_zp, self.cfg.group_size, self.cfg.weight_quant_dtype)
        return GptqDynamicQuantLinearInferCell(self._layer_name, self.layer, self.cfg, self.q_weight,
                                               w_qparam, self.compute_type, self.parallel_type)


class GptqDynamicQuantLinearInferCell(LinearInferCell):
    """GptqDynamicQuantLinearInferCell"""

    def __init__(self, layer_name, linear: Linear, cfg, q_weight, w_qparam: QuantParam, compute_type,
                 parallel_type: ParallelType):
        super().__init__(linear, parallel_type)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        if not is_deploy:
            logger.debug(f"GptqDynamicQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is "
                         f"{w_qparam}")
            logger.debug(f"GptqDynamicQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}, {q_weight.asnumpy()}}}")
        qmm, q_weight, dynamic_quant_op = GptqDynamicQuantMatmul.create(layer_name, q_weight, linear,
                                                                        w_qparam, is_deploy, False,
                                                                        self.layer.transpose_b, compute_type,
                                                                        self.cfg.save_gmm_bias_in_quant_phase)
        self._set_act_dynamic_quant(dynamic_quant_op)
        self.layer.matmul = qmm
        self.layer.weight = q_weight
