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

from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import RowParallelLinear, ColumnParallelLinear
from mindspore import dtype
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode, QuantGranularity
from mindspore_gs.ptq.ptq.hal import QuantParam, DynamicQuantMatmul, ParallelType
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .linear_weight_quant_wrappers import WeightQuantLinearCell
from .linear_wrapper import LinearInferCell


class DynamicQuantLinearCell(WeightQuantLinearCell):
    """WeightQuantLinearCell"""

    @staticmethod
    def reg_self():
        class DynamicA8W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype == dtype.int8 and \
                       config.act_quant_granularity is QuantGranularity.PER_TOKEN

        Quantizer.reg_layer_map(Linear, DynamicQuantLinearCell, DynamicA8W8Checker())
        Quantizer.reg_layer_map(ColumnParallelLinear, DynamicQuantLinearCell, DynamicA8W8Checker())
        Quantizer.reg_layer_map(RowParallelLinear, DynamicQuantLinearCell, DynamicA8W8Checker())

    def deploy(self):
        return DynamicQuantLinearInferCell(self._layer_name, self.layer, self.cfg, self.q_weight,
                                           QuantParam(self.w_scale, self.w_zp), self.compute_type, self.parallel_type)


class DynamicQuantLinearInferCell(LinearInferCell):
    """DynamicQuantLinearInferCell"""

    def __init__(self, layer_name, linear: Linear, cfg, q_weight, w_qparam: QuantParam, compute_type,
                 parallel_type: ParallelType):
        super().__init__(linear, parallel_type)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        if not is_deploy:
            logger.debug(f"DynamicQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"DynamicQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}, {q_weight.asnumpy()}}}")
        qmm = DynamicQuantMatmul.create(layer_name, linear.matmul, w_qparam, is_deploy, False, self.layer.transpose_b,
                                        compute_type)
        self.layer.matmul = qmm
        self.layer.weight = q_weight

    def sharded_state_dict(self, **kwargs):
        """provide the sharded state dict based on the config"""
        state_dict = super().sharded_state_dict()
        qmm_state_dict = self.layer.matmul.param_shard_state(self.layer.tensor_parallel_group_size)
        state_dict = state_dict.update(qmm_state_dict)
        return state_dict
