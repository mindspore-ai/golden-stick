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

import numpy as np

from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer

from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear

from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode
from mindspore_gs.quantization.quant_utils import quant_tensor
from mindspore_gs.ptq.ptq.hal import QuantParam, WeightQuantMatmul, ParallelType
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .parallel_minmax import get_quant_min_max_op
from .linear_wrapper import WrapperLinearCell, LinearInferCell


class WeightQuantLinearCell(WrapperLinearCell):
    """WeightQuantLinearCell"""

    @staticmethod
    def reg_self():
        class A16W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype is None

        Quantizer.reg_layer_map(Linear, WeightQuantLinearCell, A16W8Checker())
        Quantizer.reg_layer_map(ColumnParallelLinear, WeightQuantLinearCell, A16W8Checker())
        Quantizer.reg_layer_map(RowParallelLinear, WeightQuantLinearCell, A16W8Checker())

    def __init__(self, linear_name, linear, cfg: InnerPTQConfig, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)
        if isinstance(self.layer, RowParallelLinear):
            self.parallel_type = ParallelType.ROW_PARALLEL
        elif isinstance(self.layer, ColumnParallelLinear):
            self.parallel_type = ParallelType.COL_PARALLEL
        elif isinstance(self.layer, Linear):
            self.parallel_type = ParallelType.NO_PARALLEL
        else:
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        if self.cfg.act_per_channel:
            raise ValueError("only per-tensor activation quantization now.")
        if not self.cfg.weight_per_channel:
            raise ValueError("only per-channel weight quantization now.")
        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if linear.transpose_b else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]

        self.compute_type = self.layer.dtype if self.parallel_type == ParallelType.NO_PARALLEL else \
            self.layer.compute_dtype

        is_rowparallel = self.parallel_type == ParallelType.ROW_PARALLEL
        self.w_quant_max, self.w_quant_min = get_quant_min_max_op(self.tensor_parallel, is_rowparallel)

        self.q_weight = Parameter(initializer("ones", self.layer.weight.shape, dtype.int8), name=self.layer.weight.name)
        self.w_scale = Parameter(initializer('ones', (self.oc,), dtype=dtype.float64))
        self.w_zp = Parameter(initializer('ones', (self.oc,), dtype=dtype.float64))

    def quant(self):
        """quant"""
        # quant weight
        w_scale, w_zp, q_weight = quant_tensor(self.layer.weight, self.w_quant_min, self.w_quant_max,
                                               self.cfg.weight_narrow_range, self.cfg.weight_symmetric,
                                               self.cfg.weight_quant_dtype, self.weight_quantizer_axis)
        self.q_weight.set_data(Tensor(q_weight.asnumpy(), dtype=dtype.int8))
        self.w_scale.set_data(Tensor(np.squeeze(w_scale), dtype=dtype.float64))
        self.w_zp.set_data(Tensor(np.squeeze(w_zp), dtype=dtype.float64))

    def process(self):
        super().process()
        self.quant()
        # pylint: disable=protected-access
        self.layer.weight._offload()
        self.cat_samples = None

    def deploy(self):
        return WeightQuantLinearInferCell(self._layer_name, self.layer, self.cfg, self.q_weight,
                                          QuantParam(self.w_scale, self.w_zp), self.compute_type, self.parallel_type)


class WeightQuantLinearInferCell(LinearInferCell):
    """WeightQuantLinearInferCell"""

    def __init__(self, layer_name, linear: Linear, cfg, q_weight, w_qparam: QuantParam, compute_type,
                 parallel_type: ParallelType):
        super().__init__(linear, parallel_type, dtype.int8)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        if not is_deploy:
            logger.debug(f"WeightQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"WeightQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}, {q_weight.asnumpy()}}}")
        qmm = WeightQuantMatmul.create(layer_name, linear.matmul, w_qparam, is_deploy, False, self.layer.transpose_b,
                                       compute_type)
        self.layer.matmul = qmm
        self.layer.weight = q_weight

    def sharded_state_dict(self, **kwargs):
        """provide the sharded state dict based on the config"""
        state_dict = super().sharded_state_dict()
        qmm_state_dict = self.layer.matmul.param_shard_state(self.layer.tensor_parallel_group_size)
        state_dict = state_dict.update(qmm_state_dict)
        return state_dict
