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
from mindspore_gs.ptq.ptq_config import PTQMode, QuantGranularity, PrecisionRecovery
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_quant_func import quant_tensor
from mindspore_gs.ptq.ptq.hal import QuantParam, WeightQuantMatmul, WeightQuantInt4Matmul, ParallelType
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .parallel_minmax import get_min_max_op
from .linear_wrapper import WrapperLinearCell, LinearInferCell


class WeightQuantLinearCell(WrapperLinearCell):
    """WeightQuantLinearCell"""

    @staticmethod
    def reg_self():
        """register WeightQuantLinearCell"""
        class A16WxChecker(Checker):
            def check(self, config: InnerPTQConfig):
                support_dtype = [dtype.int8, dtype.qint4x2]
                return (config.weight_quant_dtype in support_dtype and config.act_quant_dtype is None
                        and config.precision_recovery == PrecisionRecovery.NONE)

        Quantizer.reg_layer_map(Linear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(ColumnParallelLinear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(RowParallelLinear, WeightQuantLinearCell, A16WxChecker())
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear,
                                                ColumnParallelLinearWorldRegion, RowParallelLinearWorldRegion)
            Quantizer.reg_layer_map(ColumnParallelGroupLinear, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(RowParallelGroupLinear, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(ColumnParallelLinearWorldRegion, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(RowParallelLinearWorldRegion, WeightQuantLinearCell, A16WxChecker())
        except ImportError:
            pass

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
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
        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if linear.transpose_b else rank - 2
        self.weight_quantizer_axis = rank - 2 if linear.transpose_b else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]

        self.compute_type = self.layer.dtype if self.parallel_type == ParallelType.NO_PARALLEL else \
            self.layer.compute_dtype

        is_rowparallel = self.parallel_type == ParallelType.ROW_PARALLEL
        if cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, False)
        else:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, is_rowparallel)

        self.q_weight = Parameter(initializer("ones", self.layer.weight.shape, dtype.int8), name=self.layer.weight.name)
        if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            if self.ic % self.cfg.group_size != 0:
                raise ValueError(f"input channel {self.ic} can not divide group_size {self.cfg.group_size}.")
            if self.ic == self.cfg.group_size:
                raise ValueError(f"input channel {self.ic} can not equal to group_size {self.cfg.group_size}.")
            if rank == 2:
                scale_zp_shape = (self.ic // self.cfg.group_size, self.oc)
            elif rank == 3:
                scale_zp_shape = (linear.weight.shape[0],
                                  linear.weight.shape[1] // self.cfg.group_size,
                                  linear.weight.shape[2]) if not self.layer.transpose_b else \
                                 (linear.weight.shape[0],
                                  linear.weight.shape[2] // self.cfg.group_size,
                                  linear.weight.shape[1])
            else:
                raise ValueError(f"Only support rank of weight is 2 or 3, but got {rank}.")
        else:
            if rank == 2:
                scale_zp_shape = (self.oc,)
            elif rank == 3:
                scale_zp_shape = (linear.weight.shape[0], linear.weight.shape[2]) if not self.layer.transpose_b else \
                                 (linear.weight.shape[0], linear.weight.shape[1])
            else:
                raise ValueError(f"Only support rank of weight is 2 or 3, but got {rank}.")
        self.w_scale = Parameter(initializer('ones', scale_zp_shape, dtype=dtype.float64))
        self.w_zp = Parameter(initializer('zeros', scale_zp_shape, dtype=dtype.float64))

    def _quant_info(self):
        if self.cfg.weight_quant_dtype == dtype.int8:
            return f'W8-{str(self.cfg.weight_quant_granularity)}'
        if self.cfg.weight_quant_dtype == dtype.qint4x2:
            return f'W4-{str(self.cfg.weight_quant_granularity)}'
        raise RuntimeError(f"Unexpected weight_quant_dtype: {self.cfg.weight_quant_dtype}.")

    def quant(self):
        """quant"""
        # quant weight
        w_scale, w_zp, q_weight = quant_tensor(self.layer.weight, self.w_quant_min, self.w_quant_max,
                                               self.cfg.weight_narrow_range, self.cfg.weight_symmetric,
                                               self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP,
                                               self.cfg.group_size, self.cfg.weight_quant_dtype,
                                               self.weight_quantizer_axis,
                                               is_transpose=self._layer.transpose_b)
        if self.cfg.weight_quant_granularity == QuantGranularity.PER_CHANNEL:
            w_scale = np.squeeze(w_scale)
            w_zp = np.squeeze(w_zp)
        self.q_weight.set_data(q_weight.astype(dtype=dtype.int8))
        self.w_scale.set_data(Tensor(w_scale, dtype=dtype.float64))
        self.w_zp.set_data(Tensor(w_zp, dtype=dtype.float64))
        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|input0_weight", self.layer.weight)
        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|output0_qweight", self.q_weight)
        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|output1_weight_scale", self.w_scale)
        self.cfg.dumper.dump_data(self.layer_name, "|weight_params|output2_weight_zp", self.w_zp)

    def process(self):
        super().process()
        self.quant()
        if not self.cfg.skip_offload_in_processing:
            # FIXME: Experiments show that offloading weight here may lead to memory leak. Set the temporary flag
            # 'skip_offload_in_processing' to skip this call, the weight param will be offloaded in PTQ.apply procedure.
            # The switch should be removed after the issue is fixed. -- @tongl2
            # pylint: disable=protected-access
            self.layer.weight._offload()
        self.cat_samples = None

    def deploy(self):
        w_qparam = QuantParam(self.w_scale, self.w_zp, self.cfg.group_size, self.cfg.weight_quant_dtype)
        return WeightQuantLinearInferCell(self._layer_name, self.layer, self.cfg, self.q_weight, w_qparam,
                                          self.compute_type, self.parallel_type)


class WeightQuantLinearInferCell(LinearInferCell):
    """WeightQuantLinearInferCell"""

    def __init__(self, layer_name, linear: Linear, cfg, q_weight, w_qparam: QuantParam, compute_type,
                 parallel_type: ParallelType):
        super().__init__(linear, parallel_type)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        if not is_deploy:
            logger.debug(f"WeightQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"WeightQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}, {q_weight.asnumpy()}}}")
        if w_qparam.quant_dtype == dtype.int8:
            qmm = WeightQuantMatmul.create(layer_name, linear, q_weight, w_qparam, is_deploy, False,
                                           self.layer.transpose_b, compute_type)
        elif w_qparam.quant_dtype == dtype.qint4x2:
            qmm, q_weight = WeightQuantInt4Matmul.create(layer_name, linear, q_weight, w_qparam, is_deploy, False,
                                                         self.layer.transpose_b, compute_type)
            self.layer.transpose_b = False
        else:
            raise ValueError("Only support int8 and int4 quantization of weight, please check config info.")
        self.layer.matmul = qmm
        del self.layer.weight
        self.layer.weight = q_weight
