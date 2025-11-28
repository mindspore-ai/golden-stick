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
from mindformers.parallel_core.inference.tensor_parallel.layers import (
    ColumnParallelLinear as McoreColumnParallelLinear, RowParallelLinear as McoreRowParallelLinear)
from mindformers.parallel_core.inference.tensor_parallel.layers import QKVParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.layers import ReplicatedLinear
from mindformers.parallel_core.inference.tensor_parallel.layers import MergedColumnParallelLinear
from mindformers.parallel_core.inference.tensor_parallel.grouped_layers import (
    ColumnParallelGroupedLinear,
    RowParallelGroupedLinear
)

from mindspore_gs.common import logger
from mindspore_gs.common.utils import offload_param
from mindspore_gs.ptq.ptq_config import PTQMode, QuantGranularity, PrecisionRecovery
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_functions.basic_quant_func import quant_tensor
from mindspore_gs.ptq.ptq.hal import QuantParam, WeightQuantMatmul, WeightQuantInt4Matmul, ParallelType
from mindspore_gs.ptq.algo_modules import Quantizer
from mindspore_gs.ptq.utils import QuantType
from mindspore_gs.ptq.quant_cells.quant_cell import Checker
from .parallel_minmax import get_min_max_op
from .linear_wrapper import WrapperLinearCell, LinearInferCell
from .mcore_linear_wrapper import McoreLinearInferCell


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
        Quantizer.reg_layer_map(McoreColumnParallelLinear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(McoreRowParallelLinear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(QKVParallelLinear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(MergedColumnParallelLinear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(ColumnParallelGroupedLinear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(RowParallelGroupedLinear, WeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(ReplicatedLinear, WeightQuantLinearCell, A16WxChecker())
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear)
            from research.deepseek3.infer.layers import ColumnParallelLinear as DSColumnParallelLinear
            from research.deepseek3.infer.layers import RowParallelLinear as DSRowParallelLinear
            Quantizer.reg_layer_map(ColumnParallelGroupLinear, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(RowParallelGroupLinear, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(DSColumnParallelLinear, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(DSRowParallelLinear, WeightQuantLinearCell, A16WxChecker())
        except ImportError:
            pass
        try:
            from research.llama3_1.infer.layers import ColumnParallelLinear as LlamaColumnParallelLinear
            from research.llama3_1.infer.layers import RowParallelLinear as LlamaRowParallelLinear
            Quantizer.reg_layer_map(LlamaColumnParallelLinear, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(LlamaRowParallelLinear, WeightQuantLinearCell, A16WxChecker())
        except ImportError:
            pass
        try:
            from research.telechat2.infer.layers import ColumnParallelLinear as TC2ColumnParallelLinear
            from research.telechat2.infer.layers import RowParallelLinear as TC2RowParallelLinear
            Quantizer.reg_layer_map(TC2ColumnParallelLinear, WeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(TC2RowParallelLinear, WeightQuantLinearCell, A16WxChecker())
        except ImportError:
            pass

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)

        type_map = {Linear: ParallelType.NO_PARALLEL}
        type_map[McoreColumnParallelLinear] = ParallelType.COL_PARALLEL
        type_map[McoreRowParallelLinear] = ParallelType.ROW_PARALLEL
        type_map[QKVParallelLinear] = ParallelType.COL_PARALLEL
        type_map[MergedColumnParallelLinear] = ParallelType.COL_PARALLEL
        type_map[ColumnParallelGroupedLinear] = ParallelType.COL_PARALLEL
        type_map[RowParallelGroupedLinear] = ParallelType.ROW_PARALLEL
        type_map[ReplicatedLinear] = ParallelType.NO_PARALLEL
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear)
            from research.deepseek3.infer.layers import ColumnParallelLinear as DSColumnParallelLinear
            from research.deepseek3.infer.layers import RowParallelLinear as DSRowParallelLinear
            type_map[ColumnParallelGroupLinear] = ParallelType.COL_PARALLEL
            type_map[RowParallelGroupLinear] = ParallelType.ROW_PARALLEL
            type_map[DSColumnParallelLinear] = ParallelType.COL_PARALLEL
            type_map[DSRowParallelLinear] = ParallelType.ROW_PARALLEL
        except ImportError:
            pass
        try:
            from research.llama3_1.infer.layers import ColumnParallelLinear as LlamaColumnParallelLinear
            from research.llama3_1.infer.layers import RowParallelLinear as LlamaRowParallelLinear
            type_map[LlamaColumnParallelLinear] = ParallelType.COL_PARALLEL
            type_map[LlamaRowParallelLinear] = ParallelType.ROW_PARALLEL
        except ImportError:
            pass
        try:
            from research.telechat2.infer.layers import ColumnParallelLinear as TC2ColumnParallelLinear
            from research.telechat2.infer.layers import RowParallelLinear as TC2RowParallelLinear
            type_map[TC2ColumnParallelLinear] = ParallelType.COL_PARALLEL
            type_map[TC2RowParallelLinear] = ParallelType.ROW_PARALLEL
        except ImportError:
            pass
        self.parallel_type = type_map.get(type(self.layer), None)
        if not self.parallel_type:
            raise ValueError("only Linear、ColumnParallelLinear、RowParallelLinear cell is supported,"
                             f"but {linear_name} type is {type(linear)}.")
        if self.cfg.act_per_channel:
            raise ValueError("only per-tensor activation quantization now.")

        rank = len(linear.weight.shape)
        ic_axis = rank - 1 if self._transpose_b() else rank - 2
        self.weight_quantizer_axis = rank - 2 if self._transpose_b() else rank - 1
        self.ic = linear.weight.shape[ic_axis]
        self.oc = linear.weight.shape[self.weight_quantizer_axis]

        self.compute_type = self.layer.dtype if self.parallel_type == ParallelType.NO_PARALLEL \
             and isinstance(linear, Linear) else self.layer.compute_dtype

        is_rowparallel = self.parallel_type == ParallelType.ROW_PARALLEL
        if cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, False)
        else:
            self.w_quant_max, self.w_quant_min = get_min_max_op(cfg.tp_size, is_rowparallel)

        self.q_weight = Parameter(initializer("ones", self.layer.weight.shape, dtype.int8), name=self.layer.weight.name)
        if self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP:
            if self.ic % self.cfg.group_size != 0:
                raise ValueError(f"input channel {self.ic} can not divide group_size {self.cfg.group_size}.")
            if rank == 2:
                scale_zp_shape = (self.ic // self.cfg.group_size, self.oc)
            elif rank == 3:
                scale_zp_shape = (linear.weight.shape[0],
                                  linear.weight.shape[1] // self.cfg.group_size,
                                  linear.weight.shape[2])
            else:
                raise ValueError(f"Only support rank of weight is 2 or 3, but got {rank}.")
        else:
            if rank == 2:
                scale_zp_shape = (self.oc,)
            elif rank == 3:
                scale_zp_shape = (linear.weight.shape[0], linear.weight.shape[2])
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
                                               is_transpose=self._transpose_b())
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
            offload_param(self.layer.weight)
        self.cat_samples = None

    def deploy(self):
        w_qparam = QuantParam(self.w_scale, self.w_zp, self.cfg.group_size, self.cfg.weight_quant_dtype)
        if self.is_mcorelinear:
            return WeightQuantMcoreLinearInferCell(self._layer_name, self.layer, self.context, self.cfg,
                                                   self.q_weight, w_qparam, self.compute_type, self.parallel_type)
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
                         f"{{{q_weight.shape}, {q_weight.dtype}}}")
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


class WeightQuantMcoreLinearInferCell(McoreLinearInferCell):
    """WeightQuantLinearInferCell"""

    def __init__(self, layer_name, linear: Linear, context, cfg, q_weight, w_qparam: QuantParam, compute_type,
                 parallel_type: ParallelType):
        super().__init__(linear, parallel_type)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        if not is_deploy:
            logger.debug(f"WeightQuantLinearInferCell: w_qparam of Layer({parallel_type}:{layer_name}) is {w_qparam}")
            logger.debug(f"WeightQuantLinearInferCell: q_weight of Layer({parallel_type}:{layer_name}) is "
                         f"{{{q_weight.shape}, {q_weight.dtype}, {q_weight.asnumpy()}}}")
        if w_qparam.quant_dtype == dtype.int8:
            _ = WeightQuantMatmul.create(layer_name, linear, q_weight, w_qparam, is_deploy, False,
                                         self._transpose_b(), compute_type, experimental=True)
        elif w_qparam.quant_dtype == dtype.qint4x2:
            _, q_weight = WeightQuantInt4Matmul.create(layer_name, linear, q_weight, w_qparam, is_deploy, False,
                                                       self._transpose_b(), compute_type, experimental=True,
                                                       fake_quant=context.fake_quant)
            self._set_transpose_b_to_false()
        else:
            raise ValueError("Only support int8 and int4 quantization of weight, please check config info.")
        del self.layer.weight
        self.layer.weight = None
        self.weight = q_weight
        self.weight_scale = Parameter(w_qparam.scale.astype(compute_type))
        self.weight_offset = Parameter(w_qparam.zero_point.astype(dtype.int32))
        self.has_bias = self.layer.has_bias
        if self.has_bias:
            self.bias = self.layer.bias
            self.layer.bias = None

    def quant_type_dict(self):
        """quant_type_dict"""
        type_ = QuantType.FLOAT.value
        if self.cfg.weight_quant_dtype == dtype.int8:
            type_ = QuantType.W8A16.value
        elif self.cfg.weight_quant_dtype == dtype.qint4x2:
            type_ = QuantType.W4A16.value
        quant_type = {
            self.weight_scale.name: type_,
            self.weight_offset.name: type_,
            self.weight.name: type_
        }
        if self.has_bias:
            quant_type.update({self.bias.name: type_})
        return quant_type
