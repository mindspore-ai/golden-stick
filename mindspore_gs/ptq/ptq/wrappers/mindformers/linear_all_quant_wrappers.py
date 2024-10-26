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
from mindspore import ops as msops
from mindspore.common.initializer import initializer
from mindspore.ops.operations.comm_ops import ReduceOp
from mindspore.communication.management import GlobalComm

from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import RowParallelLinear, ColumnParallelLinear

from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode
from mindspore_gs.quantization.quant_utils import quant_tensor
from mindspore_gs.ptq.ptq.hal import QuantParam, AllQuantMatmul, ParallelType
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .parallel_minmax import MaxFromTensorParallelRegion, MinFromTensorParallelRegion
from .linear_weight_quant_wrappers import WeightQuantLinearCell
from .linear_wrapper import LinearInferCell


class AllQuantLinearCell(WeightQuantLinearCell):
    """QuantLinearCell"""

    @staticmethod
    def reg_self():
        class A8W8Checker(Checker):
            def check(self, config: InnerPTQConfig):
                return config.weight_quant_dtype == dtype.int8 and config.act_quant_dtype == dtype.int8 and \
                       config.act_dynamic_quant is False

        Quantizer.reg_layer_map(Linear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(ColumnParallelLinear, AllQuantLinearCell, A8W8Checker())
        Quantizer.reg_layer_map(RowParallelLinear, AllQuantLinearCell, A8W8Checker())

    def __init__(self, linear_name, linear, cfg: InnerPTQConfig, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)

        if self.parallel_type == ParallelType.ROW_PARALLEL:
            self.x_quant_max = MaxFromTensorParallelRegion()
            self.x_quant_min = MinFromTensorParallelRegion()
        else:
            self.x_quant_max = msops.max
            self.x_quant_min = msops.min

        self.x_scale = Parameter(initializer('ones', (1,), dtype=self.compute_type))
        self.x_zp = Parameter(initializer('zeros', (1,), dtype=self.compute_type))

    def quant(self):
        """quant"""
        # quant weight
        super().quant()
        # quant activation
        x_scale, x_zp, _ = quant_tensor(self.cat_samples, self.x_quant_min, self.x_quant_max,
                                        self.cfg.act_narrow_range, self.cfg.act_symmetric,
                                        self.cfg.act_quant_dtype, -1, False)
        self.x_scale.set_data(Tensor([x_scale], dtype=self.compute_type))
        self.x_zp.set_data(Tensor([x_zp], dtype=self.compute_type))

    def deploy(self):
        return AllQuantLinearInferCell(self.layer, self.cfg, self.q_weight, QuantParam(self.x_scale, self.x_zp),
                                       QuantParam(self.w_scale, self.w_zp), self.compute_type, self.parallel_type)


class AllQuantLinearInferCell(LinearInferCell):
    """DeployLinearCell"""

    def __init__(self, linear: Linear, cfg: InnerPTQConfig, q_weight, x_qparam: QuantParam, w_qparam: QuantParam,
                 compute_type, parallel_type: ParallelType):
        super().__init__(linear, parallel_type, dtype.int8)
        self.cfg = cfg
        is_deploy = cfg.mode == PTQMode.DEPLOY
        qmm, smooth_scale = AllQuantMatmul.create(linear.matmul, x_qparam, w_qparam, is_deploy, False,
                                                  self.layer.transpose_b, compute_type)
        bias_name = linear.bias.name if linear.has_bias else q_weight.name + "_bias"
        rank = len(q_weight.shape)
        ic_idx, oc_idx = (rank - 1, rank - 2) if linear.transpose_b else (rank - 2, rank - 1)
        ic, oc = q_weight.shape[ic_idx], q_weight.shape[oc_idx]
        if is_deploy:
            self.layer.bias = Parameter(initializer("ones", (oc,), compute_type), name=bias_name)
            self.input_scale = Parameter(initializer('ones', (ic,), compute_type))
            self.input_zp = Parameter(initializer('zeros', (ic,), dtype.int8))
        else:
            # FIXME: @hangangqiang sink into hal
            # fuse bias
            dequant_scale = np.squeeze(x_qparam.scale.asnumpy() * w_qparam.scale.asnumpy()).astype(np.float32)
            origin_bias = self.layer.bias.asnumpy() if self.layer.has_bias else None
            if isinstance(self.layer, RowParallelLinear):
                bias = self._fused_bias(q_weight, x_qparam.zero_point.asnumpy(), True, dequant_scale, origin_bias)
            else:
                bias = self._fused_bias(q_weight, x_qparam.zero_point.asnumpy(), False, dequant_scale, origin_bias)
            self.layer.bias = Parameter(Tensor(bias, dtype=compute_type), name=bias_name)
            # fuse smooth.mul and quant
            input_scale_np = x_qparam.scale.asnumpy().astype(np.float16)
            if smooth_scale is not None:
                final_scale_np = input_scale_np / smooth_scale.astype(np.float16)
            else:
                if input_scale_np.shape == (1,):
                    final_scale_np = np.array([input_scale_np[0]] * ic).astype(np.float16)
                elif not input_scale_np.shape:
                    final_scale_np = np.array([input_scale_np] * ic).astype(np.float16)
                else:
                    final_scale_np = input_scale_np
            self.input_scale = Parameter(Tensor(final_scale_np, dtype=dtype.float16))
            if self.input_scale.shape != x_qparam.zero_point.shape:
                if isinstance(x_qparam.zero_point, np.number):
                    raise RuntimeError("Shape of scale and zero point are not compatible.")
                param_len = self.input_scale.shape[0]
                self.input_zp = Parameter(Tensor(np.array([x_qparam.zero_point] * param_len)
                                                 .reshape(self.input_scale.shape)
                                                 .astype(np.int8),
                                                 dtype=dtype.int8))
            else:
                self.input_zp = Parameter(Tensor(x_qparam.zero_point, dtype=dtype.int8))
        self._set_act_quant(self.input_scale, self.input_zp)
        if linear.has_bias is False:
            self.layer.has_bias = True
            self.layer.bias_add = msops.Add()
        self.layer.matmul = qmm
        self.layer.weight = q_weight

    def _fused_bias(self, quant_weight, act_offset, new_bias_need_allreduce=False, dequant_scale=None,
                    origin_bias=None):
        """compute fused bias"""
        if quant_weight is None:
            if origin_bias:
                return origin_bias
            return None
        new_bias = -np.sum(act_offset.astype(np.int32) * quant_weight.asnumpy().astype(np.int32),
                           axis=1 if self.layer.transpose_b else 0).astype(np.int32)
        if new_bias_need_allreduce:
            t_new_bias = Tensor(new_bias)
            t_new_bias = msops.AllReduce(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)(t_new_bias)
            new_bias = t_new_bias.asnumpy()
        if dequant_scale is not None:
            new_bias = new_bias.astype(np.float64) * dequant_scale
        if origin_bias is not None:
            new_bias = new_bias + origin_bias
        return new_bias

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        state_dict = super().sharded_state_dict()
        if self.parallel_type == ParallelType.COL_PARALLEL:
            input_scale_shard = (1,)
            input_zp_shard = (1,)
        elif self.parallel_type == ParallelType.ROW_PARALLEL:
            input_scale_shard = (self.layer.tensor_parallel_group_size,)
            input_zp_shard = (self.layer.tensor_parallel_group_size,)
        else:
            return {}
        state_dict[self.input_scale.name] = {'shape': self.input_scale.shape, 'shard': input_scale_shard}
        state_dict[self.input_zp.name] = {'shape': self.input_zp.shape, 'shard': input_zp_shard}

        qmm_state_dict = self.layer.matmul.param_shard_state(self.layer.tensor_parallel_group_size)
        state_dict = state_dict.update(qmm_state_dict)
        return state_dict
