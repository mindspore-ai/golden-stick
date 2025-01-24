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

import math
import time

from mindspore import Tensor, dtype, numpy
from mindspore import ops as msops
from mindspore.communication import get_rank
from mindspore.communication.management import GlobalComm
from mindspore.ops import sub as aclnn_sub, add as aclnn_add
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import PrecisionRecovery, QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.basic_quant_func import quant_tensor, get_quant_min_max
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.hal import ParallelType
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from .linear_weight_quant_wrappers import WeightQuantLinearCell


class GptqWeightQuantLinearCell(WeightQuantLinearCell):
    """GptqWeightQuantLinearCell"""

    @staticmethod
    def reg_self():
        class A16WxChecker(Checker):
            def check(self, config: InnerPTQConfig):
                support_dtype = [dtype.int8, dtype.qint4x2]
                return (config.weight_quant_dtype in support_dtype and config.act_quant_dtype is None
                        and config.precision_recovery == PrecisionRecovery.GPTQ)

        Quantizer.reg_layer_map(Linear, GptqWeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(ColumnParallelLinear, GptqWeightQuantLinearCell, A16WxChecker())
        Quantizer.reg_layer_map(RowParallelLinear, GptqWeightQuantLinearCell, A16WxChecker())

    def __init__(self, linear_name, linear, cfg: InnerPTQConfig, network_helper, **kwargs):
        super().__init__(linear_name, linear, cfg, network_helper, **kwargs)
        self.nsamples = 0
        self.h = msops.zeros((self.layer.weight.shape[1], self.layer.weight.shape[1]))
        self.cfg.reflash_inputs_after_each_processor = True
        self.group_scale = []
        self.group_zero = []
        self.qweight = []
        if self.cfg.tp_size > 1:
            self.rank_id = get_rank()
        if self.parallel_type == ParallelType.ROW_PARALLEL and self.cfg.tp_size > 1:
            self.weight_need_allgather = True
        else:
            self.weight_need_allgather = False
        if self.weight_need_allgather:
            self.h = msops.zeros((self.ic * self.cfg.tp_size, self.ic * self.cfg.tp_size))
        else:
            self.h = msops.zeros((self.ic, self.ic))
        self.weight_quant_min, self.weight_quant_max = get_quant_min_max(num_bits=4, signed=self.cfg.weight_symmetric,
                                                                         narrow_range=self.cfg.weight_narrow_range)

    def _hessian_compute(self):
        """compute Hessian Matrix"""
        for i in range(len(self.samples)):
            if len(self.samples[i].shape) == 3:
                self.samples[i] = self.samples[i].reshape((-1, self.samples[i].shape[-1]))
            sqe = self.nsamples / (self.nsamples + 1)
            self.nsamples += 1
            sqr = math.sqrt(2 / self.nsamples)
            self.samples[i] = sqr * self.samples[i]
            self.h *= sqe
            if self.weight_need_allgather:
                inp = msops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)(self.samples[i].T)
                inp = Tensor(inp.T, dtype=dtype.float32)
                self.h += msops.matmul(inp.T, inp)
            else:
                self.samples[i] = self.samples[i].astype(dtype.float32)
                self.h += msops.matmul(self.samples[i].T, self.samples[i])
        self.cfg.dumper.dump_data(self.layer_name, "|hessian_matrix|input0_activation_inputs",
                                  msops.cat(tuple(self.samples), axis=0))
        self.cfg.dumper.dump_data(self.layer_name, "|hessian_matrix|output0_hessian", self.h)

    def _gptq_precision_recovery(self, weight, hinv, scale, zero, perm):
        """precision recovery use gptq"""
        group_size = self.cfg.group_size
        losses = msops.zeros_like(weight, dtype=weight.dtype)
        q = msops.zeros_like(weight, dtype=weight.dtype)
        now_idx = 1
        for i1 in range(0, weight.shape[1], self.cfg.algo_args["block_size"]):
            i2 = min(i1 + self.cfg.algo_args["block_size"], weight.shape[1])
            count = i2 - i1
            w1 = weight[:, i1:i2]
            q1 = msops.zeros_like(w1, dtype=w1.dtype)
            err = msops.zeros_like(w1, dtype=dtype.float32)
            losses1 = msops.zeros_like(w1, dtype=dtype.float32)
            hinv1 = hinv[i1:i2, i1:i2]
            hinv2 = hinv[i1:i2, i2:]
            for i in range(count):
                w0 = w1[:, i]
                d = hinv1[i, i]

                if group_size != 0:
                    if not self.cfg.algo_args["static_groups"]:
                        if (i1 + i) % group_size == 0:
                            scale, zero, _ = quant_tensor(weight[:, (i1 + i):(i1 + i + group_size)], self.w_quant_min,
                                                          self.w_quant_max, self.cfg.weight_narrow_range,
                                                          self.cfg.weight_symmetric, False, 0,
                                                          self.cfg.weight_quant_dtype, self.weight_quantizer_axis,
                                                          False)
                            scale = Tensor(scale, dtype.float32)
                            zero = Tensor(zero, dtype.float32)
                        if ((i1 + i) // group_size) - now_idx == -1:
                            self.group_scale.append(scale.T)
                            self.group_zero.append(zero.T)
                            now_idx += 1
                    else:
                        idx = i1
                        if self.cfg.algo_args["desc_act"]:
                            idx = perm[idx]
                        scale = self.group_scale[idx // group_size].T
                        zero = self.group_zero[idx // group_size].T
                q0 = msops.clip_by_value(aclnn_add(msops.round(w0.unsqueeze(1) / scale), zero),
                                         Tensor(self.weight_quant_min), Tensor(self.weight_quant_max))
                self.qweight.append(q0)
                q0 = scale * aclnn_sub(q0, zero)
                q0 = q0.flatten()
                delta_loss = aclnn_sub(w0, q0) ** 2 / d ** 2
                err1 = aclnn_sub(w0, q0) / d
                delta_w = err1.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                q1[:, i] = q0
                losses1[:, i] = delta_loss
                w1[:, i:] = aclnn_sub(w1[:, i:], delta_w)
                err[:, i] = err1

            q[:, i1:i2] = q1
            weight[:, i2:] = aclnn_sub(weight[:, i2:], err.matmul(hinv2))
            losses[:, i1:i2] = losses1 / 2
        if group_size == 0:
            self.group_scale.append(scale)
            self.group_zero.append(zero)
            self.group_scale = msops.cat(self.group_scale)
            self.group_zero = msops.cat(self.group_zero)
            self.group_scale = msops.squeeze(self.group_scale)
            self.group_zero = msops.squeeze(self.group_zero)
        else:
            self.group_scale = msops.cat(self.group_scale)
            self.group_zero = msops.cat(self.group_zero)
        self.qweight = msops.cat(self.qweight, 1)
        logger.info(f'error: {msops.sum(losses)}')
        if self.weight_need_allgather and group_size != 0:
            self.group_scale = self.group_scale[self.rank_id * self.w_scale.shape[0] :
                                                self.rank_id * self.w_scale.shape[0] + self.w_scale.shape[0], :]
            self.group_zero = self.group_zero[self.rank_id * self.w_zp.shape[0] :
                                              self.rank_id * self.w_zp.shape[0] + self.w_zp.shape[0], :]
        return weight

    def _apply_gptq(self, scale, zero):
        """apply gptq"""
        self._hessian_compute()
        self.samples.clear()
        dead = numpy.diag(self.h) == 0
        dead = msops.nonzero(dead)
        if dead.shape[0] > 0:
            self.h[dead, dead] = 1
            self.layer.weight[:, dead] = 0
        if self.weight_need_allgather:
            weight = msops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)(self.layer.weight.T)
            weight = Tensor(weight.T, dtype=self.layer.weight.dtype)
        else:
            weight = self.layer.weight.value()
        if self.cfg.algo_args["static_groups"] and self.cfg.group_size != 0:
            for i in range(0, weight.shape[1], self.cfg.group_size):
                scale, zero, _ = quant_tensor(weight[:, i : i + self.cfg.group_size], self.w_quant_min,
                                              self.w_quant_max, self.cfg.weight_narrow_range, self.cfg.weight_symmetric,
                                              False, 0, self.cfg.weight_quant_dtype,
                                              self.weight_quantizer_axis, False)
                self.group_scale.append(Tensor(scale, self.layer.weight.dtype).T)
                self.group_zero.append(Tensor(zero, self.layer.weight.dtype).T)
        perm = []
        if self.cfg.algo_args["desc_act"]:
            perm = msops.argsort(numpy.diag(self.h), descending=True)
            weight = weight[:, perm]
            self.h = self.h[perm][:, perm]
            invperm = msops.argsort(perm)
        from mindspore_gs.ptq.cholesky_trans import cholesky_compute
        cholesky_time = time.time()
        hinv = cholesky_compute(self.h, self.cfg.algo_args["damp_percent"])
        self.cfg.dumper.dump_data(self.layer_name, "|cholesky_decomposition|input0_hessian", self.h)
        self.cfg.dumper.dump_data(self.layer_name, "|cholesky_decomposition|output0_inv_hessian", hinv)
        del self.h
        logger.info(f'[TIME]end cholesky part with time {time.time() - cholesky_time}s')
        quant_tick = time.time()
        qweight = self._gptq_precision_recovery(weight, hinv, scale, zero, perm)
        logger.info(f'[TIME]quant layers with time {time.time() - quant_tick}s')
        if self.cfg.algo_args["desc_act"]:
            qweight = qweight[:, invperm]
            self.qweight = self.qweight[:, invperm]
        if self.weight_need_allgather:
            self.layer.weight = qweight[:, self.rank_id * self.ic : self.rank_id * self.ic + self.ic]
            self.qweight = self.qweight[:, self.rank_id * self.ic : self.rank_id * self.ic + self.ic]
        else:
            self.layer.weight = qweight

    def quant(self):
        """quant"""
        scale, zp, _ = quant_tensor(self.layer.weight, self.w_quant_min, self.w_quant_max,
                                    self.cfg.weight_narrow_range, self.cfg.weight_symmetric,
                                    self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP,
                                    self.cfg.group_size, self.cfg.weight_quant_dtype,
                                    self.weight_quantizer_axis, False)
        self._apply_gptq(Tensor(scale, dtype=self.layer.weight.dtype), Tensor(zp, dtype=self.layer.weight.dtype))
        self.q_weight.set_data(Tensor(self.qweight, dtype=dtype.int8))
        self.w_scale.set_data(Tensor(self.group_scale, dtype=dtype.float64))
        self.w_zp.set_data(Tensor(self.group_zero, dtype=dtype.float64))
        self.group_scale = None
        self.group_zero = None
        self.qweight = None

    def process(self):
        self.quant()
        # pylint: disable=protected-access
        self.layer.weight._offload()
