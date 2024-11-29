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
from mindspore.ops import sub as aclnn_sub, add as aclnn_add
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PrecisionRecovery
from mindspore_gs.quantization.quant_utils import quant_tensor, quant_tensor_data, get_quant_min_max
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.cholesky_trans import cholesky_compute
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

    def __init__(self, linear_name, linear, cfg: InnerPTQConfig, network_helper):
        super().__init__(linear_name, linear, cfg, network_helper)
        self.nsamples = 0
        self.h = msops.zeros((self.layer.weight.shape[1], self.layer.weight.shape[1]))
        self.cfg.reflash_inputs_after_each_processor = True
        self.group_scale = []
        self.group_zero = []
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
            self.samples[i] = self.samples[i].astype(dtype.float32)
            self.h += msops.matmul(self.samples[i].t(), self.samples[i])

    def _gptq_precision_recovery(self, w, hinv, scale, zero):
        """precision recovery use gptq"""
        if self.cfg.algo_args["static_group"]:
            for i in range(0, self.ic, self.cfg.group_size):
                scale, zero, _ = quant_tensor(self.layer.weight[:, i : i + self.cfg.group_size], self.w_quant_min,
                                              self.w_quant_max, self.cfg.weight_quant_dtype, self.weight_quantizer_axis,
                                              False, 4)
                self.group_scale.append(Tensor(scale, self.layer.weight.dtype).T)
                self.group_zero.append(Tensor(zero, self.layer.weight.dtype).T)
        group_size = self.cfg.group_size
        losses = msops.zeros_like(w, dtype=w.dtype)
        q = msops.zeros_like(w, dtype=w.dtype)
        now_idx = 1
        for i1 in range(0, self.ic, self.cfg.algo_args["block_columns"]):
            i2 = min(i1 + self.cfg.algo_args["block_columns"], self.ic)
            count = i2 - i1
            w1 = w[:, i1:i2]
            q1 = msops.zeros_like(w1, dtype=w1.dtype)
            err = msops.zeros_like(w1, dtype=w1.dtype)
            losses1 = msops.zeros_like(w1, dtype=w1.dtype)
            hinv1 = hinv[i1:i2, i1:i2]
            hinv2 = hinv[i1:i2, i2:]
            for i in range(count):
                w0 = w1[:, i]
                d = hinv1[i, i]

                if group_size != 0:
                    if (i1 + i) % group_size == 0:
                        scale, zero, _ = quant_tensor(w[:, (i1 + i):(i1 + i + group_size)], self.w_quant_min,
                                                      self.w_quant_max, self.cfg.weight_narrow_range,
                                                      self.cfg.weight_symmetric, self.cfg.weight_quant_dtype,
                                                      self.weight_quantizer_axis, False, 4)
                        scale = Tensor(scale, w0.dtype)
                        zero = Tensor(zero, w0.dtype)
                    if ((i1 + i) // group_size) - now_idx == -1:
                        self.group_scale.append(scale.T)
                        self.group_zero.append(zero.T)
                        now_idx += 1

                q0 = msops.clip_by_value(aclnn_add(msops.round(w0.unsqueeze(1) / scale), zero),
                                         Tensor(self.weight_quant_min), Tensor(self.weight_quant_max))
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
            w[:, i2:] = aclnn_sub(w[:, i2:], err.matmul(hinv2))
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
        logger.info(f'error: {msops.sum(losses).item()}')

    def _apply_gptq(self, w, scale, zero):
        """apply gptq"""
        self._hessian_compute()
        if self.cfg.algo_args["activation_order"]:
            sort = msops.Sort(descending=True)
            _, perm = sort(numpy.diag(self.h))
            w = w[:, perm]
            self.h = self.h[perm][:, perm]

        cholesky_time = time.time()
        hinv = cholesky_compute(self.h, self.cfg.algo_args["damp_percent"])
        logger.info(f'[TIME]end cholesky part with time {time.time() - cholesky_time}s')
        quant_tick = time.time()
        self._gptq_precision_recovery(w, hinv, scale, zero)
        logger.info(f'[TIME]quant layers with time {time.time() - quant_tick}s')
        if self.cfg.algo_args["activation_order"]:
            w = w[:, perm]
        self.layer.weight = w

    def quant(self):
        """quant"""
        # quant weight
        scale, zp, _ = quant_tensor(self.layer.weight, self.w_quant_min, self.w_quant_max, self.cfg.weight_narrow_range,
                                    self.cfg.weight_symmetric, self.cfg.weight_quant_dtype, self.weight_quantizer_axis,
                                    False, 4)
        self._apply_gptq(self.layer.weight, Tensor(scale, dtype=self.layer.weight.dtype),
                         Tensor(zp, dtype=self.layer.weight.dtype))
        weight = quant_tensor_data(self.layer.weight, self.group_scale, self.group_zero, self.weight_quant_min,
                                   self.weight_quant_max, self.weight_quantizer_axis)
        self.q_weight.set_data(Tensor(weight, dtype=dtype.int8))
        self.w_scale.set_data(Tensor(self.group_scale, dtype=dtype.float64))
        self.w_zp.set_data(Tensor(self.group_zero, dtype=dtype.float64))
        del self.h

    def process(self):
        self.quant()
        # pylint: disable=protected-access
        self.layer.weight._offload()
