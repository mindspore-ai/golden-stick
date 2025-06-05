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
import os
import math
import time
from typing import Optional
from mindspore import Tensor, dtype, numpy, Parameter
from mindspore import ops as msops
from mindspore.common.initializer import initializer
from mindspore.communication import get_rank
from mindspore.communication.management import GlobalComm
from mindspore.ops import sub as aclnn_sub, add as aclnn_add
from mindformers.modules.layers import Linear
from mindformers.experimental.infer.core.layers import ColumnParallelLinear, RowParallelLinear
from mindspore_gs.common import logger
from mindspore_gs.common.json_cache import JSONCache
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
        try:
            from research.deepseek3.moe import (ColumnParallelGroupLinear, RowParallelGroupLinear,
                                                ColumnParallelLinearWorldRegion, RowParallelLinearWorldRegion)
            Quantizer.reg_layer_map(ColumnParallelGroupLinear, GptqWeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(RowParallelGroupLinear, GptqWeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(ColumnParallelLinearWorldRegion, GptqWeightQuantLinearCell, A16WxChecker())
            Quantizer.reg_layer_map(RowParallelLinearWorldRegion, GptqWeightQuantLinearCell, A16WxChecker())
        except ImportError:
            pass

    def __init__(self, linear_name, linear, context, cfg: InnerPTQConfig, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        if linear.expert_num and linear.expert_num > 1:
            self.is_moe = True
        else:
            self.is_moe = False
        self.linear_name = linear_name.split('.')[-1] if not self.is_moe else 'moe|' + linear_name.split('.')[-1]
        if context.algorithm_cache_path:
            self.enable_cache = True
            cache_file_path = os.path.join(context.algorithm_cache_path, f'rank_{context.rank_id}', 'gptq.json')
        else:
            self.enable_cache = False
            cache_file_path = ''
        self.cache: Optional[JSONCache] = JSONCache(cache_file_path)
        self.bits = 4 if self.cfg.weight_quant_dtype == dtype.qint4x2 else 8
        self.nsamples = 0
        self.cfg.reflash_inputs_after_each_processor = True
        self.group_scale = []
        self.group_zero = []
        self.qweight = []
        self.scale_rank_size = self.w_scale.shape[0] if len(linear.weight.shape) == 2 else self.w_scale.shape[1]
        if self.cfg.tp_size > 1:
            self.rank_id = get_rank()
        if self.parallel_type == ParallelType.ROW_PARALLEL and self.cfg.tp_size > 1:
            self.weight_need_allgather = True
        else:
            self.weight_need_allgather = False
        if self.weight_need_allgather:
            self.h = msops.zeros((self.ic * self.cfg.tp_size, self.ic * self.cfg.tp_size), dtype=dtype.float32)
        else:
            self.h = msops.zeros((self.ic, self.ic), dtype=dtype.float32)
        self.perm = Parameter(initializer('ones', self.ic, dtype=dtype.int32))
        self.weight_quant_min, self.weight_quant_max = get_quant_min_max(num_bits=self.bits,
                                                                         signed=self.cfg.weight_symmetric,
                                                                         narrow_range=self.cfg.weight_narrow_range)

    def _quant_info(self):
        if self.cfg.weight_quant_dtype == dtype.int8:
            return f'GPTQ-W8-{str(self.cfg.weight_quant_granularity)}'
        if self.cfg.weight_quant_dtype == dtype.qint4x2:
            return f'GPTQ-W4-{str(self.cfg.weight_quant_granularity)}'
        raise RuntimeError(f"Unexpected weight_quant_dtype: {self.cfg.weight_quant_dtype}.")

    def _hessian_compute(self):
        """compute Hessian Matrix"""
        for i in range(len(self.samples)):
            if len(self.samples[i].shape) == 1 or len(self.samples[i].shape) == 3:
                self.samples[i] = self.samples[i].reshape((-1, self.samples[i].shape[-1]))
            sqe = self.nsamples / (self.nsamples + 1)
            self.nsamples += 1
            sqr = math.sqrt(2 / self.nsamples)
            self.samples[i] = sqr * self.samples[i]
            self.h *= sqe
            if self.weight_need_allgather:
                inp = msops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)(self.samples[i].transpose(1, 0))
                inp = Tensor(inp.transpose(1, 0), dtype=dtype.float32)
                self.h += msops.matmul(inp.transpose(1, 0), inp)
            else:
                samples = self.samples[i].astype(dtype.float32)
                self.h += msops.matmul(samples.transpose(1, 0), samples)
        self.cfg.dumper.dump_data(self.layer_name, "|hessian_matrix|input0_activation_inputs",
                                  msops.cat(tuple(self.samples), axis=0))
        self.cfg.dumper.dump_data(self.layer_name, "|hessian_matrix|output0_hessian", self.h)
        dead = numpy.diag(self.h) == 0
        dead = msops.nonzero(dead)
        if dead.shape[0] > 0:
            self.h[dead, dead] = 1
        perm = []
        invperm = []
        if self.cfg.algo_args["desc_act"]:
            perm = msops.argsort(numpy.diag(self.h), descending=True)
            self.h = self.h[perm][:, perm]
            invperm = msops.argsort(perm)
        from mindspore_gs.ptq.cholesky_trans import cholesky_compute
        cholesky_time = time.time()
        hinv = cholesky_compute(self.h, self.cfg.algo_args["damp_percent"])
        logger.info(f'[TIME]end cholesky part with time {time.time() - cholesky_time}s')
        self.cfg.dumper.dump_data(self.layer_name, "|cholesky_decomposition|input0_hessian", self.h)
        self.cfg.dumper.dump_data(self.layer_name, "|cholesky_decomposition|output0_inv_hessian", hinv)
        self.samples.clear()
        self.h = None
        return dead, perm, invperm, hinv

    def _gptq_precision_recovery(self, weight, hinv, scale, zero, perm):
        """precision recovery use gptq"""
        group_size = self.cfg.group_size
        losses = 0
        now_idx = 1
        for i1 in range(0, weight.shape[1], self.cfg.algo_args["block_size"]):
            i2 = min(i1 + self.cfg.algo_args["block_size"], weight.shape[1])
            count = i2 - i1
            w1 = weight[:, i1:i2]
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
                                                          self.cfg.weight_quant_dtype, 0, False)
                            scale = Tensor(scale, dtype.float32)
                            zero = Tensor(zero, dtype.float32)
                        if ((i1 + i) // group_size) - now_idx == -1:
                            self.group_scale.append(scale.transpose(1, 0))
                            self.group_zero.append(zero.transpose(1, 0))
                            now_idx += 1
                    else:
                        idx = i1
                        if self.cfg.algo_args["desc_act"]:
                            idx = perm[idx]
                        scale = self.group_scale[idx // group_size].transpose(1, 0)
                        zero = self.group_zero[idx // group_size].transpose(1, 0)
                q0 = msops.clip_by_value(aclnn_add(msops.round(w0.unsqueeze(1) / scale), zero),
                                         Tensor(self.weight_quant_min), Tensor(self.weight_quant_max))
                self.qweight.append(q0)
                q0 = scale * aclnn_sub(q0, zero)
                q0 = q0.flatten()
                delta_loss = aclnn_sub(w0, q0) ** 2 / d ** 2
                err1 = aclnn_sub(w0, q0) / d
                delta_w = err1.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                losses1[:, i] = delta_loss
                w1[:, i:] = aclnn_sub(w1[:, i:], delta_w)
                err[:, i] = err1

            weight[:, i2:] = aclnn_sub(weight[:, i2:], err.matmul(hinv2))
            losses += msops.sum(losses1 / 2)
        if group_size == 0:
            self.group_scale.append(scale)
            self.group_zero.append(zero)
            self.group_scale = msops.cat(self.group_scale)
            self.group_zero = msops.cat(self.group_zero)
            self.group_scale = msops.squeeze(self.group_scale) if not self.is_moe else self.group_scale
            self.group_zero = msops.squeeze(self.group_zero) if not self.is_moe else self.group_zero
        else:
            self.group_scale = msops.cat(self.group_scale)
            self.group_zero = msops.cat(self.group_zero)
        self.qweight = msops.cat(self.qweight, 1)
        logger.info(f'error: {losses}')
        if self.weight_need_allgather and group_size != 0:
            self.group_scale = self.group_scale[self.rank_id * self.scale_rank_size :
                                                self.rank_id * self.scale_rank_size + self.scale_rank_size, :]
            self.group_zero = self.group_zero[self.rank_id * self.scale_rank_size :
                                              self.rank_id * self.scale_rank_size + self.scale_rank_size, :]

    def _apply_gptq(self, weight, scale, zero, dead, perm, invperm, hinv):
        """apply gptq"""
        if dead.shape[0] > 0:
            weight[:, dead] = 0
        if self.cfg.algo_args["desc_act"]:
            weight = weight[:, perm]
        if self.cfg.algo_args["static_groups"] and self.cfg.group_size != 0:
            for i in range(0, weight.shape[1], self.cfg.group_size):
                scale, zero, _ = quant_tensor(weight[:, i : i + self.cfg.group_size], self.w_quant_min,
                                              self.w_quant_max, self.cfg.weight_narrow_range, self.cfg.weight_symmetric,
                                              False, 0, self.cfg.weight_quant_dtype,
                                              self.weight_quantizer_axis, False)
                self.group_scale.append(Tensor(scale, dtype.float32).T)
                self.group_zero.append(Tensor(zero, dtype.float32).T)
        quant_tick = time.time()
        self._gptq_precision_recovery(weight, hinv, scale, zero, perm)
        logger.info(f'[TIME]quant layers with time {time.time() - quant_tick}s')
        if self.cfg.algo_args["desc_act"] and self.weight_need_allgather:
            self.qweight = self.qweight[:, invperm]
        if self.weight_need_allgather:
            self.qweight = self.qweight[:, self.rank_id * self.ic : self.rank_id * self.ic + self.ic]

    def _qweight_reshape(self, qweight, scale, zero, weight_shape, scale_shape):
        """reshape quant weight and quant params"""
        qweight = qweight.reshape(weight_shape)
        qweight = msops.transpose(qweight, (0, 2, 1))
        if self.cfg.group_size != 0:
            scale = scale.transpose(1, 0)
            zero = zero.transpose(1, 0)
            scale = scale.reshape(scale_shape)
            zero = zero.reshape(scale_shape)
            scale = msops.transpose(scale, (0, 2, 1))
            zero = msops.transpose(zero, (0, 2, 1))
        else:
            scale = scale.reshape(scale_shape)
            zero = zero.reshape(scale_shape)
        return qweight, scale, zero

    def _get_quant_params(self, weight):
        """get quant params, scale and zp"""
        if self.weight_need_allgather:
            weight = msops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)(msops.transpose(weight, (1, 0)))
            weight = msops.transpose(weight, (1, 0))
        if self.is_moe:
            weight = weight.reshape(self.layer.weight.shape[0], -1, weight.shape[-1])
            weight = msops.transpose(weight, (0, 2, 1))
        scale, zp, _ = quant_tensor(weight, self.w_quant_min, self.w_quant_max,
                                    self.cfg.weight_narrow_range, self.cfg.weight_symmetric,
                                    self.cfg.weight_quant_granularity == QuantGranularity.PER_GROUP,
                                    self.cfg.group_size, self.cfg.weight_quant_dtype,
                                    self.weight_quantizer_axis, False, False, self.layer.transpose_b, False)
        return weight, scale, zp

    def quant(self):
        """quant"""
        weight = self.layer.weight.value()
        dead, perm, invperm, hinv = self._hessian_compute()
        if not self.is_moe:
            weight, scale, zp = self._get_quant_params(weight)
            self._apply_gptq(weight, scale, zp, dead, perm, invperm, hinv)
        else:
            weight = msops.transpose(weight, (0, 2, 1))
            weight_shape = weight.shape
            weight = weight.reshape(-1, weight.shape[-1])
            _, scale, zp = self._get_quant_params(weight)
            scale = msops.transpose(scale, (0, 2, 1))
            scale_shape = scale.shape
            scale = scale.reshape(-1, weight_shape[1])
            zp = msops.transpose(zp, (0, 2, 1))
            zp = zp.reshape(-1, weight_shape[1])
            if self.weight_need_allgather:
                block_size = 32 * weight_shape[1]
                gather_qweight = []
                gather_scale = []
                gather_zero = []
                for i in range(0, weight.shape[0], block_size):
                    split_weight = weight[i : i + block_size, :]
                    split_weight = msops.AllGather(group=GlobalComm.WORLD_COMM_GROUP)(split_weight.transpose(1, 0))
                    split_weight = split_weight.transpose(1, 0)
                    split_scale = scale[i : i + block_size, :] if self.cfg.group_size == 0 else scale
                    split_zero = zp[i : i + block_size, :] if self.cfg.group_size == 0 else zp
                    self._apply_gptq(split_weight, split_scale, split_zero, dead, perm, invperm, hinv)
                    gather_qweight.append(self.qweight)
                    gather_scale.append(self.group_scale)
                    gather_zero.append(self.group_zero)
                    self.qweight = []
                    self.group_scale = []
                    self.group_zero = []
                gather_qweight = msops.cat(gather_qweight)
                gather_scale = msops.cat(gather_scale, 1)
                gather_zero = msops.cat(gather_zero, 1)
                scale_shape = (scale_shape[0], scale_shape[1], scale_shape[-1] // self.cfg.tp_size)
                self.qweight, self.group_scale, self.group_zero = self._qweight_reshape(gather_qweight, gather_scale,
                                                                                        gather_zero, weight_shape,
                                                                                        scale_shape)
                del gather_qweight
                del gather_scale
                del gather_zero
            else:
                self._apply_gptq(weight, scale, zp, dead, perm, invperm, hinv)
                self.qweight, self.group_scale, self.group_zero = self._qweight_reshape(self.qweight, self.group_scale,
                                                                                        self.group_zero, weight_shape,
                                                                                        scale_shape)
        self.q_weight.set_data(Tensor(self.qweight.asnumpy(), dtype=dtype.int8))
        self.w_scale.set_data(Tensor(self.group_scale, dtype=dtype.float64))
        self.w_zp.set_data(Tensor(self.group_zero, dtype=dtype.float64))
        self.perm.set_data(Tensor(perm, dtype=dtype.int32))
        del self.group_scale
        del self.group_zero
        del self.qweight

    def _cache_key(self):
        qweight_name = self.layer_name + '.qweight'
        scale_name = self.layer_name + '.scale'
        zero_name = self.layer_name + '.zero'
        return qweight_name, scale_name, zero_name

    def process(self):
        qweight_name, scale_name, zero_name = self._cache_key()
        scale = self.cache.get(scale_name)
        zero = self.cache.get(zero_name)
        qweight = self.cache.get(qweight_name)
        if scale is not None and zero is not None and qweight is not None:
            logger.info(f'layer {self.layer_name} use cache for qweight, scale and zero.')
            self.q_weight.set_data(Tensor(qweight, dtype=dtype.int8))
            self.w_scale.set_data(Tensor(scale, dtype=dtype.float64))
            self.w_zp.set_data(Tensor(zero, dtype=dtype.float64))
        else:
            self.quant()
            if self.enable_cache:
                self.cache.put(qweight_name, self.q_weight.asnumpy())
                self.cache.put(scale_name, self.w_scale.asnumpy())
                self.cache.put(zero_name, self.w_zp.asnumpy())
        # pylint: disable=protected-access
        self.layer.weight._offload()
