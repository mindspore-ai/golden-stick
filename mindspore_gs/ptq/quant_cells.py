# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""ptq quant cells."""
import abc
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops.operations import FakeQuantParam
from mindspore import Parameter, Tensor, dtype
from mindformers.modules.kvcache_mgr import KVCacheMgr

from mindspore_gs.common.gs_enum import BackendTarget
from mindspore_gs.common import logger
from mindspore_gs.quantization.fake_quantizer import LinearFakeQuantizer
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.ptq.convert_utils import convert_to_antiquant, convert_to_quant


class PTQCell(QuantCell):
    """Wrapper Cell to PTQCell with FakeQuantizer"""

    @abc.abstractmethod
    def calibrate(self):
        raise NotImplementedError

    @staticmethod
    def antiquant_strategy(weight_strategy=None):
        """antiquant strategy for w8a16"""
        if weight_strategy is None:
            return None
        strategy_len = len(weight_strategy)
        if strategy_len != 2:
            raise RuntimeError(f'strategy length shall be 2, but got {strategy_len}')
        x_strategy = weight_strategy

        anti_strategy = (x_strategy, (), ())
        return anti_strategy

    @staticmethod
    def antiquant_bmm_strategy(act_strategy, weight_strategy, has_bias=False, has_offset=True, is_transpose=False):
        """parallel strategy for antiquant bmm"""
        if act_strategy is None or weight_strategy is None:
            return None
        if is_transpose:
            scale_strategy = (weight_strategy[0],)
        else:
            scale_strategy = (weight_strategy[1],)
        offset_strategy = scale_strategy
        if not has_bias:
            if has_offset:
                return act_strategy, weight_strategy, scale_strategy, offset_strategy
            return act_strategy, weight_strategy, scale_strategy
        bias_strategy = scale_strategy
        if has_offset:
            return act_strategy, weight_strategy, scale_strategy, offset_strategy, bias_strategy
        return act_strategy, weight_strategy, scale_strategy, bias_strategy



class KVCacheMgrQuant(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, kvcache: KVCacheMgr, policy: LayerPolicy):
        super(KVCacheMgrQuant, self).__init__(kvcache, None)
        self._policy = policy
        self._inputs_insert_fq = self._policy.get_input_need_insert_fq()
        logger.info(f"Create KVCacheMgrQuant for KVCacheMgr {kvcache.key_past.name}")
        self._kvcache = kvcache

        # KVCacheMgr's shape is BNSD currently.
        b = kvcache.max_batch_size
        n = kvcache.n_head
        s = kvcache.max_seq_length
        d = kvcache.head_dim
        # BNSD -> BSND
        self._perm = (0, 2, 1, 3)
        # BSND -> BSH
        self._pre_reshape = (b, s, -1)
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

        self._quantparam_reshape = (1, n, 1, d)

        perchannel_args = PerChannelArgs(n * d, 2, 3)
        self._key_input_quantizer = self._policy.get_input_quantizer(input_index=0, perchannel_args=perchannel_args)
        self._value_input_quantizer = self._policy.get_input_quantizer(input_index=1, perchannel_args=perchannel_args)
        self._weight_quantizer = None
        self._key_output_quantizer = self._policy.get_output_quantizer(perchannel_args=perchannel_args)
        self._value_output_quantizer = self._policy.get_output_quantizer(perchannel_args=perchannel_args)
        prex = ""
        for _, param in kvcache.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        if self._key_input_quantizer:
            self._key_input_quantizer.float_min.data.name = prex + "_key_input_float_min"
            self._key_input_quantizer.float_max.data.name = prex + "_key_input_float_max"
        if self._value_input_quantizer:
            self._value_input_quantizer.float_min.data.name = prex + "_value_input_float_min"
            self._value_input_quantizer.float_max.data.name = prex + "_value_input_float_max"
        if self._key_output_quantizer:
            self._key_output_quantizer.float_min.data.name = prex + "_key_output_float_min"
            self._key_output_quantizer.float_max.data.name = prex + "_key_output_float_max"
        if self._value_output_quantizer:
            self._value_output_quantizer.float_min.data.name = prex + "_value_output_float_min"
            self._value_output_quantizer.float_max.data.name = prex + "_value_output_float_max"
        self._is_deployed = False

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    def _reshape_quant_param(self, fq: FakeQuantParam):
        """Reshape quant param to support broadcast quant."""
        scale = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
        zp = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_zero_point, None)
        if scale:
            t_scale = Tensor(scale, dtype=dtype.float32)
            t_scale: Tensor = t_scale.reshape(self._quantparam_reshape)
            scale = t_scale.asnumpy().tolist()
            fq.attrs[FakeQuantParam.attr_key_linear_quant_scale] = scale
        if zp:
            t_zp = Tensor(zp, dtype=dtype.int32)
            t_zp = t_zp.reshape(self._quantparam_reshape)
            zp = t_zp.asnumpy().tolist()
            fq.attrs[FakeQuantParam.attr_key_linear_quant_zero_point] = zp

    def convert(self, backend: BackendTarget = BackendTarget.NONE, is_deploy=False):
        if backend in (BackendTarget.NONE, BackendTarget.ASCEND):
            if is_deploy:
                if isinstance(self._key_input_quantizer, LinearFakeQuantizer):
                    self._key_input_quantizer.foo_init()
                if isinstance(self._value_input_quantizer, LinearFakeQuantizer):
                    self._value_input_quantizer.foo_init()
                if isinstance(self._key_output_quantizer, LinearFakeQuantizer):
                    self._key_output_quantizer.foo_init()
                if isinstance(self._value_output_quantizer, LinearFakeQuantizer):
                    self._value_output_quantizer.foo_init()
            self._key_input_quantizer = self._key_input_quantizer.convert_to_fakequantparam()
            self._reshape_quant_param(self._key_input_quantizer.fq)
            self._value_input_quantizer = self._value_input_quantizer.convert_to_fakequantparam()
            self._reshape_quant_param(self._value_input_quantizer.fq)
            self._key_output_quantizer = self._key_output_quantizer.convert_to_fakequantparam()
            self._reshape_quant_param(self._key_output_quantizer.fq)
            self._value_output_quantizer = self._value_output_quantizer.convert_to_fakequantparam()
            self._reshape_quant_param(self._value_output_quantizer.fq)
        else:
            raise ValueError("Only support convert KVCacheMgrQuant to GE_ASCEND or MS backend.")
        if backend == BackendTarget.ASCEND:
            key_compute_type = self._kvcache.key_past.dtype
            value_compute_type = self._kvcache.value_past.dtype
            self._key_input_quantizer = convert_to_quant(self._key_input_quantizer)
            self._value_input_quantizer = convert_to_quant(self._value_input_quantizer)
            self._key_output_quantizer = convert_to_antiquant(self._key_output_quantizer, dst_dtype=key_compute_type)
            self._value_output_quantizer = convert_to_antiquant(self._value_output_quantizer,
                                                                dst_dtype=value_compute_type)

            self._kvcache.key_past = Parameter(Tensor(np.zeros(self._kvcache.key_past.shape), dtype=dtype.int8),
                                               name=self._kvcache.key_past.name, requires_grad=False)
            self._kvcache.value_past = Parameter(Tensor(np.zeros(self._kvcache.value_past.shape), dtype=dtype.int8),
                                                 name=self._kvcache.value_past.name, requires_grad=False)
            self._is_deployed = True

    def calibrate(self):
        pass

    # pylint: disable=W0221
    def construct(self, key, value, kvcache_inputs=None):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        if self._is_deployed:
            key = self._key_input_quantizer(key)
            value = self._value_input_quantizer(value)
        kcache, vcache = self._kvcache(key, value, kvcache_inputs)
        if self._is_deployed:
            kcache = self._key_output_quantizer(kcache)
            vcache = self._value_output_quantizer(vcache)
        return kcache, vcache
