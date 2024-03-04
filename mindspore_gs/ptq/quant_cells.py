# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations import FakeQuantParam
from mindspore import log as logger
from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer
from mindspore_gs.common.gs_enum import BackendTarget
from mindspore_gs.quantization.fake_quantizer import LinearFakeQuantizer
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.quant_utils import get_quant_min_max, quant_tensor_data
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.ptq.convert_utils import (
    convert_to_antiquant, convert_to_fusion_antiquant, convert_to_quant, convert_to_dequant
)
from mindformers.modules import Linear
from mindformers.modules import KVCacheMgr


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
    def antiquant_bmm_strategy(act_strategy,
                               weight_strategy,
                               has_bias=False,
                               is_transpose=False):
        """parallel strategy for antiquant bmm"""
        if act_strategy is None or weight_strategy is None:
            return None
        if is_transpose:
            scale_strategy = (weight_strategy[0],)
        else:
            scale_strategy = (weight_strategy[1],)
        offset_strategy = scale_strategy
        if not has_bias:
            return act_strategy, weight_strategy, scale_strategy, offset_strategy
        bias_strategy = scale_strategy
        return act_strategy, weight_strategy, scale_strategy, offset_strategy, bias_strategy


class LinearQuant(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy: LayerPolicy):
        super(LinearQuant, self).__init__(linear, policy)
        self._linear = linear
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        input_fq_args = {}
        weight_perchannel_args = PerChannelArgs(self._linear.out_channels, self._weight_axis, rank)
        weight_fq_args = {}
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            self._act_strategy = self._linear.matmul.in_strategy[0]
            self._weight_strategy = self._linear.matmul.in_strategy[1]
            input_fq_args["strategy"] = (self._linear.matmul.in_strategy[0],)
            weight_fq_args["strategy"] = (self._weight_strategy,)
        self._input_quantizer = self._policy.get_input_quantizer(input_index=0, **input_fq_args)
        self._output_quantizer = None
        self._weight_quantizer = self._policy.get_weight_quantizer(self._linear.weight.name, weight_perchannel_args,
                                                                   **weight_fq_args)

        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        if self._input_quantizer:
            self._input_quantizer.float_min.data.name = prex + "_input_float_min"
            self._input_quantizer.float_max.data.name = prex + "_input_float_max"
        self._weight_quantizer.float_min.data.name = prex + "_weight_float_min"
        self._weight_quantizer.float_max.data.name = prex + "_weight_float_max"

        has_dtype = hasattr(self._linear, "dtype")
        self._cast_dtype = self._linear.dtype if has_dtype else self._linear.weight.dtype
        self._quant_deployed = False

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    def convert(self, backend: str = BackendTarget.NONE.value, is_deploy=False):
        if backend == BackendTarget.NONE.value:
            super(LinearQuant, self).convert(backend)
            if self._weight_quantizer:
                self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            return
        if backend == BackendTarget.ASCEND.value:
            weight_only = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                          self._weight_quantizer.get_attr("weight_only_quant", False)
            all_quant = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                        isinstance(self._input_quantizer, LinearFakeQuantizer)
            if not all_quant and not weight_only:
                logger.info(f"LinearQuant {self} is not quanted.")
                return
            if is_deploy:
                if isinstance(self._input_quantizer, LinearFakeQuantizer):
                    self._input_quantizer.foo_init()
                if isinstance(self._weight_quantizer, LinearFakeQuantizer):
                    self._weight_quantizer.foo_init()
            super(LinearQuant, self).convert(backend)
            self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            # quant weight to int8
            if not is_deploy:
                weight_quantizer: P.FakeQuantParam = self._weight_quantizer.fq
                weight = self._linear.cast(self._linear.weight, self._cast_dtype)
                quant_min, quant_max = get_quant_min_max(
                    weight_quantizer.attrs[LinearFakeQuantizer.attr_key_num_bits],
                    weight_quantizer.attrs[LinearFakeQuantizer.attr_key_symmetric],
                    weight_quantizer.attrs[LinearFakeQuantizer.attr_key_narrow_range])
                scale = weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_scale]
                zp = weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_zero_point]
                weight_quant = quant_tensor_data(weight, np.squeeze(np.array(scale)), np.squeeze(np.array(zp)),
                                                 quant_min, quant_max, self._weight_axis, dtype.int8)
                self._linear.weight = Parameter(Tensor(weight_quant, dtype=dtype.int8),
                                                name=self._linear.weight.name)
            else:
                self._linear.weight = Parameter(initializer('ones', self._linear.weight.shape, dtype.int8),
                                                name=self._linear.weight.name)
            # convert to ascend quant layer
            if all_quant:
                self._output_quantizer = convert_to_dequant(self._input_quantizer, self._weight_quantizer)
                self._input_quantizer = convert_to_quant(self._input_quantizer)
                self._quant_deployed = True
            else:
                self._input_quantizer = None
                self._output_quantizer = None
                self._weight_quantizer = convert_to_fusion_antiquant(
                    self._weight_quantizer, transpose_weight=self._linear.transpose_b,
                    dst_dtype=self._cast_dtype, strategy=self.antiquant_bmm_strategy(self._act_strategy,
                                                                                     self._weight_strategy,
                                                                                     False,
                                                                                     self._linear.transpose_b)
                )
                self._quant_deployed = True

    def calibrate(self):
        logger.info(f"Calibrating weight of Linear Cell: {self._linear.weight.name}")
        self._weight_quantizer(self._linear.weight)

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = P.Shape()(x)[:-1] + (self._linear.out_channels,)
        x = P.Reshape()(x, (-1, self._linear.in_channels))
        if hasattr(self._linear, "expert_flag") and self._linear.expert_flag:
            if self._linear.use_expert_group_size is True:
                x = P.Reshape()(x, (-1, self._linear.expert_num, self._linear.expert_group_size,
                                    self._linear.in_channels))
            else:
                x = P.Reshape()(x, (self._linear.outer_batch, self._linear.expert_num, -1, self._linear.in_channels))
        ori_dtype = F.dtype(x)

        bias = None
        if self._linear.has_bias:
            if hasattr(self._linear, "dtype"):
                bias = self._linear.cast(self._linear.bias, self._linear.dtype)
            else:
                bias = self._linear.cast(self._linear.bias, x.dtype)
        x = self._linear.cast(x, self._cast_dtype)
        if self._quant_deployed:
            x = self._weight_quantizer(x, self._linear.weight)
        else:
            weight = self._linear.cast(self._linear.weight, self._cast_dtype)
            x = self._linear.matmul(x, weight)
        if self._linear.has_bias:
            x = self._linear.bias_add(x, bias)
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output


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

    def convert(self, backend: BackendTarget = BackendTarget.NONE.value, is_deploy=False):
        if backend in (BackendTarget.NONE.value, BackendTarget.ASCEND.value):
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
        if backend == BackendTarget.ASCEND.value:
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
        logger.info(f"----------- Calibrating key buffer of KVCache Cell: {self._kvcache.key_past.name}")
        key = self.transpose(self._kvcache.key_past, self._perm)
        key = self.reshape(key, self._pre_reshape)
        self._key_input_quantizer(key)
        self._key_output_quantizer(key)
        logger.info(f"----------- Calibrating value buffer of KVCache Cell: {self._kvcache.value_past.name}")
        value = self.transpose(self._kvcache.value_past, self._perm)
        value = self.reshape(value, self._pre_reshape)
        self._value_input_quantizer(value)
        self._value_output_quantizer(value)

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
