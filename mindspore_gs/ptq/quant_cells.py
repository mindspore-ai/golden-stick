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
import time
import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations import FakeQuantParam
from mindspore import log as logger
from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer
from mindformers.modules.layers import Linear
from mindformers.modules.kvcache_mgr import KVCacheMgr

from mindspore_gs.common.gs_enum import BackendTarget
from mindspore_gs.quantization.fake_quantizer import LinearFakeQuantizer
from mindspore_gs.quantization.quant_cell import QuantCell
from mindspore_gs.quantization.quant_utils import get_quant_min_max, quant_tensor_data, quant_bias_data
from mindspore_gs.quantization.layer_policy import LayerPolicy, PerChannelArgs
from mindspore_gs.ptq.ptq_config import PTQMode
from mindspore_gs.ptq.convert_utils import (
    convert_to_antiquant, convert_to_fusion_antiquant, convert_to_quant, convert_to_dequant,
    convert_to_dequant_bmm, convert_to_fusion_antiquant_for_deploy
)


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
    def antiquant_bmm_strategy(act_strategy, weight_strategy, has_bias=False, is_transpose=False):
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

    def convert(self, backend: str = BackendTarget.NONE, is_deploy=False):
        if backend == BackendTarget.NONE:
            super(LinearQuant, self).convert(backend)
            if self._weight_quantizer:
                self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            return
        if backend == BackendTarget.ASCEND:
            weight_only = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                        self._weight_quantizer.get_attr("weight_only_quant", False)
            all_quant = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                        isinstance(self._input_quantizer, LinearFakeQuantizer)
            if not all_quant and not weight_only:
                logger.info(f"LinearQuant {self} is not quanted.")
                return

            super(LinearQuant, self).convert(backend)
            # quant weight to int8
            if not is_deploy:
                self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
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
                np_weight_quant = weight_quant.asnumpy()
                del weight_quant
                self._linear.weight = Parameter(Tensor(np_weight_quant, dtype=dtype.int8),
                                                name=self._linear.weight.name)
                if not all_quant:
                    self._input_quantizer = None
                    self._output_quantizer = None
                    self._weight_quantizer = convert_to_fusion_antiquant(
                        self._weight_quantizer, transpose_weight=self._linear.transpose_b,
                        dst_dtype=self._cast_dtype, strategy=
                        self.antiquant_bmm_strategy(self._act_strategy, self._weight_strategy,
                                                    False, self._linear.transpose_b)
                    )
            else:
                if isinstance(self._input_quantizer, LinearFakeQuantizer):
                    self._input_quantizer.foo_init()
                if isinstance(self._weight_quantizer, LinearFakeQuantizer):
                    self._weight_quantizer.foo_init()
                self._linear.weight = Parameter(initializer('ones', self._linear.weight.shape, dtype.int8),
                                                name=self._linear.weight.name)
                if not all_quant:
                    self._input_quantizer = None
                    self._output_quantizer = None
                    self._weight_quantizer = convert_to_fusion_antiquant_for_deploy(
                        axis=self._weight_axis, output_channel=self._linear.out_channels,
                        data_rank=len(self._linear.weight.shape),
                        is_per_channel=self._weight_quantizer.is_per_channel(),
                        transpose_weight=self._linear.transpose_b,
                        dst_dtype=self._cast_dtype,
                        strategy=self.antiquant_bmm_strategy(self._act_strategy, self._weight_strategy,
                                                             False, self._linear.transpose_b)
                    )
            if all_quant:
                self._output_quantizer = convert_to_dequant(self._input_quantizer, self._weight_quantizer)
                self._input_quantizer = convert_to_quant(self._input_quantizer)
                self._quant_deployed = True
                raise RuntimeError(f'current version not support all quantization, only for weight quantization')
            self._quant_deployed = True

    def calibrate(self):
        """calibrate for weight quant"""
        start = time.time()
        self._weight_quantizer(self._linear.weight)
        logger.info(
            f"Calibrated weight of Linear Cell: {self._linear.weight.name}, time cost: {time.time() - start} s.")

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

        x = self._linear.cast(x, self._cast_dtype)
        if self._quant_deployed:
            x = self._weight_quantizer(x, self._linear.weight)
        else:
            weight = self._linear.cast(self._linear.weight, self._cast_dtype)
            weight = self._weight_quantizer(weight)
            x = self._linear.matmul(x, weight)
        if self._linear.has_bias:
            x = self._linear.bias_add(x, self._linear.cast(self._linear.bias, self._linear.dtype))
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


class SQLinearWrapper(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy=None, cfg=None):
        super().__init__(linear, policy)
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but got {type(linear)}')
        self._linear = linear
        self._cfg = cfg
        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        input_fq_args = {}
        weight_perchannel_args = PerChannelArgs(self._linear.out_channels,
                                                self._weight_axis,
                                                rank)
        weight_fq_args = {}
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            input_fq_args["strategy"] = (self._linear.matmul.in_strategy[0],)
            weight_fq_args["strategy"] = (self._linear.matmul.in_strategy[1],)

        self._input_quantizer = policy.get_input_quantizer(**input_fq_args)
        self._weight_quantizer = policy.get_weight_quantizer(self._linear.weight.name, weight_perchannel_args,
                                                             **weight_fq_args)
        act_rank = 2
        self._act_observer = policy.create_observer_perchannel(
            perchannel_args=PerChannelArgs(self._linear.in_channels, -1, act_rank), **input_fq_args)
        weight_observer_axis = 1 if linear.matmul.transpose_b else 0
        self._weight_in_observer = policy.create_observer_perchannel(
            perchannel_args=PerChannelArgs(self._linear.in_channels, weight_observer_axis, rank), **weight_fq_args)
        self._output_quantizer = None
        prex = ""
        for _, param in linear.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        if self._input_quantizer:
            self._input_quantizer.float_min.data.name = prex + "_input_float_min"
            self._input_quantizer.float_max.data.name = prex + "_input_float_max"
        self._weight_quantizer.float_min.data.name = prex + "_weight_float_min"
        self._weight_quantizer.float_max.data.name = prex + "_weight_float_max"
        self._act_observer.float_min.data.name = prex + "_act_observer_float_min"
        self._act_observer.float_max.data.name = prex + "_act_observer_float_max"
        self._weight_in_observer.float_min.data.name = prex + "_weight_in_observer_float_min"
        self._weight_in_observer.float_max.data.name = prex + "_weight_in_observer_float_max"
        self._input_scale = Parameter(Tensor([1.0] * self._linear.in_channels), name=f'{prex}_input_scale')
        self._scale_store = Parameter(Tensor([1.0] * self._linear.in_channels), name=f'{prex}_scale_store')
        mode = self._cfg.mode
        self._is_deploy = mode == PTQMode.DEPLOY
        self._alpha = self._cfg.algo_args.get('alpha', None)
        if self._alpha is None:
            self._alpha = 0.5
            # raise ValueError(f'Shall input alpha in smooth quant args, but is None')
        if self._is_deploy:
            self._linear.weight = Parameter(initializer("ones", self._linear.weight.shape, dtype.int8),
                                            name=self._linear.weight.name)
            if self._linear.has_bias:
                self._linear.bias = Parameter(initializer("ones", self._linear.bias.shape, dtype.int32),
                                              name=self._linear.bias.name)
        self._expand = P.ExpandDims()
        self._act_mul = P.Mul()
        self._weight_mul = P.Mul()
        self._weight_div = P.Div()
        self._smooth_act_maximum = P.Maximum()
        self._smooth_act_abs = P.Abs()
        self._act_pow = P.Pow()
        self._smooth_weight_maximum = P.Maximum()
        self._smooth_weight_abs = P.Abs()
        self._weight_pow = P.Pow()
        self._pow_div = P.Div()
        self._div = P.Div()
        self._assign = P.Assign()
        self._weight_assign = P.Assign()
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            self.shard()

    def _get_bias_reduce_num(self):
        """
        1. matmul may have four kind of in_strategy: (1,m)(m,1);(m,1)(1,1);(1,1)(1,m);(1,1)(1,1). We can find that
         (1,m)(m,1) will add allreduce after matmul, (m,1)(1,1) and (1,1)(1,m) will add allgather after matmul.
         (1,1)(1,1) will not add any operation after matmul.
        2. We can simplify Linear construct to matmul + bias + act, and quant-linear construct to matmulint8 + bias +
         dequant + act. In allreduce-parallel mode, allreduce should insert as matmulint8 + allreduce + bias + dequant +
         act, while in allgather-parallel mode: matmulint8 + allgather + bias + dequant + act or
         matmulint8 + bias + allgather + dequant + act or matmulint8 + bias + dequant + allgather + act.
        3. If matmulint8 + bias + dequant use fused kernel QuantBatchMatmul, in allreduce-paralle mode, allreduce can
         not be inserted between matmul and bias anymore, so bias will act on more than one time. To correct this issue,
         we can move bias out of fused kernel or divide value in bias by x. This function is designed to find out the x.
        """

        if "in_strategy" not in self._linear.matmul.get_attr_dict():
            return 1
        if self._linear.matmul.in_strategy is None:
            return 1
        act_strategy = self._linear.matmul.in_strategy[0]
        weight_strategy = self._linear.matmul.in_strategy[1]
        weight_strategy_0 = weight_strategy[1] if self._linear.transpose_b else weight_strategy[0]
        weight_strategy_1 = weight_strategy[0] if self._linear.transpose_b else weight_strategy[1]
        # allreduce
        if act_strategy[0] == 1 and act_strategy[1] != 1 and weight_strategy_0 != 1 and weight_strategy_1 == 1:
            if act_strategy[1] != weight_strategy_0:
                raise RuntimeError(f"Invalid in_strategy for matmul: {self._linear.matmul.in_strategy}.")
            return act_strategy[1]
        # allgather or no-parallel
        if act_strategy[1] == 1 and weight_strategy_0 == 1:
            return 1
        raise RuntimeError(f"Invalid in_strategy for matmul: {self._linear.matmul.in_strategy}.")

    def shard(self):
        """
        shard.
        should consider out_strategy.
        """
        self._act_strategy = self._linear.matmul.in_strategy[0]
        self._weight_strategy = self._linear.matmul.in_strategy[1]
        mul_strategy = (self._act_strategy[1],)
        weight_in_strategy = self._weight_in_strategy(self._weight_strategy, self._linear.transpose_b)
        weight_out_strategy = self._weight_out_strategy(self._weight_strategy, self._linear.transpose_b)
        # activation * smooth_scale(channel_in)
        self._act_mul.shard((self._act_strategy, mul_strategy))
        # weight * smooth_scale(weight_channel_in)
        if self._linear.transpose_b:
            self._weight_mul.shard((self._weight_strategy, weight_in_strategy))
            self._weight_div.shard((self._weight_strategy, weight_in_strategy))
        else:
            self._weight_mul.shard((self._weight_strategy, (weight_in_strategy[0], 1)))
            self._weight_div.shard((self._weight_strategy, (weight_in_strategy[0], 1)))
        # act observer pow, activation in
        self._smooth_act_maximum.shard((mul_strategy, mul_strategy))
        self._smooth_act_abs.shard((mul_strategy,))
        self._act_pow.shard((mul_strategy, ()))
        # weight observer pow, weight channel in
        self._smooth_weight_maximum.shard((weight_in_strategy, weight_in_strategy))
        self._smooth_weight_abs.shard((weight_in_strategy,))
        self._weight_pow.shard((weight_in_strategy, ()))
        # act_max_pow / weight_max_pow
        self._pow_div.shard((mul_strategy, weight_in_strategy))
        # 1 / smooth_scale
        self._div.shard(((), weight_in_strategy))
        # store_scale assign to smooth scale
        self._assign.shard((mul_strategy, mul_strategy))
        # new weight assign to linear weight
        self._weight_assign.shard((self._weight_strategy, self._weight_strategy))
        # bias add strategy: activation index 0 to weight channel out, bias: weight channel out
        if self._linear.has_bias:
            self._linear.bias_add.shard(((self._act_strategy[0], weight_out_strategy[0]), weight_out_strategy))

    @staticmethod
    def _weight_in_strategy(strategy, is_transpose):
        if is_transpose:
            return (strategy[1],)
        return (strategy[0],)

    @staticmethod
    def _weight_out_strategy(strategy, is_transpose):
        if is_transpose:
            return (strategy[0],)
        return (strategy[1],)

    def _create_scale_param(self, rank, name):
        """create_scale_param"""
        bmm_rank = 3
        if rank == bmm_rank:
            param = Parameter(Tensor([[[1.0] * self._linear.in_channels]], dtype=dtype.float32), name=name)
        else:
            param = Parameter(Tensor([[1.0] * self._linear.in_channels], dtype=dtype.float32), name=name)
        return param

    def _infer_scale_strategy(self, weight_strategy):
        """infer_scale_strategy"""
        scale_weight_strategy = (1,)
        rank = len(weight_strategy)
        if rank < 2:
            raise RuntimeError(f'weight_startegy length is less than 2, please check!')
        w_strategy = weight_strategy
        bmm_rank = 3
        if rank == bmm_rank:
            w_strategy = weight_strategy[1:]
        if self._linear.transpose_b:
            if w_strategy[1] > 1:
                scale_weight_strategy += (w_strategy[1],)
            else:
                scale_weight_strategy += (1,)
        else:
            if w_strategy[0] > 1:
                scale_weight_strategy = (w_strategy[0],) + scale_weight_strategy
            else:
                scale_weight_strategy += (1,)
        if rank == bmm_rank:
            scale_weight_strategy = (1,) + scale_weight_strategy
        return scale_weight_strategy

    def _calc_input_scale(self):
        """calc_input_scale"""
        act_max = self._smooth_act_maximum(self._smooth_act_abs(self._act_observer.float_max),
                                           self._smooth_act_abs(self._act_observer.float_min))
        input_max_pow = self._act_pow(act_max, self._alpha)
        weight_max = self._smooth_weight_maximum(self._smooth_weight_abs(self._weight_in_observer.float_max),
                                                 self._smooth_weight_abs(self._weight_in_observer.float_min))
        weight_max_pow = self._weight_pow(weight_max, 1 - self._alpha)
        input_scale = self._pow_div(input_max_pow, weight_max_pow).clip(1e-5)

        # set 0 or nan to 1.0 to avoid quantization error
        input_scale[input_max_pow == 0] = 1.0
        input_scale[weight_max_pow == 0] = 1.0
        return input_scale

    def _adjust_parameter(self):
        self._assign(self._input_scale, self._scale_store.data)
        weight_scale = self._expand(self._input_scale, 0)
        if not self._linear.transpose_b:
            weight_scale = weight_scale.transpose()
        orin_dtype = self._linear.weight.dtype
        weight = self._weight_mul(self._linear.weight, weight_scale)
        weight = self._linear.cast(weight, orin_dtype)
        self._weight_assign(self._linear.weight, weight)

    def weight_quantizer(self):
        return self._weight_quantizer

    def convert(self, backend: BackendTarget = BackendTarget.NONE, is_deploy=False):
        if not self._is_deploy:
            self._adjust_parameter()

        if self._cfg.backend == BackendTarget.ASCEND:
            # quant weight to int8, bias to int32
            self._input_quantizer = self._input_quantizer.convert_to_fakequantparam()
            self._weight_quantizer = self._weight_quantizer.convert_to_fakequantparam()
            weight_quant = None
            bias_quant = None
            bias_name = self._linear.weight.name + "_bias"
            if not self._is_deploy:
                weight_quantizer: P.FakeQuantParam = self._weight_quantizer.fq
                if hasattr(self._linear, "dtype"):
                    weight = self._linear.cast(self._linear.weight, self._linear.dtype)
                else:
                    weight = self._linear.weight
                quant_min, quant_max = get_quant_min_max(
                    num_bits=weight_quantizer.attrs[LinearFakeQuantizer.attr_key_num_bits],
                    signed=weight_quantizer.attrs[LinearFakeQuantizer.attr_key_signed],
                    narrow_range=weight_quantizer.attrs[LinearFakeQuantizer.attr_key_narrow_range])
                scale = np.array(weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_scale])
                zp = np.array(weight_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_zero_point])
                if not self._linear.transpose_b:
                    scale = scale.transpose()
                    zp = zp.transpose()
                weight_quant = quant_tensor_data(weight, scale, zp, quant_min, quant_max,
                                                 self._weight_axis)
                self._linear.weight = Parameter(Tensor(weight_quant, dtype=dtype.int8), name=self._linear.weight.name)
                input_quantizer = self._input_quantizer.fq
                act_scale = np.array(input_quantizer.attrs[P.FakeQuantParam.attr_key_linear_quant_scale])
                dequant_scale = scale * act_scale
                if self._linear.has_bias:
                    bias_quant = quant_bias_data(self._linear.bias, dequant_scale)
                    bias_name = self._linear.bias.name
                    self._linear.bias = Parameter(bias_quant, name=bias_name)
            param_bias_quant = bias_quant.asnumpy() if bias_quant is not None else None
            if param_bias_quant is not None:
                bn = self._get_bias_reduce_num()
                # refer to docstring of _get_bias_reduce_num for the reason of this divide operation.
                param_bias_quant = param_bias_quant / bn
            self._output_quantizer, bias = convert_to_dequant_bmm(self._input_quantizer,
                                                                  self._weight_quantizer,
                                                                  weight_quant,
                                                                  param_bias_quant,
                                                                  dst_dtype=self._linear.dtype,
                                                                  transpose_a=False,
                                                                  transpose_b=self._linear.transpose_b,
                                                                  strategy=self.antiquant_bmm_strategy(
                                                                      act_strategy=self._act_strategy,
                                                                      weight_strategy=self._weight_strategy,
                                                                      has_bias=True,  # offset correct by bias
                                                                      is_transpose=self._linear.transpose_b))
            self._linear.has_bias = True
            if bias is not None:
                self._linear.bias = Parameter(Tensor(bias, dtype=dtype.int32), name=bias_name)
            self._input_quantizer = convert_to_quant(self._input_quantizer,
                                                     strategy=(self._act_strategy,) if self._act_strategy else None)

    def calibrate(self):
        if hasattr(self._linear, "dtype"):
            weight = self._linear.cast(self._linear.weight, self._linear.dtype)
        else:
            weight = self._linear.weight
        self._weight_quantizer(weight)

    def core_construct(self, *args):
        pass

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
        # make activation shallow, would take effect in deploy mode
        x = self._act_mul(x, self._div(1.0, self._input_scale))
        weight = self._linear.weight

        if self._is_deploy:
            x = self._input_quantizer(x)
        else:
            x = self._act_observer(x)
            weight = self._weight_in_observer(weight)
            input_scale = self._calc_input_scale()
            self._assign(self._scale_store, input_scale)
            if self._linear.transpose_b:
                weight = self._weight_mul(weight, input_scale)
                weight = self._weight_quantizer(weight)
                weight = self._weight_div(weight, input_scale)
            else:
                # now only Matmul is supported, shall generalize to bmm
                weight_scale = self._expand(input_scale, 1)
                weight = self._weight_mul(weight, weight_scale)
                weight = self._weight_quantizer(weight)
                weight = self._weight_div(weight, weight_scale)
            x = self._act_mul(x, self._div(1.0, input_scale))
            x = self._input_quantizer(x)
            x = self._act_mul(x, input_scale)

        if self._is_deploy:
            # (matmul(x, int8_weight) + int32_bias) * dequant_scale
            bias = None
            if self._linear.has_bias:
                bias = self._linear.bias
            x = self._output_quantizer(x, weight, bias)
        else:
            if hasattr(self._linear, "dtype"):
                weight = self._linear.cast(weight, self._linear.dtype)
                x = self._linear.cast(x, self._linear.dtype)
            else:
                weight = self._linear.weight
                x = self._linear.cast(x, self._linear.weight.dtype)

            x = self._linear.matmul(x, weight)
            if self._linear.has_bias:
                if hasattr(self._linear, "dtype"):
                    bias = self._linear.cast(self._linear.bias, self._linear.dtype)
                else:
                    bias = self._linear.cast(self._linear.bias, x.dtype)
                x = self._linear.bias_add(x, bias)
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output
