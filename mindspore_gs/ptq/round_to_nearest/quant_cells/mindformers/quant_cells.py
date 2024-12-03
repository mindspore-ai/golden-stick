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
"""ptq quant cells."""
import copy
import abc
import math
import numpy as np
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import ops
from mindspore import Parameter, Tensor, dtype
from mindspore.common.initializer import initializer, Zero
from mindspore.ops.auto_generate import DynamicQuantExt, KVCacheScatterUpdate
from mindformers.modules.layers import Linear
from mindformers.modules.paged_attention_mgr import PagedAttentionMgr

from mindspore_gs.common.gs_enum import BackendTarget
from mindspore_gs.common import logger
from mindspore_gs.ptq.basic_quant_func import get_quant_min_max, quant_tensor_data, convert_fp32_to_int64
from mindspore_gs.quantization.layer_policy import PerChannelArgs
from mindspore_gs.ptq.quant_cell import PTQCell
from mindspore_gs.ptq.ptq_policy import PTQLayerPolicy
from mindspore_gs.ptq.ptq_config import OutliersSuppressionType, QuantGranularity
from mindspore_gs.ptq.fake_quantizer import LinearFakeQuantizer
from mindspore_gs.ptq.convert_utils import (
    convert_to_fusion_antiquant, convert_to_quant, convert_to_dequant,
    convert_to_fusion_antiquant_for_deploy, convert_to_quant_for_deploy, convert_to_dynamic_quant_for_deploy,
    convert_to_antiquant_for_deploy
)


class LinearQuant(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy: PTQLayerPolicy):
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

    def convert(self, backend=BackendTarget.NONE, is_deploy=False):
        weight_only = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                      self._policy.get_config().weight_quant_dtype == dtype.int8 and \
                      self._policy.get_config().act_quant_dtype is None
        all_quant = isinstance(self._weight_quantizer, LinearFakeQuantizer) and \
                    isinstance(self._input_quantizer, LinearFakeQuantizer)
        if not all_quant and not weight_only:
            logger.info(f"LinearQuant {self} is not quanted.")
            return

        super(LinearQuant, self).convert(backend)
        # quant weight to int8
        weight_qparams = self._weight_quantizer.quant_params()
        weight = self._linear.cast(self._linear.weight, self._cast_dtype)
        quant_min, quant_max = get_quant_min_max(
            weight_qparams.get(LinearFakeQuantizer.attr_key_num_bits),
            weight_qparams.get(LinearFakeQuantizer.attr_key_symmetric),
            weight_qparams.get(LinearFakeQuantizer.attr_key_narrow_range))
        scale = weight_qparams.get(LinearFakeQuantizer.attr_key_quant_scale)
        zp = weight_qparams.get(LinearFakeQuantizer.attr_key_quant_zero_point)
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
                weight_qparams, transpose_weight=self._linear.transpose_b,
                dst_dtype=self._cast_dtype, strategy=
                self.wqbmm_strategy(self._act_strategy, self._weight_strategy, is_transpose=self._linear.transpose_b)
            )
            self._quant_deployed = True
        else:
            input_qparams = self._input_quantizer.quant_params()
            self._output_quantizer = convert_to_dequant(input_qparams, weight_qparams)
            self._input_quantizer = convert_to_quant(input_qparams)
            self._quant_deployed = True
            raise RuntimeError(f'current version not support all quantization, only for weight quantization')

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
        weight = self._linear.cast(self._linear.weight, self._cast_dtype)
        if not self._quant_deployed:
            weight = self._weight_quantizer(weight)
        x = self._linear.matmul(x, weight)
        if self._linear.has_bias:
            x = self._linear.bias_add(x, self._linear.cast(self._linear.bias, self._linear.dtype))
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        x = F.cast(x, ori_dtype)
        output = P.Reshape()(x, out_shape)
        return output


class LinearDeploy(PTQCell):
    """Linear layer wrapper with min max"""

    def __init__(self, linear: Linear, policy: PTQLayerPolicy):
        super(LinearDeploy, self).__init__(linear, policy)
        self._converted = True
        if not isinstance(linear, Linear):
            raise ValueError(f'only Linear cell is supported, but got {type(linear)}')
        self._linear = linear

        rank = len(linear.weight.shape)
        self._weight_axis = rank - 2 if linear.matmul.transpose_b else rank - 1
        has_dtype = hasattr(self._linear, "dtype")
        self._cast_dtype = self._linear.dtype if has_dtype else self._linear.weight.dtype
        self._act_strategy = None
        self._weight_strategy = None
        if "in_strategy" in self._linear.matmul.get_attr_dict():
            self._act_strategy = self._linear.matmul.in_strategy[0]
            self._weight_strategy = self._linear.matmul.in_strategy[1]
        self._linear.weight = Parameter(initializer('ones', self._linear.weight.shape, dtype.int8),
                                        name=self._linear.weight.name)
        self._input_quantizer = None
        self._output_quantizer = None
        # self._linear.out_channels maybe not equal w_out
        w_out = self._linear.weight.shape[0] if self._linear.transpose_b else self._linear.weight.shape[1]
        w_in = self._linear.weight.shape[1] if self._linear.transpose_b else self._linear.weight.shape[0]
        if self._policy.get_config().act_quant_granularity is QuantGranularity.PER_TOKEN:
            self._weight_quantizer = convert_to_dynamic_quant_for_deploy(
                w_out=w_out,
                w_in=w_in,
                is_per_channel=True,
                has_smooth=self._policy.get_config().outliers_suppression != OutliersSuppressionType.NONE,
                transpose_weight=self._linear.transpose_b,
                dst_dtype=self._cast_dtype,
                strategy=self.dynamic_bmm_strategy(self._act_strategy, self._weight_strategy,
                                                   is_transpose=self._linear.transpose_b)
                )
        else:
            self._weight_quantizer = convert_to_fusion_antiquant_for_deploy(
                axis=self._weight_axis, output_channel=w_out,
                data_rank=len(self._linear.weight.shape),
                is_per_channel=True,
                transpose_weight=self._linear.transpose_b,
                dst_dtype=self._cast_dtype,
                strategy=self.wqbmm_strategy(self._act_strategy, self._weight_strategy,
                                             is_transpose=self._linear.transpose_b)
                )

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    # pylint: disable=W0221
    def construct(self, x):
        """
        Defines the computation of LinearWithMinMax to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        out_shape = self.handler().shape(x)[:-1] + (self._linear.out_channels,)
        x = P.Reshape()(x, (-1, self._linear.in_channels))
        if hasattr(self._linear, "expert_flag") and self._linear.expert_flag:
            if self._linear.use_expert_group_size is True:
                x = self.handler().reshape(x, (-1, self._linear.expert_num, self._linear.expert_group_size,
                                               self._linear.in_channels))
            else:
                x = self.handler().reshape(x, (self._linear.outer_batch, self._linear.expert_num, -1,
                                               self._linear.in_channels))
        ori_dtype = F.dtype(x)

        x = self._linear.cast(x, self._cast_dtype)
        x = self._weight_quantizer(x, self._linear.weight)
        if self._linear.has_bias:
            x = self._linear.bias_add(x, self._linear.cast(self._linear.bias, self._linear.dtype))
        if self._linear.activation_flag:
            x = self._linear.activation(x)
        x = F.cast(x, ori_dtype)
        output = self.handler().reshape(x, out_shape)
        return output


class PagedAttentionQuant(PTQCell):
    """PagedAttention Quant wrapper with min max"""

    def __init__(self, kvcache: PagedAttentionMgr, policy: PTQLayerPolicy):
        super(PagedAttentionQuant, self).__init__(kvcache, policy)
        self._kvcache = kvcache

        # PagedAttentionMgr's shape is BSH currently.
        n = kvcache.n_kv_heads
        d = kvcache.head_dim
        key_fq_args = {}
        value_fq_args = {}
        self._key_strategy = None
        self._value_strategy = None
        self.key_t_scale = None
        self.key_t_zp = None
        self.value_t_scale = None
        self.value_t_zp = None
        self.key_value_t_zp = None
        self.key_value_t_scale = None
        perchannel_args = PerChannelArgs(n * d, 2, 3)

        if "in_strategy" in kvcache.reshape_and_cache.get_attr_dict():
            self._key_strategy = kvcache.reshape_and_cache.in_strategy[0]
            self._value_strategy = kvcache.reshape_and_cache.in_strategy[1]
            perchannel_args = PerChannelArgs(n * d * self._key_strategy[2], 2, 3)
            key_fq_args["strategy"] = (kvcache.reshape_and_cache.in_strategy[0],)
            value_fq_args["strategy"] = (kvcache.reshape_and_cache.in_strategy[1],)

        self._key_input_quantizer = self._policy.get_kvcache_quantizer(input_index=0, perchannel_args=perchannel_args,
                                                                       **key_fq_args)
        self._value_input_quantizer = self._policy.get_kvcache_quantizer(input_index=1, perchannel_args=perchannel_args,
                                                                         **value_fq_args)
        self._weight_quantizer = None
        prex = ""
        for _, param in kvcache.parameters_and_names():
            prex = param.name.rsplit(".", 1)[0]
        if self._key_input_quantizer:
            self._key_input_quantizer.float_min.data.name = prex + "_key_input_float_min"
            self._key_input_quantizer.float_max.data.name = prex + "_key_input_float_max"
        if self._value_input_quantizer:
            self._value_input_quantizer.float_min.data.name = prex + "_value_input_float_min"
            self._value_input_quantizer.float_max.data.name = prex + "_value_input_float_max"

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    def convert(self, backend: BackendTarget = BackendTarget.NONE, is_deploy=False):
        if backend == BackendTarget.ASCEND:
            key_input_qparams = self._key_input_quantizer.quant_params()
            value_input_qparams = self._value_input_quantizer.quant_params()
            self._key_input_quantizer = convert_to_quant(key_input_qparams, self._key_strategy)
            self._value_input_quantizer = convert_to_quant(value_input_qparams, self._value_strategy)
            self.key_t_scale = copy.deepcopy(self._key_input_quantizer.t_scale)
            self.value_t_scale = copy.deepcopy(self._value_input_quantizer.t_scale)

            key_t_scale_np = self.key_t_scale.asnumpy()
            value_t_scale_np = self.value_t_scale.asnumpy()
            t_scale_len = key_t_scale_np.shape[0]
            key_t_scale_np = convert_fp32_to_int64(key_t_scale_np.astype(np.float32))
            value_t_scale_np = convert_fp32_to_int64(value_t_scale_np.astype(np.float32))
            key_value_t_scale_np = np.concatenate((key_t_scale_np.reshape((1, t_scale_len)),
                                                   value_t_scale_np.reshape((1, t_scale_len))))
            self.key_value_t_scale = Parameter(Tensor(key_value_t_scale_np, dtype=dtype.int64),
                                               name="key_value_t_scale")

            key_t_zp = copy.deepcopy(self._key_input_quantizer.t_zp)
            key_t_zp_np = key_t_zp.asnumpy().astype(np.float16)
            self.key_t_zp = Parameter(Tensor(key_t_zp_np, dtype=dtype.float16), name=key_t_zp.name)

            value_t_zp = copy.deepcopy(self._value_input_quantizer.t_zp)
            value_t_zp_np = value_t_zp.asnumpy().astype(np.float16)
            self.value_t_zp = Parameter(Tensor(value_t_zp_np, dtype=dtype.float16), name=value_t_zp.name)

            t_zp_len = value_t_zp_np.shape[0]
            key_t_zp_np = (key_t_zp_np*-1).astype(np.int32)
            value_t_zp_np = (value_t_zp_np*-1).astype(np.int32)
            key_value_t_zp = np.concatenate((key_t_zp_np.reshape((1, t_zp_len)), value_t_zp_np.reshape((1, t_zp_len))))
            self.key_value_t_zp = Parameter(Tensor(key_value_t_zp, dtype=dtype.int32), name="key_value_t_zp")

            self._kvcache.key_cache = Parameter(initializer('ones', self._kvcache.key_cache.shape, dtype.int8),
                                                name=self._kvcache.key_cache.name, requires_grad=False)
            self._kvcache.value_cache = Parameter(initializer('ones', self._kvcache.value_cache.shape, dtype.int8),
                                                  name=self._kvcache.value_cache.name, requires_grad=False)
        else:
            raise ValueError("Only support convert PagedAttentionMgr to MS backend.")

    # pylint: disable=W0221
    # pylint: disable=W0613
    def construct(self, key, value, slot_mapping, batch_valid_length=None):
        """
        Defines the computation of PagedAttentionQuant to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        key = self._key_input_quantizer(key)
        value = self._value_input_quantizer(value)
        return self._kvcache.reshape_and_cache(key, value, self._kvcache.key_cache, self._kvcache.value_cache,
                                               slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        return self._kvcache.paged_attention(query, self._kvcache.key_cache, self._kvcache.value_cache, block_tables,
                                             batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self._kvcache.paged_attention_with_alibi(query, self._kvcache.key_cache, self._kvcache.value_cache,
                                                        block_tables, batch_valid_length, alibi_tensor)


class PagedAttentionDeployBase(PTQCell):
    """PagedAttention deploy base class"""

    def __init__(self, kvcache: PagedAttentionMgr, policy: PTQLayerPolicy):
        super(PagedAttentionDeployBase, self).__init__(kvcache, policy)
        self._converted = True
        self._kvcache = kvcache
        # PagedAttentionMgr's shape is BSH currently.
        n = kvcache.n_kv_heads
        d = kvcache.head_dim
        self._key_in_strategy = None
        self._value_in_strategy = None
        self._key_out_strategy = None
        self._value_out_strategy = None

        if "in_strategy" in kvcache.reshape_and_cache.get_attr_dict():
            self._key_in_strategy = kvcache.reshape_and_cache.in_strategy[0]
            self._value_in_strategy = kvcache.reshape_and_cache.in_strategy[1]
            n = n * self._key_in_strategy[2]
            self._key_out_strategy = kvcache.reshape_and_cache.in_strategy[2]
            self._value_out_strategy = kvcache.reshape_and_cache.in_strategy[3]
        ic = n * d
        self._key_input_quantizer = convert_to_quant_for_deploy(ic, self._key_in_strategy)
        self._value_input_quantizer = convert_to_quant_for_deploy(ic, self._value_in_strategy)
        self._weight_quantizer = None
        self._input_quantizer = None
        self._output_quantizer = None

        self.key_t_scale = copy.deepcopy(self._key_input_quantizer.t_scale)
        self.value_t_scale = copy.deepcopy(self._value_input_quantizer.t_scale)
        key_t_zp = copy.deepcopy(self._key_input_quantizer.t_zp)
        self.key_t_zp = Parameter(Tensor(key_t_zp.asnumpy().astype(np.float16), dtype=dtype.float16),
                                  name=key_t_zp.name)
        value_t_zp = copy.deepcopy(self._value_input_quantizer.t_zp)
        self.value_t_zp = Parameter(Tensor(value_t_zp.asnumpy().astype(np.float16), dtype=dtype.float16),
                                    name=value_t_zp.name)

        dst_type = self._kvcache.key_cache.dtype
        self._key_output_quantizer = convert_to_antiquant_for_deploy(n, d, self._key_out_strategy, dst_type)
        self._value_output_quantizer = convert_to_antiquant_for_deploy(n, d, self._value_out_strategy, dst_type)
        self._kvcache.key_cache = Parameter(initializer('ones', self._kvcache.key_cache.shape, dtype.int8),
                                            name=self._kvcache.key_cache.name, requires_grad=False)
        self._kvcache.value_cache = Parameter(initializer('ones', self._kvcache.value_cache.shape, dtype.int8),
                                              name=self._kvcache.value_cache.name, requires_grad=False)

    def weight_quantizer(self):
        return self._weight_quantizer

    def core_construct(self, *args):
        pass

    # pylint: disable=W0221
    # pylint: disable=W0613
    def construct(self, key, value, slot_mapping, batch_valid_length=None):
        """
        Defines the computation of PagedAttentionQuant to be performed.

        Returns:
            Tensor, returns the computed result.
        """
        key = self._key_input_quantizer(key)
        value = self._value_input_quantizer(value)
        return self._kvcache.reshape_and_cache(key, value, self._kvcache.key_cache, self._kvcache.value_cache,
                                               slot_mapping)

    @abc.abstractmethod
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        return NotImplementedError

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        kcache = self._key_output_quantizer(self._kvcache.key_cache, self.key_t_zp, self.key_t_scale)
        vcache = self._value_output_quantizer(self._kvcache.value_cache, self.value_t_zp, self.value_t_scale)
        return self.paged_attention_with_alibi(query, kcache, vcache, block_tables, batch_valid_length, alibi_tensor)


class PagedAttentionDeploy(PagedAttentionDeployBase):
    """PagedAttention deploy with no fuison"""

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        kcache = self._key_output_quantizer(self._kvcache.key_cache, self.key_t_zp, self.key_t_scale)
        vcache = self._value_output_quantizer(self._kvcache.value_cache, self.value_t_zp, self.value_t_scale)
        return self._kvcache.paged_attention(query, kcache, vcache, block_tables, batch_valid_length)


class PagedAttentionDeployFusion(PagedAttentionDeployBase):
    """PagedAttention deploy with fuison ops."""

    def __init__(self, kvcache: PagedAttentionMgr, policy: PTQLayerPolicy):
        super(PagedAttentionDeployFusion, self).__init__(kvcache, policy)
        self.key_value_t_zp = Parameter(Tensor(np.zeros((2, self.key_t_zp.shape[0])), dtype=dtype.int32),
                                        name="key_value_t_zp")
        self.key_value_t_scale = Parameter(Tensor(np.zeros((2, self.value_t_scale.shape[0])), dtype=dtype.int64),
                                           name="key_value_t_scale")
        if "in_strategy" in kvcache.paged_attention.get_attr_dict():
            pa_strategy = kvcache.paged_attention.in_strategy
            antiquant_strategy = (1, self._key_in_strategy[2],)
            self._kvcache.paged_attention.shard((*pa_strategy, antiquant_strategy, antiquant_strategy))

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        kcache = self._kvcache.key_cache
        vcache = self._kvcache.value_cache
        return self._kvcache.paged_attention(query, kcache, vcache, block_tables, batch_valid_length,
                                             self.key_value_t_scale, self.key_value_t_zp)


class DynamicQuantPagedAttentionDeploy(PTQCell):
    """PagedAttention deploy base class"""

    def __init__(self, kvcache: PagedAttentionMgr, policy: PTQLayerPolicy):
        super(DynamicQuantPagedAttentionDeploy, self).__init__(kvcache, policy)
        self._kvcache = kvcache
        self._converted = True
        n = kvcache.n_kv_heads
        d = kvcache.head_dim
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.concat_scale = P.Concat(axis=0)
        self.cast = P.Cast()
        self.ori_shape = self._kvcache.key_cache.shape
        block_num = self.ori_shape[0]
        block_size = self.ori_shape[1]
        max_seq_length = kvcache.seq_length
        max_batch_size = math.floor(block_num * block_size / max_seq_length)
        self.is_first_iteration = kvcache.is_first_iteration
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self._weight_quantizer = None
        self._input_quantizer = None
        self._output_quantizer = None
        self.dynamic_quant = DynamicQuantExt()
        self.paged_attention = ops.auto_generate.PagedAttention(self._kvcache.n_heads,
                                                                self._kvcache.scale_value,
                                                                self._kvcache.n_kv_heads,
                                                                "PERTOKEN")
        self.paged_attention_with_alibi = ops.auto_generate.PagedAttentionMask(self._kvcache.n_heads,
                                                                               self._kvcache.scale_value,
                                                                               self._kvcache.n_kv_heads,
                                                                               "PERTOKEN")
        self._kvcache.key_cache = Parameter(initializer('ones', self._kvcache.key_cache.shape, dtype.int8),
                                            name=self._kvcache.key_cache.name, requires_grad=False)
        self._kvcache.value_cache = Parameter(initializer('ones', self._kvcache.value_cache.shape, dtype.int8),
                                              name=self._kvcache.value_cache.name, requires_grad=False)

        self.key_scale = Parameter(initializer('ones', (max_batch_size, max_seq_length), dtype.float32),
                                   requires_grad=False)
        self.value_scale = Parameter(initializer('ones', (max_batch_size, max_seq_length), dtype.float32),
                                     requires_grad=False)
        self.scatter_scales = KVCacheScatterUpdate()
        self.kv_offset = Tensor(shape=(2, max_batch_size, max_seq_length), dtype=dtype.float16, init=Zero())

        key_out_strategy = None
        key_in_stragegy = None
        dp = None
        if "in_strategy" in kvcache.reshape_and_cache.get_attr_dict():
            key_in_stragegy = kvcache.reshape_and_cache.in_strategy[0]
            key_out_strategy = kvcache.reshape_and_cache.in_strategy[2]
            n = n * key_out_strategy[2]
            dp = key_in_stragegy[0]
        self.ic = n*d

        if "in_strategy" in kvcache.paged_attention.get_attr_dict():
            pa_strategy = kvcache.paged_attention.in_strategy
            antiquant_strategy = (1, 1, 1,)
            self.paged_attention.shard((*pa_strategy, antiquant_strategy, antiquant_strategy))
            self.dynamic_quant.shard((key_in_stragegy,))
            self.scatter_scales.shard(((dp, 1), (dp,), (dp, 1),))
            self.concat_scale.shard(((1, dp, 1), (1, dp, 1)))

        if "in_strategy" in kvcache.paged_attention_with_alibi.get_attr_dict():
            pa_strategy = kvcache.paged_attention_with_alibi.in_strategy
            antiquant_strategy = (1, 1, 1,)
            self.paged_attention_with_alibi.shard((*pa_strategy[:-1], antiquant_strategy, antiquant_strategy,
                                                   pa_strategy[-1]))
            self.dynamic_quant.shard((key_in_stragegy,))
            self.scatter_scales.shard(((dp, 1), (dp,), (dp, 1),))
            self.concat_scale.shard(((1, dp, 1), (1, dp, 1)))

    def weight_quantizer(self):
        return None

    def core_construct(self, *args):
        pass

    # pylint: disable=W0221
    def construct(self, key, value, slot_mapping, batch_valid_length):
        """The forward compute of KVCache for Paged Attention."""
        if self.is_first_iteration:
            batch_idx = batch_valid_length * 0
        else:
            batch_idx = batch_valid_length - 1

        quant_k, k_scale = self.dynamic_quant(key, None)
        quant_v, v_scale = self.dynamic_quant(value, None)
        self.scatter_scales(self.key_scale, batch_idx, k_scale, -1, "update")
        self.scatter_scales(self.value_scale, batch_idx, v_scale, -1, "update")
        return self._kvcache.reshape_and_cache(quant_k, quant_v, self._kvcache.key_cache, self._kvcache.value_cache,
                                               slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        key_scale = self.reshape(self.key_scale, (1, self.max_batch_size, self.max_seq_length))
        value_scale = self.reshape(self.value_scale, (1, self.max_batch_size, self.max_seq_length))
        kv_scale = self.cast(self.concat_scale((key_scale, value_scale)), dtype.float16)
        return self.paged_attention(query, self._kvcache.key_cache, self._kvcache.value_cache, block_tables,
                                    batch_valid_length, kv_scale, self.kv_offset)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        key_scale = self.reshape(self.key_scale, (1, self.max_batch_size, self.max_seq_length))
        value_scale = self.reshape(self.value_scale, (1, self.max_batch_size, self.max_seq_length))
        kv_scale = self.cast(self.concat_scale((key_scale, value_scale)), dtype.float16)
        return self.paged_attention_with_alibi(query, self._kvcache.key_cache, self._kvcache.value_cache, block_tables,
                                               batch_valid_length, kv_scale, self.kv_offset)
