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
import numpy as np

from mindspore import Parameter, dtype, Tensor
from mindspore import ops as msops
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import DynamicQuantExt, KVCacheScatterUpdate
from mindspore.common.initializer import initializer, Zero
from mindspore.nn import Cell

from mindformers.modules.paged_attention_mgr import PagedAttentionMgr
from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_world_size

from mindspore_gs.ptq.ptq_config import QuantGranularity
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.hal import C8PagedAttentionCell, QuantV2Cell, QuantParam
from mindspore_gs.ptq.basic_quant_func import quant_tensor


class QuantPageAttentionMgrCell(WrapperCell):
    """QuantPageAttentionMgrCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class KVCacheInt8(Checker):
            def check(self, config: InnerPTQConfig):
                return config.kvcache_quant_dtype == dtype.int8 and config.kvcache_quant_granularity == \
                        QuantGranularity.PER_CHANNEL

        Quantizer.reg_layer_map(PagedAttentionMgr, QuantPageAttentionMgrCell, KVCacheInt8())
        try:
            from mindformers.experimental.infer.core.parallel_paged_attention_mgr import ParallelPagedAttentionMgr
            Quantizer.reg_layer_map(ParallelPagedAttentionMgr, QuantPageAttentionMgrCell, KVCacheInt8())
        except ImportError:
            pass
        try:
            from research.llama3_1.infer.parallel_paged_attention_mgr import LlameParallelPagedAttentionMgr
            Quantizer.reg_layer_map(LlameParallelPagedAttentionMgr, QuantPageAttentionMgrCell, KVCacheInt8())
        except ImportError:
            pass

    def __init__(self, linear_name, layer, context: InnerPTQConfig, cfg, **kwargs):
        super().__init__(linear_name, layer, cfg, context, **kwargs)
        self.key_samples = []
        self.value_samples = []
        n = layer.n_kv_heads
        d = layer.head_dim
        self.ic = n * d
        self.compute_dtype = self.layer.key_cache.dtype
        self.key_t_zp = Parameter(initializer('zeros', (self.ic,), dtype=dtype.float64))
        self.key_t_scale = Parameter(initializer('zeros', (self.ic,), dtype=dtype.float64))
        self.value_t_zp = Parameter(initializer('zeros', (self.ic,), dtype=dtype.float64))
        self.value_t_scale = Parameter(initializer('zeros', (self.ic,), dtype=dtype.float64))

    def _quant_info(self):
        if self.cfg.kvcache_quant_dtype == dtype.int8:
            return f'C8-{str(self.cfg.kvcache_quant_granularity)}'
        raise RuntimeError(f"Unexpected kvcache_quant_dtype: {self.cfg.kvcache_quant_dtype}.")

    def process(self):
        if not self.key_samples or not self.value_samples:
            raise RuntimeError("Please catch ReshapeAndCache inputs before quantization.")

        key_cat_samples = msops.cat(tuple(self.key_samples), axis=0)
        value_cat_samples = msops.cat(tuple(self.value_samples), axis=0)

        key_t_scale, key_t_zp, _ = quant_tensor(key_cat_samples, msops.min, msops.max,
                                                self.cfg.kvcache_narrow_range, self.cfg.kvcache_symmetric,
                                                False, 0, self.cfg.kvcache_quant_dtype, 1, False)
        value_t_scale, value_t_zp, _ = quant_tensor(value_cat_samples, msops.min, msops.max,
                                                    self.cfg.kvcache_narrow_range, self.cfg.kvcache_symmetric,
                                                    False, 0, self.cfg.kvcache_quant_dtype, 1, False)

        self.key_t_scale.set_data(Tensor(np.squeeze(key_t_scale)))
        self.key_t_zp.set_data(Tensor(np.squeeze(key_t_zp)))
        self.value_t_scale.set_data(Tensor(np.squeeze(value_t_scale)))
        self.value_t_zp.set_data(Tensor(np.squeeze(value_t_zp)))
        self.cfg.dumper.dump_data(self.layer_name, "|key_quant_params|input0_key_cache_inputs", key_cat_samples)
        self.cfg.dumper.dump_data(self.layer_name, "|key_quant_params|output0_key_scale", self.key_t_scale)
        self.cfg.dumper.dump_data(self.layer_name, "|key_quant_params|output1_key_zp", self.key_t_zp)
        self.cfg.dumper.dump_data(self.layer_name, "|value_quant_params|input0_value_cache_inputs", value_cat_samples)
        self.cfg.dumper.dump_data(self.layer_name, "|value_quant_params|output0_value_scale", self.value_t_scale)
        self.cfg.dumper.dump_data(self.layer_name, "|value_quant_params|output1_value_zp", self.value_t_zp)
        self.key_samples.clear()
        self.value_samples.clear()

    def deploy(self):
        return DeployPageAttentionMgrCell(self._layer_name, self.layer, self.cfg,
                                          QuantParam(self.key_t_scale, self.key_t_zp),
                                          QuantParam(self.value_t_scale, self.value_t_zp))

    def add_hook(self):
        pass

    def remove_hook(self):
        pass

    def construct(self, x, *args, **kwargs):
        value = args[0]
        self.key_samples.append(msops.squeeze(x))
        self.value_samples.append(msops.squeeze(value))
        slot_mapping = args[1]
        self.layer.reshape_and_cache(x, value, self.layer.key_cache, self.layer.value_cache, slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, *args, **kwargs):
        """The forward compute of Paged Attention."""
        return self.layer.paged_attention(query, self.layer.key_cache, self.layer.value_cache, block_tables,
                                          batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor, *args, **kwargs):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self.layer.paged_attention_with_alibi(query, self.layer.key_cache, self.layer.value_cache,
                                                     block_tables, batch_valid_length, alibi_tensor)


class DeployPageAttentionMgrCell(Cell):
    """DeployPageAttentionMgrCell"""

    def __init__(self, layer_name, kvcache: PagedAttentionMgr, cfg: InnerPTQConfig, k_qparam: QuantParam,
                 v_qparam: QuantParam):
        super().__init__()
        self.layer = kvcache
        dst_type = self.layer.key_cache.dtype
        n = kvcache.n_kv_heads
        d = kvcache.head_dim
        self.enable_deploy_fusion = cfg.enable_deploy_fusion
        self.quant_pa = C8PagedAttentionCell(layer_name, cfg, dst_type, n, d, k_qparam, v_qparam)
        self._key_input_quantizer = QuantV2Cell.create(layer_name, dst_type, cfg, k_qparam)
        self._value_input_quantizer = QuantV2Cell.create(layer_name, dst_type, cfg, v_qparam)

        self.layer.key_cache = Parameter(initializer('ones', self.layer.key_cache.shape, dtype.int8),
                                         name=self.layer.key_cache.name, requires_grad=False)
        self.layer.value_cache = Parameter(initializer('ones', self.layer.value_cache.shape, dtype.int8),
                                           name=self.layer.value_cache.name, requires_grad=False)
        self.tensor_parallel_group_size = get_tensor_model_parallel_world_size()

    # pylint: disable=W0613
    def construct(self, key, value, slot_mapping, *args, **kwargs):
        """The forward compute of KVCache for Paged Attention."""
        key = self._key_input_quantizer(key)
        value = self._value_input_quantizer(value)
        self.layer.reshape_and_cache(key, value, self.layer.key_cache, self.layer.value_cache, slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, *args, **kwargs):
        """The forward compute of Paged Attention."""
        return self.quant_pa.paged_attn(self.layer, query, batch_valid_length, block_tables)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor, *args, **kwargs):
        """The forward compute of Paged Attention."""
        return self.quant_pa.paged_attn_with_alibi(self.layer, query, batch_valid_length, block_tables, alibi_tensor)

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        state_dict = self.quant_pa.param_shard_state(self.tensor_parallel_group_size)
        key_input_state_dict = self._key_input_quantizer.param_shard_state(self.tensor_parallel_group_size)
        state_dict.update(key_input_state_dict)
        value_input_state_dict = self._value_input_quantizer.param_shard_state(self.tensor_parallel_group_size)
        state_dict.update(value_input_state_dict)
        return state_dict


class DynamicQuantPageAttentionMgrCell(WrapperCell):
    """DynamicQuantPageAttentionMgrCell"""

    @staticmethod
    def reg_self():
        """reg_self"""
        class KVCacheInt8(Checker):
            def check(self, config: InnerPTQConfig):
                return config.kvcache_quant_dtype == dtype.int8 and config.kvcache_quant_granularity == \
                        QuantGranularity.PER_TOKEN

        Quantizer.reg_layer_map(PagedAttentionMgr, DynamicQuantPageAttentionMgrCell, KVCacheInt8())
        try:
            from mindformers.experimental.infer.core.parallel_paged_attention_mgr import ParallelPagedAttentionMgr
            Quantizer.reg_layer_map(ParallelPagedAttentionMgr, DynamicQuantPageAttentionMgrCell, KVCacheInt8())
        except ImportError:
            pass
        try:
            from research.llama3_1.infer.parallel_paged_attention_mgr import LlameParallelPagedAttentionMgr
            Quantizer.reg_layer_map(LlameParallelPagedAttentionMgr, QuantPageAttentionMgrCell, KVCacheInt8())
        except ImportError:
            pass

    def _quant_info(self):
        if self.cfg.kvcache_quant_dtype == dtype.int8:
            return f'C8-{str(self.cfg.kvcache_quant_granularity)}'
        raise RuntimeError(f"Unexpected kvcache_quant_dtype: {self.cfg.kvcache_quant_dtype}.")

    def deploy(self):
        return DynamicQuantPagedAttentionDeploy(self.layer)

    def add_hook(self):
        pass

    def remove_hook(self):
        pass

    def process(self):
        pass

    def paged_attn(self, *args, **kwargs):
        """paged_attn"""
        return self._layer.paged_attn(*args, **kwargs)

    def paged_attn_with_alibi(self, *args, **kwargs):
        """paged_attn_with_alibi"""
        return self._layer.paged_attn_with_alibi(*args, **kwargs)


class DynamicQuantPagedAttentionDeploy(Cell):
    """PagedAttention deploy base class"""

    def __init__(self, kvcache: PagedAttentionMgr):
        super().__init__()
        self._kvcache = kvcache
        self._converted = True
        self.is_first_iteration = kvcache.is_first_iteration
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
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self._weight_quantizer = None
        self._input_quantizer = None
        self._output_quantizer = None
        self.dynamic_quant = DynamicQuantExt()
        self.paged_attention = msops.auto_generate.PagedAttention(self._kvcache.n_heads,
                                                                  self._kvcache.scale_value,
                                                                  self._kvcache.n_kv_heads,
                                                                  "PERTOKEN")
        self.paged_attention_with_alibi = msops.auto_generate.PagedAttentionMask(self._kvcache.n_heads,
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
    # pylint: disable=W0613
    def construct(self, key, value, slot_mapping, batch_valid_length, *args, **kwargs):
        """The forward compute of KVCache for Paged Attention."""
        if self.is_first_iteration:
            batch_idx = batch_valid_length * 0
            t, h = key.shape
            key = self.reshape(key, (1, t, h))
            value = self.reshape(value, (1, t, h))
        else:
            batch_idx = batch_valid_length - 1
            t, h = key.shape
            key = self.reshape(key, (t, 1, h))
            value = self.reshape(value, (t, 1, h))

        quant_k, k_scale = self.dynamic_quant(key, None)
        quant_v, v_scale = self.dynamic_quant(value, None)
        self.scatter_scales(self.key_scale, batch_idx, k_scale, -1, "update")
        self.scatter_scales(self.value_scale, batch_idx, v_scale, -1, "update")
        return self._kvcache.reshape_and_cache(quant_k, quant_v, self._kvcache.key_cache, self._kvcache.value_cache,
                                               slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, *args, **kwargs):
        """The forward compute of Paged Attention."""
        key_scale = self.reshape(self.key_scale, (1, self.max_batch_size, self.max_seq_length))
        value_scale = self.reshape(self.value_scale, (1, self.max_batch_size, self.max_seq_length))
        kv_scale = self.cast(self.concat_scale((key_scale, value_scale)), dtype.float16)
        t, h = query.shape
        query = self.reshape(query, (t, 1, h))
        res = self.paged_attention(query, self._kvcache.key_cache, self._kvcache.value_cache, block_tables,
                                   batch_valid_length, kv_scale, self.kv_offset)
        res = self.reshape(res, (t, h))
        return res

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor, *args, **kwargs):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        key_scale = self.reshape(self.key_scale, (1, self.max_batch_size, self.max_seq_length))
        value_scale = self.reshape(self.value_scale, (1, self.max_batch_size, self.max_seq_length))
        kv_scale = self.cast(self.concat_scale((key_scale, value_scale)), dtype.float16)
        return self.paged_attention_with_alibi(query, self._kvcache.key_cache, self._kvcache.value_cache, block_tables,
                                               batch_valid_length, kv_scale, self.kv_offset, alibi_tensor)
