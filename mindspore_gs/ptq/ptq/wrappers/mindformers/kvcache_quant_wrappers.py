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
from enum import Enum

import numpy as np

from mindspore import Parameter, Tensor, dtype
from mindspore import ops as msops
from mindspore.common.initializer import initializer
from mindspore.nn import Cell

from mindformers.modules.paged_attention_mgr import PagedAttentionMgr
from mindformers.experimental.parallel_core.pynative.parallel_state import get_tensor_model_parallel_world_size

from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode, QuantGranularity
from mindspore_gs.ptq.convert_utils import QuantCellV2, AntiQuantCell
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.ptq.algorithms.quantizer import Quantizer
from mindspore_gs.ptq.ptq.wrapper_cell import Checker
from mindspore_gs.quantization.quant_utils import get_quant_min_max, cal_quantization_params, convert_fp32_to_int64


class DeviceType(Enum):
    """
    device type
    """
    ASCEND910B = 'ascend_910B'
    ASCEND310 = 'ascend_310'


class OpsPriority(Enum):
    """
    ops use priority
    """
    ACLNN = 'aclnn'
    INTERNAL = 'internal'
    ASD = 'asd'


class QuantPageAttentionMgrCell(WrapperCell):
    """QuantPageAttentionMgrCell"""

    @staticmethod
    def reg_self():
        class KVCacheInt8(Checker):
            def check(self, config: InnerPTQConfig):
                return config.kvcache_quant_dtype == dtype.int8

        Quantizer.reg_layer_map(PagedAttentionMgr, QuantPageAttentionMgrCell, KVCacheInt8())

    def __init__(self, linear_name, layer, cfg, network_helper):
        super().__init__(linear_name, layer, cfg, network_helper)
        self.key_samples = []
        self.value_samples = []
        self.quantizer_key_max = None
        self.quantizer_key_min = None
        self.quantizer_value_max = None
        self.quantizer_value_min = None
        self.kvcache_symmetric = cfg.kvcache_symmetric
        n = layer.n_kv_heads
        d = layer.head_dim
        self.ic = n * d
        self.k_scale_no_fusion = None
        self.k_zp_no_fusion = None
        self.v_scale_no_fusion = None
        self.v_zp_no_fusion = None
        self.k_v_scale_fusion = None
        self.k_v_zp_fusion = None
        self.kvcache_quant_min, self.kvcache_quant_max = get_quant_min_max(num_bits=8,
                                                                           signed=True,
                                                                           narrow_range=cfg.kvcache_narrow_range)
        if self.cfg.kvcache_quant_granularity is not QuantGranularity.PER_TOKEN:
            param_init_func = QuantPageAttentionMgrCell.param_init_map.get((DeviceType.ASCEND910B, OpsPriority.ASD))
            if param_init_func is None:
                raise ValueError("key ({cfg.device_type}, {cfg.ops_priority}) is not in \
                                QuantPageAttentionMgrCell.param_init_map.")
            param_init_func(self)

    def _param_init_asd(self):
        """_param_init_asd"""
        self.k_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_v_scale_fusion = Parameter(initializer('ones', (2, self.ic), dtype.int64))
        self.k_v_zp_fusion = Parameter(initializer('ones', (2, self.ic), dtype.int32))

    def _param_init_internal(self):
        """_param_init_internal"""
        self.k_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_scale_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.v_zp_no_fusion = Parameter(initializer('ones', (self.ic), dtype.float16))
        self.k_v_scale_fusion = Parameter(initializer('ones', (2, self.ic), dtype.float16))
        self.k_v_zp_fusion = Parameter(initializer('ones', (2, self.ic), dtype.float16))

    param_init_map = {
        (DeviceType.ASCEND910B, OpsPriority.ACLNN): _param_init_asd,
        (DeviceType.ASCEND910B, OpsPriority.INTERNAL): _param_init_internal,
        (DeviceType.ASCEND910B, OpsPriority.ASD): _param_init_asd,
        (DeviceType.ASCEND310, OpsPriority.ACLNN): _param_init_asd,
        (DeviceType.ASCEND310, OpsPriority.INTERNAL): _param_init_internal,
        (DeviceType.ASCEND310, OpsPriority.ASD): _param_init_asd,
    }

    def process(self):
        if self.cfg.kvcache_quant_granularity == QuantGranularity.PER_TOKEN:
            return
        if not self.key_samples or not self.value_samples:
            raise RuntimeError("Please catch ReshapeAndCache inputs before quantization.")
        key_cat_samples = msops.cat(tuple(self.key_samples), axis=0)
        self.quantizer_key_max = msops.max(key_cat_samples, 0)[0]
        self.quantizer_key_min = msops.min(key_cat_samples, 0)[0]

        value_cat_samples = msops.cat(tuple(self.value_samples), axis=0)
        self.quantizer_value_max = msops.max(value_cat_samples, 0)[0]
        self.quantizer_value_min = msops.min(value_cat_samples, 0)[0]

        key_t_scale, key_t_zp = cal_quantization_params(self.quantizer_key_min.asnumpy(),
                                                        self.quantizer_key_max.asnumpy(),
                                                        self.kvcache_quant_min,
                                                        self.kvcache_quant_max,
                                                        symmetric=self.kvcache_symmetric)
        value_t_scale, value_t_zp = cal_quantization_params(self.quantizer_value_min.asnumpy(),
                                                            self.quantizer_value_max.asnumpy(),
                                                            self.kvcache_quant_min,
                                                            self.kvcache_quant_max,
                                                            symmetric=self.kvcache_symmetric)
        key_t_scale = np.squeeze(key_t_scale).astype(np.float16)
        key_t_zp = np.squeeze(key_t_zp).astype(np.float16)
        value_t_scale = np.squeeze(value_t_scale).astype(np.float16)
        value_t_zp = np.squeeze(value_t_zp).astype(np.float16)
        self.k_scale_no_fusion.set_data(Tensor(key_t_scale, dtype=dtype.float16))
        self.k_zp_no_fusion.set_data(Tensor(key_t_zp, dtype=dtype.float16))
        self.v_scale_no_fusion.set_data(Tensor(value_t_scale, dtype=dtype.float16))
        self.v_zp_no_fusion.set_data(Tensor(value_t_zp, dtype=dtype.float16))
        param_compute_func = QuantPageAttentionMgrCell.param_compute_map[(DeviceType.ASCEND910B, OpsPriority.ASD)]
        if param_compute_func is None:
            raise ValueError("key ({self.cfg.device_type}, {self.cfg.ops_priority}) is \
                                    not in QuantPageAttentionMgrCell.param_compute_map.")
        param_compute_func(self, key_t_scale, value_t_scale, key_t_zp, value_t_zp)

        self.key_samples.clear()
        self.value_samples.clear()

    def _param_compute_asd(self, key_t_scale, value_t_scale, key_t_zp, value_t_zp):
        """_param_compute_asd"""
        t_scale_len = self.k_scale_no_fusion.shape[0]
        key_t_scale = convert_fp32_to_int64(key_t_scale.astype(np.float32))
        value_t_scale = convert_fp32_to_int64(value_t_scale.astype(np.float32))
        key_value_t_scale = np.concatenate((key_t_scale.reshape((1, t_scale_len)),
                                            value_t_scale.reshape((1, t_scale_len))))

        t_zp_len = self.v_zp_no_fusion.shape[0]
        key_t_zp = (key_t_zp*-1).astype(np.int32)
        value_t_zp = (value_t_zp*-1).astype(np.int32)
        key_value_t_zp = np.concatenate((key_t_zp.reshape((1, t_zp_len)), value_t_zp.reshape((1, t_zp_len))))

        self.k_v_scale_fusion.set_data(Tensor(key_value_t_scale, dtype=dtype.int64))
        self.k_v_zp_fusion.set_data(Tensor(key_value_t_zp, dtype=dtype.int32))

    def _param_compute_internal(self, key_t_scale, value_t_scale, key_t_zp, value_t_zp):
        """_param_compute_internal"""
        t_scale_len = self.k_scale_no_fusion.shape[0]
        key_value_t_scale = np.concatenate((key_t_scale.reshape((1, t_scale_len)),
                                            value_t_scale.reshape((1, t_scale_len))))
        t_zp_len = self.v_zp_no_fusion.shape[0]
        key_t_zp = key_t_zp*-1
        value_t_zp = value_t_zp*-1
        key_value_t_zp = np.concatenate((key_t_zp.reshape((1, t_zp_len)), value_t_zp.reshape((1, t_zp_len))))
        self.k_v_scale_fusion.set_data(Tensor(key_value_t_scale, dtype=dtype.float16))
        self.k_v_zp_fusion.set_data(Tensor(key_value_t_zp, dtype=dtype.float16))

    param_compute_map = {
        (DeviceType.ASCEND910B, OpsPriority.ACLNN): _param_compute_asd,
        (DeviceType.ASCEND910B, OpsPriority.INTERNAL): _param_compute_internal,
        (DeviceType.ASCEND910B, OpsPriority.ASD): _param_compute_asd,
        (DeviceType.ASCEND310, OpsPriority.ACLNN): _param_compute_asd,
        (DeviceType.ASCEND310, OpsPriority.INTERNAL): _param_compute_internal,
        (DeviceType.ASCEND310, OpsPriority.ASD): _param_compute_asd,
    }

    def deploy(self):
        if self.cfg.kvcache_quant_granularity == QuantGranularity.PER_TOKEN:
            return DeployDynamicQuantPagedAttentionCell(self.layer)
        return DeployPageAttentionMgrCell(self.layer, self.v_scale_no_fusion, self.v_zp_no_fusion,
                                          self.k_scale_no_fusion, self.k_zp_no_fusion, self.k_v_scale_fusion,
                                          self.k_v_zp_fusion, self.cfg)

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
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        return self.layer.paged_attention(query, self.layer.key_cache, self.layer.value_cache, block_tables,
                                          batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self.layer.paged_attention_with_alibi(query, self.layer.key_cache, self.layer.value_cache,
                                                     block_tables, batch_valid_length, alibi_tensor)


class DeployPageAttentionMgrCell(Cell):
    """DeployPageAttentionMgrCell"""

    def __init__(self, kvcache: PagedAttentionMgr, v_scale_no_fusion, v_zp_no_fusion, k_scale_no_fusion, k_zp_no_fusion,
                 k_v_scale_fusion, k_v_zp_fusion, cfg: InnerPTQConfig):
        super().__init__()
        self.layer = kvcache
        self.enable_deploy_fusion = cfg.enable_deploy_fusion
        self._key_input_quantizer = QuantCellV2(Tensor(k_scale_no_fusion.asnumpy(), dtype=dtype.float16),
                                                Tensor(k_zp_no_fusion.asnumpy().astype(np.int8), dtype=dtype.int8))
        self._value_input_quantizer = QuantCellV2(Tensor(v_scale_no_fusion.asnumpy(), dtype=dtype.float16),
                                                  Tensor(v_zp_no_fusion.asnumpy().astype(np.int8), dtype=dtype.int8))
        dst_type = self.layer.key_cache.dtype
        n = kvcache.n_kv_heads
        d = kvcache.head_dim
        self._key_output_quantizer = AntiQuantCell(n, d, dst_type)
        self._value_output_quantizer = AntiQuantCell(n, d, dst_type)
        if cfg.mode == PTQMode.QUANTIZE or not self.enable_deploy_fusion:
            self.k_zp_no_fusion = k_zp_no_fusion
            self.v_zp_no_fusion = v_zp_no_fusion
            self.k_scale_no_fusion = k_scale_no_fusion
            self.v_scale_no_fusion = v_scale_no_fusion
        if cfg.mode == PTQMode.QUANTIZE or self.enable_deploy_fusion:
            self.k_v_scale_fusion = k_v_scale_fusion
            self.k_v_zp_fusion = k_v_zp_fusion

        self.layer.key_cache = Parameter(initializer('ones', self.layer.key_cache.shape, dtype.int8),
                                         name=self.layer.key_cache.name, requires_grad=False)
        self.layer.value_cache = Parameter(initializer('ones', self.layer.value_cache.shape, dtype.int8),
                                           name=self.layer.value_cache.name, requires_grad=False)
        self.tensor_parallel_group_size = get_tensor_model_parallel_world_size()

    def construct(self, key, value, slot_mapping):
        """The forward compute of KVCache for Paged Attention."""
        key = self._key_input_quantizer(key)
        value = self._value_input_quantizer(value)
        self.layer.reshape_and_cache(key, value, self.layer.key_cache, self.layer.value_cache, slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        if not self.enable_deploy_fusion:
            kcache = self._key_output_quantizer(self.layer.key_cache, self.k_zp_no_fusion, self.k_scale_no_fusion)
            vcache = self._value_output_quantizer(self.layer.value_cache, self.v_zp_no_fusion, self.v_scale_no_fusion)
            return self.layer.paged_attention(query, kcache, vcache, block_tables, batch_valid_length)
        return self.layer.paged_attention(query, self.layer.key_cache, self.layer.value_cache, block_tables,
                                          batch_valid_length, self.k_v_scale_fusion, self.k_v_zp_fusion)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of Paged Attention."""
        if not self.enable_deploy_fusion:
            kcache = self._key_output_quantizer(self.layer.key_cache, self.k_zp_no_fusion, self.k_scale_no_fusion)
            vcache = self._value_output_quantizer(self.layer.value_cache, self.v_zp_no_fusion, self.v_scale_no_fusion)
            return self.layer.paged_attention_with_alibi(query, kcache, vcache, block_tables, batch_valid_length,
                                                         alibi_tensor)
        return self.layer.paged_attention_with_alibi(query, self.layer.key_cache, self.layer.value_cache,
                                                     block_tables, batch_valid_length, self.k_v_scale_fusion,
                                                     self.k_v_zp_fusion, alibi_tensor)

    def sharded_state_dict(self):
        """provide the sharded state dict based on the config"""
        state_dict = {}
        if self.enable_deploy_fusion:
            key_value_t_scale_shard = (1, self.tensor_parallel_group_size)
            key_value_t_zp_shard = (1, self.tensor_parallel_group_size)
            state_dict[self.k_v_scale_fusion.name] = {'shape': self.k_v_scale_fusion.shape,
                                                      'shard': key_value_t_scale_shard}
            state_dict[self.k_v_zp_fusion.name] = {'shape': self.k_v_zp_fusion.shape,
                                                   'shard': key_value_t_zp_shard}
        else:
            key_t_scale_shard = (self.tensor_parallel_group_size,)
            key_t_zp_shard = (self.tensor_parallel_group_size,)

            value_t_scale_shard = (self.tensor_parallel_group_size,)
            value_t_zp_shard = (self.tensor_parallel_group_size,)

            state_dict[self.k_scale_no_fusion.name] = {'shape': self.k_scale_no_fusion.shape,
                                                       'shard': key_t_scale_shard}
            state_dict[self.k_zp_no_fusion.name] = {'shape': self.k_zp_no_fusion.shape,
                                                    'shard': key_t_zp_shard}
            state_dict[self.v_scale_no_fusion.name] = {'shape': self.v_scale_no_fusion.shape,
                                                       'shard': value_t_scale_shard}
            state_dict[self.v_zp_no_fusion.name] = {'shape': self.v_zp_no_fusion.shape,
                                                    'shard': value_t_zp_shard}
        state_dict = self.sharded_input_quantizer_state_dict(state_dict)
        return state_dict

    def sharded_input_quantizer_state_dict(self, state_dict):
        """provide the sharded state dict based on the config"""

        key_input_quantizer_t_scale_shard = (self.tensor_parallel_group_size,)
        key_input_quantizer_t_zp_shard = (self.tensor_parallel_group_size,)
        value_input_quantizer_t_scale_shard = (self.tensor_parallel_group_size,)
        value_input_quantizer_t_zp_shard = (self.tensor_parallel_group_size,)

        state_dict[self._key_input_quantizer.t_scale.name] = {'shape': self._key_input_quantizer.t_scale.shape,
                                                              'shard': key_input_quantizer_t_scale_shard}
        state_dict[self._key_input_quantizer.t_zp.name] = {'shape': self._key_input_quantizer.t_zp.shape,
                                                           'shard': key_input_quantizer_t_zp_shard}
        state_dict[self._value_input_quantizer.t_scale.name] = {'shape': self._value_input_quantizer.t_scale.shape,
                                                                'shard': value_input_quantizer_t_scale_shard}
        state_dict[self._value_input_quantizer.t_zp.name] = {'shape': self._value_input_quantizer.t_zp.shape,
                                                             'shard': value_input_quantizer_t_zp_shard}
        return state_dict


class DeployDynamicQuantPagedAttentionCell(Cell):
    """DeployDynamicQuantPagedAttentionCell"""

    def __init__(self, kvcache: PagedAttentionMgr):
        super().__init__()
        self._kvcache = kvcache
        self.paged_attention = msops.auto_generate.PagedAttention(self._kvcache.n_heads,
                                                                  self._kvcache.scale_value,
                                                                  self._kvcache.n_kv_heads,
                                                                  "PERTOKEN")
        self.paged_attention_with_alibi = msops.auto_generate.PagedAttentionMask(self._kvcache.n_heads,
                                                                                 self._kvcache.scale_value,
                                                                                 self._kvcache.n_kv_heads,
                                                                                 "PERTOKEN")
        if "in_strategy" in kvcache.paged_attention.get_attr_dict():
            pa_strategy = kvcache.paged_attention.in_strategy
            self.paged_attention.shard(pa_strategy)

        if "in_strategy" in kvcache.paged_attention_with_alibi.get_attr_dict():
            pa_strategy = kvcache.paged_attention_with_alibi.in_strategy
            self.paged_attention_with_alibi.shard(pa_strategy)

    # pylint: disable=W0221
    def construct(self, key, value, slot_mapping):
        """The forward compute of KVCache for Paged Attention."""
        return self._kvcache.reshape_and_cache(key, value, self._kvcache.key_cache, self._kvcache.value_cache,
                                               slot_mapping)

    # pylint: disable=W0613
    def paged_attn(self, query, batch_valid_length, block_tables, attn_mask=None, q_seq_lens=None):
        """The forward compute of Paged Attention."""
        return self.paged_attention(query, self._kvcache.key_cache, self._kvcache.value_cache,
                                    block_tables, batch_valid_length)

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        return self.paged_attention_with_alibi(query, self._kvcache.key_cache, self._kvcache.value_cache,
                                               block_tables, batch_valid_length, None, None, alibi_tensor)
