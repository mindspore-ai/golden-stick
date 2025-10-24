# Copyright 2025 Huawei Technologies Co., Ltd
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
"""base class of mindformers quant model"""


import re
from tqdm import tqdm
import mindspore as ms
from mindspore import Parameter
from mindspore import ops as msops

from mindformers.parallel_core.inference.parallel_state import get_tensor_model_parallel_world_size


class MLAParamProcessor:
    """Split parameters of MLA block."""
    def __init__(self, network):
        self.network = network
        self.qkv_split_rules = (r'\.linear_qkv_down_proj\.',
                                '.linear_q_down_proj.', '.linear_kv_down_proj.')

    def _split_qkv_name(self, param_dict):
        """_split_qkv_name"""
        new_dict = {}
        pattern, *replacements = self.qkv_split_rules
        for key, value in param_dict.items():
            match = re.search(pattern, key)
            if not match:
                new_dict[key] = value
                continue
            for replacement in replacements:
                new_dict[re.sub(pattern, replacement, key)] = value
        return new_dict

    def _split_qkv_weight(self, param_dict):
        """Split linear_qkv_down to q_down_proj and kv_down_proj.
        Params q_down_proj and kv_down_proj are cloned so far, just do the narrow operation."""
        split_axis_map = {
            # {param_name: axis}
            'weight': 0,
            'weight_offset': 0,
            'weight_scale': 0,
            'deq_scale': 0,
            'quant_bias': 0,
            'bias': 0,
        }

        new_param_dict = {}
        param_name_trace = {}

        for key, value in tqdm(param_dict.items(), desc="split mla qkv weights"):
            match = re.search(self.qkv_split_rules[0], key)
            if not match:
                new_param_dict[key] = value
                continue
            q_down_name = re.sub(self.qkv_split_rules[0],
                                 self.qkv_split_rules[1],
                                 key)
            kv_down_name = re.sub(self.qkv_split_rules[0],
                                  self.qkv_split_rules[2],
                                  key)
            param_name = key.split('.')[-1]
            split_axis = split_axis_map.get(param_name, None)
            if split_axis is None:
                # No need to split, just clone the param to both gating and hidden
                new_param_dict[q_down_name] = value
                new_param_dict[kv_down_name] = value
                param_name_trace[q_down_name] = key
                param_name_trace[kv_down_name] = key
                continue
            # Split tensor
            q_lora_rank = self.network.config.q_lora_rank
            kv_lora_rank = self.network.config.kv_lora_rank
            qk_rope_head_dim = self.network.config.qk_rope_head_dim
            # q_down
            shard_offset = kv_lora_rank + qk_rope_head_dim
            shard_size = q_lora_rank
            new_param_dict[q_down_name] = Parameter(
                value.narrow(split_axis, shard_offset, shard_size))
            # kv_down
            shard_offset = 0
            shard_size = kv_lora_rank + qk_rope_head_dim
            new_param_dict[kv_down_name] = Parameter(
                value.narrow(split_axis, shard_offset, shard_size))

            param_name_trace[q_down_name] = key
            param_name_trace[kv_down_name] = key
        return new_param_dict, param_name_trace

    def _invert_trans_lkv2kv_weight(self, param_dict):
        """Permutate the kv_up_proj weight.
        Mutate in place. Change from kkkvvv to kvkvkv."""
        for key, value in param_dict.items():
            match = re.search(r'\.linear_kv_up_proj\.weight$', key)
            if not match:
                continue

            k_head_dim = self.network.config.qk_nope_head_dim
            v_head_dim = self.network.config.v_head_dim
            kv_lora_rank = self.network.config.kv_lora_rank
            num_heads = self.network.config.num_attention_heads

            tp_size = get_tensor_model_parallel_world_size()
            k_shard_size = (num_heads * k_head_dim) // tp_size
            v_shard_size = (num_heads * v_head_dim) // tp_size
            if k_shard_size + v_shard_size != value.shape[0]:
                raise ValueError(f'The sum of k_shard_size and v_shard_size should equal value.shape[0], '
                                 f'but the actual k_shard_size and v_shard_size are {k_shard_size} and {v_shard_size}, '
                                 f'and value.shape[0] is {value.shape[0]}.')
            k_tensor = value[:k_shard_size]
            v_tensor = value[k_shard_size:]
            k_tensor = k_tensor.reshape(-1, k_head_dim, kv_lora_rank)
            v_tensor = v_tensor.reshape(-1, v_head_dim, kv_lora_rank)
            kv_permutated = msops.stack((k_tensor, v_tensor), axis=1)
            kv_permutated = kv_permutated.reshape(-1, kv_lora_rank)
            param_dict[key] = Parameter(kv_permutated)
        return param_dict

    def _invert_trans_rope_weight(self, param_dict):
        """_invert_trans_rope_weight"""
        return param_dict

    def split_param(self, param_dict):
        """The split process of MLA parameters"""
        param_dict, param_name_trace = self._split_qkv_weight(param_dict)
        param_dict = self._invert_trans_rope_weight(param_dict)
        param_dict = self._invert_trans_lkv2kv_weight(param_dict)
        return param_dict, param_name_trace

    def split_name(self, param_dict):
        """The split process of MLA parameters"""
        param_dict = self._split_qkv_name(param_dict)
        return param_dict


class FFNParamProcessor:
    """Split parameters of FFN block."""
    def __init__(self, network):
        self.network = network
        self.split_rules = (r'\.linear_fc1\.',
                            '.gating.', '.hidden.')
        self.split_axis_map = {
            # {param_name: axis}
            'weight': -2,
            'weight_offset': 0,
            'weight_scale': 0,
            'deq_scale': 0,
            'quant_bias': 0,
            'bias': 0,
            'input_scale': None,  # Input quantization params don't need splitting in ColumnParallelLinear
            'input_offset': None,  # Input quantization params don't need splitting in ColumnParallelLinear
            'smooth_scale': None,  # Input quantization params don't need splitting in ColumnParallelLinear
        }

    def _split_ffn_name(self, param_dict, split_rules):
        """_split_ffn_name"""
        new_dict = {}
        pattern, *replacements = split_rules
        for key, value in param_dict.items():
            match = re.search(pattern, key)
            if not match:
                new_dict[key] = value
                continue
            for replacement in replacements:
                new_dict[re.sub(pattern, replacement, key)] = value
        return new_dict

    def _split_ffn_weight(self, param_dict, split_rules, split_axis_map):
        """Split linear_fc1 to gating and hidden.
        Params gating and hidden are cloned so far, just do the narrow operation.
        Returns: (new_param_dict, param_name_trace)"""
        new_param_dict = {}
        param_name_trace = {}
        for key, value in tqdm(param_dict.items(), desc="split mlp ffn weights"):
            match = re.search(split_rules[0], key)
            if not match:
                new_param_dict[key] = value
                continue

            gating_name = re.sub(split_rules[0],
                                 split_rules[1],
                                 key)
            hidden_name = re.sub(split_rules[0],
                                 split_rules[2],
                                 key)
            param_name = key.split('.')[-1]
            split_axis = split_axis_map.get(param_name, -1)
            if split_axis is None:
                # No need to split, just clone the param to both gating and hidden
                new_param_dict[gating_name] = value
                new_param_dict[hidden_name] = value
                # Update param_name_trace for both gating and hidden
                param_name_trace[gating_name] = key
                param_name_trace[hidden_name] = key
                continue
            # Split tensor
            shard_size = value.shape[split_axis] // 2
            # gating
            gating_shard_range = (0, shard_size)
            # hidden
            hidden_shard_range = (shard_size, shard_size)

            if value.dtype == ms.qint4x2:
                gating_start, gating_shared_size = gating_shard_range
                hidden_start, hidden_shared_size = hidden_shard_range
                new_param_dict[gating_name] = Parameter(ms.Tensor(
                    value.asnumpy()[gating_start:gating_start+gating_shared_size, :],
                    dtype=ms.qint4x2))
                new_param_dict[hidden_name] = Parameter(ms.Tensor(
                    value.asnumpy()[hidden_start:hidden_start+hidden_shared_size, :],
                    dtype=ms.qint4x2))
            else:
                new_param_dict[gating_name] = Parameter(value.narrow(split_axis, *gating_shard_range))
                new_param_dict[hidden_name] = Parameter(value.narrow(split_axis, *hidden_shard_range))

            # Update param_name_trace
            param_name_trace[gating_name] = key
            param_name_trace[hidden_name] = key

        return new_param_dict, param_name_trace

    def split_param(self, param_dict):
        """split_param"""
        param_name_trace = {}
        param_dict, cur_trace = self._split_ffn_weight(param_dict,
                                                       self.split_rules,
                                                       self.split_axis_map)
        param_name_trace.update(cur_trace)
        return param_dict, param_name_trace

    def split_name(self, param_dict):
        """The split process of FFN parameters"""
        param_dict = self._split_ffn_name(param_dict, self.split_rules)
        return param_dict


class MoeParamProcessor():
    """Parameter processor for MoE (Mixture of Experts) models."""
    def __init__(self, network):
        self.network = network
        # Try to get num_experts from different config attributes
        if hasattr(self.network.config, 'n_routed_experts'):
            self.num_experts = self.network.config.n_routed_experts
        elif hasattr(self.network.config, 'num_experts'):
            self.num_experts = self.network.config.num_experts
        elif hasattr(self.network.config, 'expert_num'):
            self.num_experts = self.network.config.expert_num
        else:
            raise RuntimeError("Could not find experts number in config.json")
        self.moe_split_rules = (r'\.mlp\.experts\.',
                                *(f'.mlp.experts.{i}.' for i in range(self.num_experts)))

    def _split_moe_name(self, param_dict):
        """_split_moe_name"""
        new_dict = {}
        pattern, *replacements = self.moe_split_rules
        for key, value in param_dict.items():
            match = re.search(pattern, key)
            if not match:
                new_dict[key] = value
                continue
            for replacement in replacements:
                new_dict[re.sub(pattern, replacement, key)] = value
        return new_dict

    def _split_route_moe_weight(self, param_dict):
        """Split merged routed moe to saperated experts.
        Params for each expert are cloned so far, just do the narrow operation.

        Args:
            param_dict (dict): Original parameter dictionary.

        Returns:
            tuple: A tuple containing (new_param_dict, param_name_trace), where
                  param_name_trace maps original parameter names to lists of split parameter names.
        """
        new_param_dict = {}
        param_name_trace = {}
        # need to transpose after split, i.e. [ic, oc] -> [oc, ic]
        need_transpose_params = ['weight', 'weight_scale', 'weight_offset']
        for key, value in tqdm(param_dict.items(), desc="split moe ffn weights"):
            match = re.search(self.moe_split_rules[0], key)
            if not match:
                new_param_dict[key] = value
                continue
            value_dtype = value.dtype
            for expert_id in range(self.num_experts):
                experts_name = re.sub(self.moe_split_rules[0],
                                      f'.mlp.experts.{expert_id}.', key)
                if value_dtype == ms.dtype.qint4x2:
                    experts_value = value.asnumpy()[expert_id]
                else:
                    experts_value = value[expert_id]

                param_name = experts_name.split('.')[-1]
                if param_name in need_transpose_params:
                    experts_value = experts_value.transpose()
                new_param_dict[experts_name] = Parameter(ms.Tensor(experts_value, dtype=value_dtype))
                param_name_trace[experts_name] = key
        return new_param_dict, param_name_trace

    def split_param(self, param_dict):
        """Split parameters for MoE model and return parameter name trace.

        Args:
            param_dict (dict): Original parameter dictionary.

        Returns:
            tuple: A tuple containing (new_param_dict, param_name_trace), where
                  param_name_trace maps original parameter names to lists of split parameter names.
        """
        return self._split_route_moe_weight(param_dict)

    def split_name(self, param_dict):
        """split_name"""
        param_dict = self._split_moe_name(param_dict)
        return param_dict


class QKVParamProcessor:
    """Split parameters of QKV block for GQA/MHA networks."""
    def __init__(self, network):
        self.network = network
        self.qkv_split_rules = (r'\.linear_qkv\.',
                                '.linear_q.', '.linear_k.', '.linear_v.')
        self.qkv_split_axis_map = {
            # {param_name: axis}
            'weight': 0,
            'weight_offset': 0,
            'weight_scale': 0,
            'deq_scale': 0,
            'quant_bias': 0,
            'bias': 0,
            'input_scale': None,   # Input quantization params don't need splitting
            'input_offset': None,  # Input quantization params don't need splitting
            'smooth_scale': None,  # Input quantization params don't need splitting
        }

    def _split_qkv_name(self, param_dict):
        """Split linear_qkv parameter names to linear_q, linear_k, linear_v."""
        new_dict = {}
        pattern, *replacements = self.qkv_split_rules
        for key, value in param_dict.items():
            match = re.search(pattern, key)
            if not match:
                new_dict[key] = value
                continue
            for replacement in replacements:
                new_dict[re.sub(pattern, replacement, key)] = value
        return new_dict

    def _split_qkv_weight(self, param_dict):
        """Split linear_qkv to linear_q, linear_k, linear_v.
        QKV split is performed on axis 0, with sizes determined from network config.
        Returns: (new_param_dict, param_name_trace)"""
        new_param_dict = {}
        param_name_trace = {}

        for key, value in tqdm(param_dict.items(), desc="split attn qkv weights"):
            match = re.search(self.qkv_split_rules[0], key)
            if not match:
                new_param_dict[key] = value
                continue

            q_name = re.sub(self.qkv_split_rules[0], self.qkv_split_rules[1], key)
            k_name = re.sub(self.qkv_split_rules[0], self.qkv_split_rules[2], key)
            v_name = re.sub(self.qkv_split_rules[0], self.qkv_split_rules[3], key)

            param_name = key.split('.')[-1]
            split_axis = self.qkv_split_axis_map.get(param_name, None)
            if split_axis is None:
                # No need to split, just clone the param to Q, K, V
                new_param_dict[q_name] = value
                new_param_dict[k_name] = value
                new_param_dict[v_name] = value
                param_name_trace[q_name] = key
                param_name_trace[k_name] = key
                param_name_trace[v_name] = key
                continue

            # Get dimensions from network config
            num_heads = self.network.config.num_attention_heads
            num_key_value_heads = getattr(self.network.config, 'num_key_value_heads', num_heads)
            head_dim = getattr(self.network.config, 'head_dim', self.network.config.hidden_size // num_heads)

            # Get tensor parallel world size for proper dimension calculation
            tensor_parallel_world_size = get_tensor_model_parallel_world_size()
            if tensor_parallel_world_size is None or tensor_parallel_world_size <= 0:
                tensor_parallel_world_size = 1

            # Calculate split sizes considering tensor parallelism
            q_size = (num_heads * head_dim) // tensor_parallel_world_size
            k_size = (num_key_value_heads * head_dim) // tensor_parallel_world_size
            v_size = (num_key_value_heads * head_dim) // tensor_parallel_world_size

            # Split tensor on axis 0
            if value.dtype == ms.qint4x2:
                # Handle quantized tensors
                q_end = q_size
                k_end = q_end + k_size
                v_end = k_end + v_size

                if len(value.shape) == 2:
                    new_param_dict[q_name] = Parameter(ms.Tensor(
                        value.asnumpy()[0:q_end, :], dtype=ms.qint4x2))
                    new_param_dict[k_name] = Parameter(ms.Tensor(
                        value.asnumpy()[q_end:k_end, :], dtype=ms.qint4x2))
                    new_param_dict[v_name] = Parameter(ms.Tensor(
                        value.asnumpy()[k_end:v_end, :], dtype=ms.qint4x2))
                elif len(value.shape) == 3:
                    new_param_dict[q_name] = Parameter(ms.Tensor(
                        value.asnumpy()[:, 0:q_end, :], dtype=ms.qint4x2))
                    new_param_dict[k_name] = Parameter(ms.Tensor(
                        value.asnumpy()[:, q_end:k_end, :], dtype=ms.qint4x2))
                    new_param_dict[v_name] = Parameter(ms.Tensor(
                        value.asnumpy()[:, k_end:v_end, :], dtype=ms.qint4x2))
                else:
                    raise ValueError(f"Unexpected value shape: {value.shape} of key {key}")
            else:
                # Handle regular tensors using narrow operation
                new_param_dict[q_name] = Parameter(value.narrow(split_axis, 0, q_size))
                new_param_dict[k_name] = Parameter(value.narrow(split_axis, q_size, k_size))
                new_param_dict[v_name] = Parameter(value.narrow(split_axis, q_size + k_size, v_size))

            # Update param_name_trace
            param_name_trace[q_name] = key
            param_name_trace[k_name] = key
            param_name_trace[v_name] = key

        return new_param_dict, param_name_trace

    def split_param(self, param_dict):
        """split_param"""
        return self._split_qkv_weight(param_dict)

    def split_name(self, param_dict):
        """The split process of QKV parameter names"""
        param_dict = self._split_qkv_name(param_dict)
        return param_dict
