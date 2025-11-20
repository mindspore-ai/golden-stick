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
"""WeightProcessor"""


import os
import json
import gc
from safetensors import safe_open
import numpy as np
from tqdm import tqdm
import mindspore as ms
from mindspore.communication.management import get_rank, get_group_size
from mindspore_gs.common import logger


class WeightProcessor:
    r"""
    Provide Qwen3 Model weight load and shards.
    """
    def __init__(self):
        self.config = None
        self.num_layers = 0
        self.num_experts = 0
        self.tie_word_embeddings = False
        self.tp_group_size = get_group_size()
        self.rank_id = get_rank()
        self._np_dict = {}
        self._src_dir = None
        self._param_map = None
        self.parameter_dict = {}
        self.handled_keys = []
        self.file_handles = {}

    def get_file_handles(self, filename):
        if filename not in self.file_handles:
            fp = safe_open(filename, framework="np")
            self.file_handles[filename] = fp
        return self.file_handles[filename]

    def release_file_handles(self):
        del self.file_handles

    def get_safetensor_from_file(self, hf_param_name, src_hf_dir, hf_weight_map, is_split_param=False, split_axis=0):
        """get_safetensor_from_file"""
        safetensor_file = hf_weight_map[hf_param_name]
        filename = os.path.join(src_hf_dir, safetensor_file)
        sf_file = self.get_file_handles(filename)
        qint4 = False
        if sf_file.metadata() is not None and hf_param_name in sf_file.metadata().keys():
            qint4 = True
        if not is_split_param:
            np_data = sf_file.get_tensor(hf_param_name)
            self.handled_keys.append(hf_param_name)
            return np_data, qint4

        np_data = sf_file.get_slice(hf_param_name)
        shape = np_data.get_shape()
        if split_axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = np_data[start:stop]
        elif split_axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = np_data[:, start:stop]
        elif split_axis == 2:
            split_size = shape[2] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = np_data[:, :, start:stop]
        else:
            raise ValueError(f"split_axis:{split_axis} is not supported.")
        self.handled_keys.append(hf_param_name)
        return split_data, qint4

    def _get_weight_slice(self, weight, axis):
        """_get_weight_slice"""
        shape = weight.shape
        if axis == 0:
            split_size = shape[0] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = weight[start:stop]
        elif axis == 1:
            split_size = shape[1] // self.tp_group_size
            start = self.rank_id * split_size
            stop = (self.rank_id + 1) * split_size
            split_data = weight[:, start:stop]
        else:
            raise ValueError(f"axis:{axis} is not supported.")
        return split_data

    def _get_split_set(self, name, split_axis=-1):
        """_get_split_set"""
        hf_name = self._convert_param_names_to_hf(name)
        if self._param_map.get(hf_name) is None:
            logger.debug(f"No parameter named {hf_name} in safetensors, skip.")
            return
        is_split_param = split_axis != -1
        np_data, _ = self.get_safetensor_from_file(hf_name, self._src_dir, self._param_map, is_split_param, split_axis)
        self._np_dict[name] = np_data

    def _split_outer_weight(self):
        """_split_outer_weight"""
        self._get_split_set("model.embedding.word_embeddings.weight", 0)
        self._get_split_set("model.decoder.final_layernorm.weight", -1)
        if not self.tie_word_embeddings:
            self._get_split_set("model.output_layer.weight", 0)

    def _split_moe_weight(self, layer_id):
        """_split_moe_ffn_weight"""
        for i in range(self.num_experts):
            # fc1
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight", 1)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.gating.weight", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.hidden.weight", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_scale", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.gating.weight_scale", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.hidden.weight_scale", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_offset", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.gating.weight_offset", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.hidden.weight_offset", 0)
            # fc2
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight", 1)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight_scale", -1)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight_offset", -1)
        pre_mlpnorm_key = f"model.decoder.layers.{layer_id}.pre_mlp_layernorm.weight"
        self._get_split_set(pre_mlpnorm_key, -1)
        post_mlpnorm_key = f"model.decoder.layers.{layer_id}.post_mlp_layernorm.weight"
        self._get_split_set(post_mlpnorm_key, -1)
        router_bias = f"model.decoder.layers.{layer_id}.mlp.router.expert_bias"
        self._get_split_set(router_bias, -1)
        router_weight = f"model.decoder.layers.{layer_id}.mlp.router.weight"
        self._get_split_set(router_weight, -1)

    def _split_mlp_weight(self, layer_id):
        """_split_dense_ffn_weight"""
        # Check if split FFN parameters exist in safetensors
        ffn_names = [
            f"model.decoder.layers.{layer_id}.mlp.gating.weight",
            f"model.decoder.layers.{layer_id}.mlp.hidden.weight",
            f"model.decoder.layers.{layer_id}.mlp.gating.bias",
            f"model.decoder.layers.{layer_id}.mlp.hidden.bias"
        ]
        hf_ffn_names = [self._convert_param_names_to_hf(name) for name in ffn_names]
        ffn_split_exists = any(hf_name in self._param_map for hf_name in hf_ffn_names)

        if ffn_split_exists:
            # Load split FFN parameters (gating, hidden)
            logger.debug(f"Loading split FFN parameters for layer {layer_id}")
            param_suffixes = ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias',
                              'input_scale', 'input_offset', 'smooth_scale', 'bias']

            for suffix in param_suffixes:
                # Determine split_axis based on parameter type
                if suffix in ['weight']:
                    split_axis = 0  # FFN weight split axis
                elif suffix in ['weight_scale', 'weight_offset', 'deq_scale', 'quant_bias', 'bias']:
                    split_axis = 0  # FFN quantization parameters split axis
                else:
                    split_axis = -1  # No split needed

                # Load split gating and hidden parameters
                self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.gating.{suffix}", split_axis)
                self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.hidden.{suffix}", split_axis)
        else:
            # Load original FFN parameters (linear_fc1) for models without FFN split
            logger.debug(f"Loading original FFN parameters for layer {layer_id}")
            param_suffixes = ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias',
                              'input_scale', 'input_offset', 'smooth_scale', 'bias']

            for suffix in param_suffixes:
                # Determine split_axis based on parameter type
                if suffix in ['weight']:
                    split_axis = 0  # FFN weight split axis
                elif suffix in ['weight_scale', 'weight_offset', 'deq_scale', 'quant_bias', 'bias']:
                    split_axis = 0  # FFN quantization parameters split axis
                else:
                    split_axis = -1  # No split needed

                self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc1.{suffix}", split_axis)
        # fc2
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc2.weight", 1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc2.bias", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc2.weight_scale", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc2.weight_offset", -1)
        pre_mlpnorm_key = f"model.decoder.layers.{layer_id}.pre_mlp_layernorm.weight"
        self._get_split_set(pre_mlpnorm_key, -1)
        post_mlpnorm_key = f"model.decoder.layers.{layer_id}.post_mlp_layernorm.weight"
        self._get_split_set(post_mlpnorm_key, -1)

    def _split_attention_weight(self, layer_id):
        """_split_attention_weight"""
        # Check if split QKV parameters exist in safetensors
        qkv_names = [
            f"model.decoder.layers.{layer_id}.self_attention.linear_q.weight",
            f"model.decoder.layers.{layer_id}.self_attention.linear_k.weight",
            f"model.decoder.layers.{layer_id}.self_attention.linear_v.weight",
            f"model.decoder.layers.{layer_id}.self_attention.linear_q.bias",
            f"model.decoder.layers.{layer_id}.self_attention.linear_k.bias",
            f"model.decoder.layers.{layer_id}.self_attention.linear_v.bias"
        ]
        hf_qkv_names = [self._convert_param_names_to_hf(name) for name in qkv_names]
        qkv_split_exists = any(hf_name in self._param_map for hf_name in hf_qkv_names)

        if qkv_split_exists:
            # Load split QKV parameters (linear_q, linear_k, linear_v)
            logger.debug(f"Loading split QKV parameters for layer {layer_id}")
            param_suffixes = ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias',
                              'input_scale', 'input_offset', 'smooth_scale', 'bias']

            for suffix in param_suffixes:
                # Determine split_axis based on parameter type
                if suffix in ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias', 'bias']:
                    split_axis = 0  # QKV split axis
                else:
                    split_axis = -1  # No split needed

                # Load split Q, K, V parameters
                self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_q.{suffix}", split_axis)
                self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_k.{suffix}", split_axis)
                self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_v.{suffix}", split_axis)
        else:
            # Load original QKV parameters (linear_qkv) for models without QKV split
            logger.debug(f"Loading original QKV parameters for layer {layer_id}")
            param_suffixes = ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias',
                              'input_scale', 'input_offset', 'smooth_scale', 'bias']

            for suffix in param_suffixes:
                # Determine split_axis based on parameter type
                if suffix in ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias', 'bias']:
                    split_axis = 0  # QKV split axis
                else:
                    split_axis = -1  # No split needed

                self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.{suffix}", split_axis)
        # wo
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.weight", 1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.bias", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.input_scale", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.input_offset", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.weight_scale", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.weight_offset", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.smooth_scale", 0)

        inputnorm_key = f"model.decoder.layers.{layer_id}.input_layernorm.weight"
        self._get_split_set(inputnorm_key, -1)
        post_attn_norm = f"model.decoder.layers.{layer_id}.post_self_attn_layernorm.weight"
        self._get_split_set(post_attn_norm, -1)
        qnorm_key = f"model.decoder.layers.{layer_id}.self_attention.q_layernorm.weight"
        self._get_split_set(qnorm_key, -1)
        knorm_key = f"model.decoder.layers.{layer_id}.self_attention.k_layernorm.weight"
        self._get_split_set(knorm_key, -1)

    def _split_weight_of_each_layer(self, layer_id):
        """_split_weight_of_each_layer"""
        self._split_attention_weight(layer_id)
        if self.num_experts > 0:
            self._split_moe_weight(layer_id)
        else:
            self._split_mlp_weight(layer_id)

    def _split_weight(self):
        """_split_weight"""
        self._split_outer_weight()
        num_layers = self.num_layers
        enable_tqdm = self.rank_id == 0
        for layer_id in tqdm(range(num_layers), desc="Load weights", disable=not enable_tqdm):
            self._split_weight_of_each_layer(layer_id)

    def _qkv_merge_of_each_layer(self, layer_id):
        """Merge split QKV parameters back to linear_qkv for each layer"""
        # Define parameter suffixes that need to be merged
        # Note: input_scale, input_offset, smooth_scale are input-related parameters
        # that should NOT be merged because they correspond to input channels
        # and are fully replicated across all devices
        merge_param_suffixes = ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias', 'bias']
        input_param_suffixes = ['input_scale', 'input_offset', 'smooth_scale']

        for suffix in input_param_suffixes:
            q_key = f"model.decoder.layers.{layer_id}.self_attention.linear_q.{suffix}"
            k_key = f"model.decoder.layers.{layer_id}.self_attention.linear_k.{suffix}"
            v_key = f"model.decoder.layers.{layer_id}.self_attention.linear_v.{suffix}"
            qkv_key = f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.{suffix}"
            # Check if all three components exist with detailed logging
            missing_keys = []
            if q_key not in self._np_dict:
                missing_keys.append(q_key)
            if k_key not in self._np_dict:
                missing_keys.append(k_key)
            if v_key not in self._np_dict:
                missing_keys.append(v_key)

            if missing_keys:
                logger.debug(f"Layer {layer_id}, suffix {suffix}: Missing keys {missing_keys}")
                continue
            # Execute merge operation
            q_param = self._np_dict.pop(q_key)
            _ = self._np_dict.pop(k_key)
            _ = self._np_dict.pop(v_key)
            self._np_dict[qkv_key] = q_param

        for suffix in merge_param_suffixes:
            q_key = f"model.decoder.layers.{layer_id}.self_attention.linear_q.{suffix}"
            k_key = f"model.decoder.layers.{layer_id}.self_attention.linear_k.{suffix}"
            v_key = f"model.decoder.layers.{layer_id}.self_attention.linear_v.{suffix}"
            qkv_key = f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.{suffix}"
            # Check if all three components exist with detailed logging
            missing_keys = []
            if q_key not in self._np_dict:
                missing_keys.append(q_key)
            if k_key not in self._np_dict:
                missing_keys.append(k_key)
            if v_key not in self._np_dict:
                missing_keys.append(v_key)
            if missing_keys:
                logger.debug(f"Layer {layer_id}, suffix {suffix}: Missing keys {missing_keys}")
                continue
            # Execute merge operation
            q_param = self._np_dict.pop(q_key)
            k_param = self._np_dict.pop(k_key)
            v_param = self._np_dict.pop(v_key)
            # Log parameter shapes for debugging
            logger.debug(f"Layer {layer_id}, {suffix}: Q shape {q_param.shape}, "
                         f"K shape {k_param.shape}, V shape {v_param.shape}")

            # Concatenate along axis 0 (QKV split axis)
            merged_param = np.concatenate((q_param, k_param, v_param), axis=0)
            self._np_dict[qkv_key] = merged_param
            logger.debug(f"Successfully merged {suffix} for layer {layer_id}, final shape: {merged_param.shape}")

    def _qkv_merge(self):
        """Merge split QKV parameters back to linear_qkv"""
        # Check if any split QKV parameters exist in the loaded data
        qkv_split_exists = any(
            key for key in self._np_dict
            if 'linear_q.' in key or 'linear_k.' in key or 'linear_v.' in key
        )

        if not qkv_split_exists:
            logger.debug("No split QKV parameters found, skipping QKV merge")
            return

        logger.debug("Split QKV parameters detected, performing merge")
        num_layers = self.num_layers
        enable_tqdm = self.rank_id == 0
        for layer_id in tqdm(range(num_layers), desc="Merge QKV weights", disable=not enable_tqdm):
            self._qkv_merge_of_each_layer(layer_id)

    def _ffn_merge_moe_experts(self, layer_id, expert_indices):
        """Merge split FFN parameters back to linear_fc1 for MoE experts"""
        # Define parameter suffixes that need to be merged
        merge_param_suffixes = ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias', 'bias']
        input_param_suffixes = ['input_scale', 'input_offset', 'smooth_scale']

        logger.debug(f"Layer {layer_id}: MoE experts found, performing FFN merge for each expert")

        # Process each expert
        for expert_idx in expert_indices:
            expert_prefix = f"model.decoder.layers.{layer_id}.mlp.experts.{expert_idx}."

            # Check if this expert has gating/hidden split
            has_gating_hidden = any(
                key for key in self._np_dict
                if key.startswith(expert_prefix) and ("gating." in key or "hidden." in key)
            )

            if has_gating_hidden:
                logger.debug(f"Processing FFN merge for layer {layer_id}, expert {expert_idx}")

                # Process input parameters
                for suffix in input_param_suffixes:
                    gating_key = f"{expert_prefix}gating.{suffix}"
                    hidden_key = f"{expert_prefix}hidden.{suffix}"
                    fc1_key = f"{expert_prefix}linear_fc1.{suffix}"
                    missing_keys = []
                    if gating_key not in self._np_dict:
                        missing_keys.append(gating_key)
                    if hidden_key not in self._np_dict:
                        missing_keys.append(hidden_key)
                    if missing_keys:
                        logger.debug(f"Expert {expert_idx}, suffix {suffix}: Missing keys {missing_keys}")
                        continue

                    # Execute merge operation
                    gate_param = self._np_dict.pop(gating_key)
                    _ = self._np_dict.pop(hidden_key)
                    self._np_dict[fc1_key] = gate_param

                # Process merge parameters
                for suffix in merge_param_suffixes:
                    gating_key = f"{expert_prefix}gating.{suffix}"
                    hidden_key = f"{expert_prefix}hidden.{suffix}"
                    fc1_key = f"{expert_prefix}linear_fc1.{suffix}"

                    missing_keys = []
                    if gating_key not in self._np_dict:
                        missing_keys.append(gating_key)
                    if hidden_key not in self._np_dict:
                        missing_keys.append(hidden_key)
                    if missing_keys:
                        logger.debug(f"Expert {expert_idx}, suffix {suffix}: Missing keys {missing_keys}")
                        continue

                    # Execute merge operation
                    gating_param = self._np_dict.pop(gating_key)
                    hidden_param = self._np_dict.pop(hidden_key)

                    # Log parameter shapes for debugging
                    logger.debug(f"Expert {expert_idx}, {suffix}: Gating shape {gating_param.shape}, "
                                 f"Hidden shape {hidden_param.shape}")

                    # Concatenate along axis 0 (FFN split axis)
                    merged_param = np.concatenate((gating_param, hidden_param), axis=0)
                    self._np_dict[fc1_key] = merged_param
                    logger.debug(f"Successfully merged FFN {suffix} for expert {expert_idx}, "
                                 f"final shape: {merged_param.shape}")

    def _ffn_merge_normal(self, layer_id):
        """Merge split FFN parameters back to linear_fc1 for normal (non-MoE) layers"""
        # Define parameter suffixes that need to be merged
        merge_param_suffixes = ['weight', 'weight_scale', 'weight_offset', 'deq_scale', 'quant_bias', 'bias']
        input_param_suffixes = ['input_scale', 'input_offset', 'smooth_scale']

        for suffix in input_param_suffixes:
            gating_key = f"model.decoder.layers.{layer_id}.mlp.gating.{suffix}"
            hidden_key = f"model.decoder.layers.{layer_id}.mlp.hidden.{suffix}"
            fc1_key = f"model.decoder.layers.{layer_id}.mlp.linear_fc1.{suffix}"
            # Check if all three components exist with detailed logging
            missing_keys = []
            if gating_key not in self._np_dict:
                missing_keys.append(gating_key)
            if hidden_key not in self._np_dict:
                missing_keys.append(hidden_key)
            if missing_keys:
                logger.debug(f"Layer {layer_id}, suffix {suffix}: Missing keys {missing_keys}")
                continue
            # Execute merge operation
            gate_param = self._np_dict.pop(gating_key)
            _ = self._np_dict.pop(hidden_key)
            self._np_dict[fc1_key] = gate_param

        for suffix in merge_param_suffixes:
            gating_key = f"model.decoder.layers.{layer_id}.mlp.gating.{suffix}"
            hidden_key = f"model.decoder.layers.{layer_id}.mlp.hidden.{suffix}"
            fc1_key = f"model.decoder.layers.{layer_id}.mlp.linear_fc1.{suffix}"
            missing_keys = []
            if gating_key not in self._np_dict:
                missing_keys.append(gating_key)
            if hidden_key not in self._np_dict:
                missing_keys.append(hidden_key)
            if missing_keys:
                logger.debug(f"Layer {layer_id}, suffix {suffix}: Missing keys {missing_keys}")
                continue
            # Execute merge operation
            gating_param = self._np_dict.pop(gating_key)
            hidden_param = self._np_dict.pop(hidden_key)

            # Log parameter shapes for debugging
            logger.debug(f"Layer {layer_id}, {suffix}: Gating shape {gating_param.shape}, "
                         f"Hidden shape {hidden_param.shape}")

            # Concatenate along axis 0 (FFN split axis)
            merged_param = np.concatenate((gating_param, hidden_param), axis=0)
            self._np_dict[fc1_key] = merged_param
            logger.debug(f"Successfully merged FFN {suffix} for layer {layer_id}, final shape: {merged_param.shape}")

    def _ffn_merge_of_each_layer(self, layer_id):
        """Merge split FFN parameters back to linear_fc1 for each layer"""
        # Check if MoE experts exist for this layer
        moe_experts_exist = any(
            key for key in self._np_dict
            if f"model.decoder.layers.{layer_id}.mlp.experts." in key
        )

        # If MoE experts exist for this layer, process each expert
        if moe_experts_exist:
            # Find all expert indices for this layer
            expert_indices = set()
            for key in self._np_dict:
                if f"model.decoder.layers.{layer_id}.mlp.experts." not in key:
                    continue
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part == "experts" and i+1 < len(parts):
                        try:
                            expert_idx = int(parts[i+1])
                            expert_indices.add(expert_idx)
                        except ValueError:
                            pass

            # Process MoE experts
            self._ffn_merge_moe_experts(layer_id, expert_indices)
            return

        # Process normal FFN (non-MoE) parameters
        self._ffn_merge_normal(layer_id)

    def _ffn_merge(self):
        """Merge split FFN parameters back to linear_fc1"""
        # Check if any split FFN parameters exist in the loaded data
        ffn_split_exists = any(
            key for key in self._np_dict
            if 'gating.' in key or 'hidden.' in key
        )

        if not ffn_split_exists:
            logger.debug("No split FFN parameters found, skipping FFN merge")
            return

        logger.debug("Split FFN parameters detected, performing merge")
        num_layers = self.num_layers
        enable_tqdm = self.rank_id == 0
        for layer_id in tqdm(range(num_layers), desc="Merge FFN weights", disable=not enable_tqdm):
            self._ffn_merge_of_each_layer(layer_id)

    def _moe_merge_of_each_layer(self, layer_id):
        """_qkv_concat_of_each_layer"""
        separate_experts_exist = any(
            key for key in self._np_dict
            if f"model.decoder.layers.{layer_id}.mlp.experts.0." in key
        )
        if not separate_experts_exist:
            return

        fc1_weights = []
        fc1_weight_scales = []
        fc1_weight_offsets = []
        fc2_weights = []
        fc2_weight_scales = []
        fc2_weight_offsets = []
        for i in range(self.num_experts):
            # fc1
            fc1 = self._np_dict.pop(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight")
            fc1 = fc1.transpose()
            fc1_weights.append(np.expand_dims(fc1, 0))
            fc1_w_scale = self._np_dict.pop(
                f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_scale")
            # Check if weight_scale is per_group quantized (has 2 dimensions)
            if len(fc1_w_scale.shape) > 1:
                fc1_w_scale = fc1_w_scale.transpose()
            fc1_weight_scales.append(np.expand_dims(fc1_w_scale, 0))
            fc1_w_offset = self._np_dict.pop(
                f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_offset")
            # Check if weight_offset is per_group quantized (has 2 dimensions)
            if len(fc1_w_offset.shape) > 1:
                fc1_w_offset = fc1_w_offset.transpose()
            fc1_weight_offsets.append(np.expand_dims(fc1_w_offset, 0))
            # fc2
            fc2 = self._np_dict.pop(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight")
            fc2 = fc2.transpose()
            fc2_weights.append(np.expand_dims(fc2, 0))
            # Handle fc2 weight_scale if exists
            fc2_w_scale_key = f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight_scale"
            if fc2_w_scale_key in self._np_dict:
                fc2_w_scale = self._np_dict.pop(fc2_w_scale_key)
                # Check if weight_scale is per_group quantized (has 2 dimensions)
                if len(fc2_w_scale.shape) > 1:
                    fc2_w_scale = fc2_w_scale.transpose()
                fc2_weight_scales.append(np.expand_dims(fc2_w_scale, 0))
            # Handle fc2 weight_offset if exists
            fc2_w_offset_key = f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight_offset"
            if fc2_w_offset_key in self._np_dict:
                fc2_w_offset = self._np_dict.pop(fc2_w_offset_key)
                # Check if weight_offset is per_group quantized (has 2 dimensions)
                if len(fc2_w_offset.shape) > 1:
                    fc2_w_offset = fc2_w_offset.transpose()
                fc2_weight_offsets.append(np.expand_dims(fc2_w_offset, 0))

        fc1_key = f"model.decoder.layers.{layer_id}.mlp.experts.linear_fc1"
        fc2_key = f"model.decoder.layers.{layer_id}.mlp.experts.linear_fc2"
        self._np_dict[f"{fc1_key}.weight"] = np.concatenate(tuple(fc1_weights), axis=0)
        self._np_dict[f"{fc1_key}.weight_scale"] = np.concatenate(tuple(fc1_weight_scales), axis=0)
        self._np_dict[f"{fc1_key}.weight_offset"] = np.concatenate(tuple(fc1_weight_offsets), axis=0)
        self._np_dict[f"{fc2_key}.weight"] = np.concatenate(tuple(fc2_weights), axis=0)
        if fc2_weight_scales:
            self._np_dict[f"{fc2_key}.weight_scale"] = np.concatenate(tuple(fc2_weight_scales), axis=0)
        if fc2_weight_offsets:
            self._np_dict[f"{fc2_key}.weight_offset"] = np.concatenate(tuple(fc2_weight_offsets), axis=0)

    def _moe_merge(self):
        """_moe_merge"""
        if self.num_experts == 0:
            logger.info("No experts in network, skip MoE weight concat.")
            return
        num_layers = self.num_layers
        enable_tqdm = self.rank_id == 0
        for layer_id in tqdm(range(num_layers), desc="Merge MoE weights", disable=not enable_tqdm):
            self._moe_merge_of_each_layer(layer_id)

    def _load_param(self, network):
        """_load_param"""
        cast_map = {}

        enable_tqdm = self.rank_id == 0
        for key, value in tqdm(self._np_dict.items(), desc="Create params", disable=not enable_tqdm):
            param = ms.from_numpy(value)
            cast_dtype = cast_map.get(key)
            if cast_dtype:
                param.astype(cast_dtype)
            self.parameter_dict[key] = ms.Parameter(param, name=key, requires_grad=False)
        self._np_dict.clear()

        ms.load_param_into_net(network, self.parameter_dict)
        self.parameter_dict.clear()
        del self.parameter_dict
        gc.collect()

    def _del_experts_weight(self, network):
        """_del_experts_weight"""
        experts_dict = {k: v for k, v in self.parameter_dict.items()
                        if ".mlp.experts." in k}
        is_fc1_quant = any(".linear_fc1.weight_scale" in k for k in experts_dict)
        is_fc2_quant = any(".linear_fc2.weight_scale" in k for k in experts_dict)
        def process(root, name_prefix):
            """Iterate the whole network and call callback function `process_cell`."""
            if root is None:
                return
            for name, cell in root.name_cells().items():
                full_cell_name = f"{name_prefix}.{name}"
                if is_fc1_quant and hasattr(cell, "weight1"):
                    del cell.weight1
                    cell.weight1 = None
                if is_fc2_quant and hasattr(cell, "weight2"):
                    del cell.weight2
                    cell.weight2 = None
                process(cell, full_cell_name)
        process(network, 'network')

    def load_safetensors_shard(self, src_hf_dir, network, convert_param_names_to_hf):
        """qwen load safetensors and shard """
        self._src_dir = src_hf_dir
        self._convert_param_names_to_hf = convert_param_names_to_hf

        index_json_path = os.path.join(src_hf_dir, 'model.safetensors.index.json')
        if not os.path.exists(index_json_path):
            raise RuntimeError(f"Not found index json file: {index_json_path}")
        with open(index_json_path, "r", encoding="utf-8") as fp:
            self._param_map = json.load(fp)['weight_map']

        config_json_path = os.path.join(src_hf_dir, 'config.json')
        if not os.path.exists(config_json_path):
            raise RuntimeError(f"Not found config json file: {config_json_path}")
        with open(config_json_path, "r", encoding="utf-8") as fp:
            self.config = json.load(fp)
        if 'num_layers' in self.config:
            self.num_layers = self.config['num_layers']
        elif 'num_hidden_layers' in self.config:
            self.num_layers = self.config['num_hidden_layers']
        elif 'n_layer' in self.config:
            self.num_layers = self.config['n_layer']
        else:
            raise RuntimeError("Can not found num_layers info in config.json.")
        if 'num_experts' in self.config:
            self.num_experts = self.config['num_experts']
        else:
            logger.info("Not found any experts info in config.json, set num_experts to zero.")
        self.tie_word_embeddings = self.config.get('tie_word_embeddings', False)

        self._split_weight()
        self._qkv_merge()
        self._ffn_merge()
        self._moe_merge()
        self._del_experts_weight(network)
        self._load_param(network)

        logger.info(f"These parameters in safetensors are not used: {self._param_map.keys() - self.handled_keys}")
