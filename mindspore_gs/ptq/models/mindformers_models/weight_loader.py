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
            raise ValueError("split_axis:{} is not supported.".format(split_axis))
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
            raise ValueError("axis:{} is not supported.".format(axis))
        return split_data

    def _get_split_set(self, name, split_axis=-1):
        """_get_split_set"""
        if self._param_map.get(name) is None:
            logger.debug(f"No parameter named {name} in safetensors, skip.")
            return
        is_split_param = split_axis != -1
        np_data, _ = self.get_safetensor_from_file(name, self._src_dir, self._param_map, is_split_param, split_axis)
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
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_scale", 0)
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_offset", 0)
            # fc2
            self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight", 0)
        mlpnorm_key = f"model.decoder.layers.{layer_id}.pre_mlp_layernorm.weight"
        self._get_split_set(mlpnorm_key, -1)

    def _split_mlp_weight(self, layer_id):
        """_split_dense_ffn_weight"""
        # fc1
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc1.weight", 1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc1.weight_scale", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc1.weight_offset", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc1.input_scale", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc1.input_offset", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc1.smooth_scale", -1)
        # fc2
        self._get_split_set(f"model.decoder.layers.{layer_id}.mlp.linear_fc2.weight", 0)
        mlpnorm_key = f"model.decoder.layers.{layer_id}.pre_mlp_layernorm.weight"
        self._get_split_set(mlpnorm_key, -1)

    def _split_attention_weight(self, layer_id):
        """_split_attention_weight"""
        # wqkv
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.weight", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.input_scale", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.input_offset", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.weight_scale", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.weight_offset", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_qkv.smooth_scale", -1)
        # wo
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.weight", 1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.input_scale", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.input_offset", 0)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.weight_scale", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.weight_offset", -1)
        self._get_split_set(f"model.decoder.layers.{layer_id}.self_attention.linear_proj.smooth_scale", 0)

        inputnorm_key = f"model.decoder.layers.{layer_id}.input_layernorm.weight"
        self._get_split_set(inputnorm_key, -1)
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

    def _qkv_concat_of_each_layer(self, layer_id):
        """_qkv_concat_of_each_layer"""
        wq_key = f"model.layers.{layer_id}.attention.wq._layer.weight"
        wk_key = f"model.layers.{layer_id}.attention.wk._layer.weight"
        wv_key = f"model.layers.{layer_id}.attention.wv._layer.weight"
        w_qkv_key = f"model.layers.{layer_id}.attention.w_qkv._layer.weight"
        wq = self._np_dict.pop(wq_key)
        wk = self._np_dict.pop(wk_key)
        wv = self._np_dict.pop(wv_key)
        self._np_dict[w_qkv_key] = np.concatenate((wq, wk, wv), axis=0)

    def _qkv_concat(self):
        """_qkv_concat"""
        num_layers = self.num_layers
        enable_tqdm = self.rank_id == 0
        for layer_id in tqdm(range(num_layers), desc="Concat QKV weights", disable=not enable_tqdm):
            self._qkv_concat_of_each_layer(layer_id)

    def _ffn_concat_of_each_layer(self, layer_id):
        """_ffn_concat_of_each_layer"""
        w_gate_hidden_key = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.weight"
        w1 = self._np_dict.pop(f"model.layers.{layer_id}.feed_forward.w1._layer.weight")
        w3 = self._np_dict.pop(f"model.layers.{layer_id}.feed_forward.w3._layer.weight")
        self._np_dict[w_gate_hidden_key] = np.concatenate((w1, w3), axis=0)

        new_key = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.matmul.dequant_scale"
        w1 = self._np_dict.pop(f"model.layers.{layer_id}.feed_forward.w1._layer.matmul.dequant_scale")
        w3 = self._np_dict.pop(f"model.layers.{layer_id}.feed_forward.w3._layer.matmul.dequant_scale")
        self._np_dict[new_key] = np.concatenate((w1, w3), axis=0)

        new_key = f"model.layers.{layer_id}.feed_forward.w_gate_hidden._layer.weight"
        w1 = self._np_dict.pop(f"model.layers.{layer_id}.feed_forward.w1._layer.matmul.quant_bias")
        w3 = self._np_dict.pop(f"model.layers.{layer_id}.feed_forward.w3._layer.matmul.quant_bias")
        self._np_dict[new_key] = np.concatenate((w1, w3), axis=0)

    def _ffn_concat(self):
        """_ffn_concat"""
        if not self.config.model.model_config.qkv_concat:
            return
        num_layers = self.config.model.model_config.num_layers
        enable_tqdm = self.rank_id == 0
        for layer_id in tqdm(range(num_layers), desc="Concat FFN weights", disable=not enable_tqdm):
            self._ffn_concat_of_each_layer(layer_id)

    def _moe_merge_of_each_layer(self, layer_id):
        """_qkv_concat_of_each_layer"""
        fc1_weights = []
        weight_scales = []
        weight_offsets = []
        fc2_weights = []
        for i in range(self.num_experts):
            # fc1
            fc1 = self._np_dict.pop(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight")
            fc1 = fc1.transpose()
            fc1_weights.append(np.expand_dims(fc1, 0))
            fc1_w_scale = self._np_dict.pop(
                f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_scale")
            weight_scales.append(np.expand_dims(fc1_w_scale, 0))
            fc1_w_offset = self._np_dict.pop(
                f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc1.weight_offset")
            weight_offsets.append(np.expand_dims(fc1_w_offset, 0))
            # fc2
            fc2 = self._np_dict.pop(f"model.decoder.layers.{layer_id}.mlp.experts.{i}.linear_fc2.weight")
            fc2 = fc2.transpose()
            fc2_weights.append(np.expand_dims(fc2, 0))

        fc1_key = f"model.decoder.layers.{layer_id}.mlp.experts.linear_fc1"
        fc2_key = f"model.decoder.layers.{layer_id}.mlp.experts.linear_fc2"
        self._np_dict[f"{fc1_key}.weight"] = np.concatenate(tuple(fc1_weights), axis=0)
        self._np_dict[f"{fc1_key}.weight_scale"] = np.concatenate(tuple(weight_scales), axis=0)
        self._np_dict[f"{fc1_key}.weight_offset"] = np.concatenate(tuple(weight_offsets), axis=0)
        self._np_dict[f"{fc2_key}.weight"] = np.concatenate(tuple(fc2_weights), axis=0)

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
        is_fc1_quant = any([".linear_fc1.weight_scale" in k for k in experts_dict.keys()])
        is_fc2_quant = any([".linear_fc2.weight_scale" in k for k in experts_dict.keys()])
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

    def load_safetensors_shard(self, src_hf_dir, network):
        """qwen load safetensors and shard """
        self._src_dir = src_hf_dir

        index_json_path = os.path.join(src_hf_dir, 'model.safetensors.index.json')
        if not os.path.exists(index_json_path):
            raise RuntimeError(f"Not found index json file: 'model.safetensors.index.json'")
        with open(index_json_path, "r") as fp:
            self._param_map = json.load(fp)

        config_json_path = os.path.join(src_hf_dir, 'config.json')
        if not os.path.exists(index_json_path):
            raise RuntimeError(f"Not found config json file: 'config.json'")
        with open(config_json_path, "r") as fp:
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
        self._moe_merge()
        self._del_experts_weight(network)
        self._load_param(network)

        logger.info(f"These parameters in safetensors are not used: {self._param_map.keys() - self.handled_keys}")
