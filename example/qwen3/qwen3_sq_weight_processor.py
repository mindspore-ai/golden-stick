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

"""
transform huggingface model to mindspore safetensor.
"""

import os
import json
import gc
import numpy as np
from tqdm import tqdm
import mindspore as ms

from weight_processor import BaseWeightProcessor


class Qwen3SQWeightProcessor(BaseWeightProcessor):
    r"""
    Provide Qwen3 Model weight load and shards.
    Args:
        config (Qwen3Config): The config of Qwen3 model.
        network (InferenceQwen3ForCausalLM): The network of Qwen3.

    """
    def __init__(self, config, network, is_quant):
        super().__init__(config, network, is_quant)
        self._np_dict = {}
        self._src_dir = None
        self._param_map = None

    def _get_split_set(self, name, split_axis=-1):
        """_get_split_set"""
        is_split_param = split_axis != -1
        np_data, _ = self.get_safetensor_from_file(name, self._src_dir, self._param_map, is_split_param, split_axis)
        self._np_dict[name] = np_data

    def _split_outer_weight(self):
        """_split_outer_weight"""
        vb_split_axis = -1 if self.config.parallel_config.vocab_emb_dp else 0
        self._get_split_set("model.tok_embeddings.embedding_weight", vb_split_axis)
        self._get_split_set("model.norm_out.weight")
        if not self.config.model.model_config.tie_word_embeddings:
            self._get_split_set("lm_head.weight", vb_split_axis)

    def _split_dense_ffn_weight(self, layer_id):
        """_split_dense_ffn_weight"""
        # w1
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w1._layer.weight", 0)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w1._layer.matmul.dequant_scale", 0)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w1._layer.matmul.quant_bias", 0)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w1.quant_op.input_scale", -1)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w1.quant_op.input_zp", -1)
        # w3
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w3._layer.weight", 0)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w3._layer.matmul.dequant_scale", 0)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w3._layer.matmul.quant_bias", 0)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w3.quant_op.input_scale", -1)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w3.quant_op.input_zp", -1)
        # w2
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w2._layer.weight", 1)
        self._get_split_set(f"model.layers.{layer_id}.feed_forward.w2._layer.matmul.weight_scale", -1)

    def _split_attention_weight(self, layer_id):
        """_split_attention_weight"""
        # wq
        self._get_split_set(f"model.layers.{layer_id}.attention.wq._layer.weight", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wq._layer.matmul.dequant_scale", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wq._layer.matmul.quant_bias", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wq.quant_op.input_scale", -1)
        self._get_split_set(f"model.layers.{layer_id}.attention.wq.quant_op.input_zp", -1)
        # wk
        self._get_split_set(f"model.layers.{layer_id}.attention.wk._layer.weight", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wk._layer.matmul.dequant_scale", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wk._layer.matmul.quant_bias", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wk.quant_op.input_scale", -1)
        self._get_split_set(f"model.layers.{layer_id}.attention.wk.quant_op.input_zp", -1)
        # wv
        self._get_split_set(f"model.layers.{layer_id}.attention.wv._layer.weight", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wv._layer.matmul.dequant_scale", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wv._layer.matmul.quant_bias", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wv.quant_op.input_scale", -1)
        self._get_split_set(f"model.layers.{layer_id}.attention.wv.quant_op.input_zp", -1)
        # wo
        self._get_split_set(f"model.layers.{layer_id}.attention.wo._layer.weight", 1)
        self._get_split_set(f"model.layers.{layer_id}.attention.wo._layer.matmul.dequant_scale", -1)
        self._get_split_set(f"model.layers.{layer_id}.attention.wo._layer.matmul.quant_bias", -1)
        self._get_split_set(f"model.layers.{layer_id}.attention.wo.quant_op.input_scale", 0)
        self._get_split_set(f"model.layers.{layer_id}.attention.wo.quant_op.input_zp", 0)

        qnorm_key = f"model.layers.{layer_id}.attention.q_norm.weight"
        self._get_split_set(qnorm_key, -1)
        knorm_key = f"model.layers.{layer_id}.attention.k_norm.weight"
        self._get_split_set(knorm_key, -1)

    def _split_norm_weight(self, layer_id):
        """_split_norm_weight"""
        # attention_norm
        attention_norm_ms_name = f"model.layers.{layer_id}.attention_norm.weight"
        self._get_split_set(attention_norm_ms_name, -1)

        # ffn_norm
        ffn_norm_ms_name = f"model.layers.{layer_id}.ffn_norm.weight"
        self._get_split_set(ffn_norm_ms_name, -1)

    def _split_weight_of_each_layer(self, layer_id):
        """_split_weight_of_each_layer"""
        self._split_attention_weight(layer_id)
        self._split_dense_ffn_weight(layer_id)
        self._split_norm_weight(layer_id)

    def _split_weight(self):
        """_split_weight"""
        self._split_outer_weight()
        num_layers = self.config.model.model_config.num_layers
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
        if not self.config.model.model_config.qkv_concat:
            return
        num_layers = self.config.model.model_config.num_layers
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

    def _load_param(self):
        """_load_param"""
        cast_map = {
            "model.tok_embeddings.embedding_weight": ms.bfloat16,
            "model.norm_out.weight": ms.bfloat16,
            "lm_head.weight": ms.bfloat16,
        }
        num_layers = self.config.model.model_config.num_layers
        for layer_id in range(num_layers):
            cast_map[f"model.layers.{layer_id}.feed_forward.w2._layer.weight"] = ms.bfloat16
            cast_map[f"model.layers.{layer_id}.attention.q_norm.weight"] = ms.bfloat16
            cast_map[f"model.layers.{layer_id}.attention.k_norm.weight"] = ms.bfloat16
            cast_map[f"model.layers.{layer_id}.attention_norm.weight"] = ms.bfloat16
            cast_map[f"model.layers.{layer_id}.ffn_norm.weight"] = ms.bfloat16

        enable_tqdm = self.rank_id == 0
        for key, value in tqdm(self._np_dict.items(), desc="Create params", disable=not enable_tqdm):
            param = ms.from_numpy(value)
            cast_dtype = cast_map.get(key)
            if cast_dtype:
                param.astype(cast_dtype)
            self.parameter_dict[key] = ms.Parameter(param, name=key, requires_grad=False)
        self._np_dict.clear()

        ms.load_param_into_net(self.network, self.parameter_dict)
        self.parameter_dict.clear()
        del self.parameter_dict
        gc.collect()

    def load_safetensors_shard(self, src_hf_dir):
        """qwen load safetensors and shard """
        param_json_path = os.path.join(src_hf_dir, 'param_name_map.json')
        if not os.path.exists(param_json_path):
            raise RuntimeError(f"Not found param map file: {param_json_path}")
        self._src_dir = src_hf_dir
        with open(param_json_path, "r") as fp:
            self._param_map = json.load(fp)

        self._split_weight()
        self._qkv_concat()
        self._ffn_concat()
        self._load_param()
