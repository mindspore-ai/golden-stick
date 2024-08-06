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
"""Network helper for network from MindFormers."""
import os.path

from typing import Union, List

import math
import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor, Model
from mindformers import MindFormerConfig, build_context, AutoModel, build_parallel_config
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.build_tokenizer import build_tokenizer
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.models.llama import LlamaForCausalLM, LlamaModel
from mindformers.models.llama.llama_transformer import LLamaDecodeLayer, LLamaAttention
from mindformers.models.llama.llama_layer import LlamaFeedForward
from mindspore_gs.common.utils import value_check
from .network_helper import NetworkHelper


class MFNetworkHelper(NetworkHelper):
    """
    Network helper for network from MindFormers.

    Args:
        config (Union[str, MindFormerConfig]): MindFormerConfig or path of config file for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig` neither a str.
        ValueError: If input `config` is not a valid file path when input `config` is a str.
    """
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        value_check("config", config, (MindFormerConfig, str))
        if isinstance(config, MindFormerConfig):
            self.mf_config = config
        else:
            if not os.path.isfile(config):
                raise ValueError(f"Input `config`({config}) is not a valid file path.")
            self.mf_config = MindFormerConfig(config)
        build_parallel_config(self.mf_config)
        self.mf_config.model.model_config.parallel_config = self.mf_config.parallel_config

    def create_network(self):
        build_context(self.mf_config)
        network = AutoModel.from_config(self.mf_config, download_checkpoint=False)
        network.set_train(False)
        network.phase = 'predict'
        model = Model(network)
        ckpt_path = self.mf_config.load_checkpoint
        if ckpt_path:
            input_ids = np.ones(shape=[self.get_spec('batch_size'), self.get_spec('seq_length')], dtype=np.int32)
            infer_data = network.prepare_inputs_for_predict_layout(input_ids)
            if os.path.isdir(ckpt_path):
                network.phase = 'infer_predict_layout'
                model.infer_predict_layout(*infer_data)
            transform_and_load_checkpoint(self.mf_config, model, network, infer_data, do_predict=True)
        ms.ms_memory_recycle()
        network.phase = 'predict'
        return network

    def get_spec(self, name: str):
        value_check('name', name, str)
        if name == 'vocab_file':
            return self.mf_config.processor.tokenizer.vocab_file
        model_config = self.mf_config.model.model_config
        if hasattr(model_config, name):
            return getattr(model_config, name)
        raise KeyError(f"Can not find network specific: {name}.")

    # pylint: disable=arguments-differ
    def create_tokenizer(self):
        """create_tokenizer."""
        return build_tokenizer(self.mf_config.processor.tokenizer)

    # pylint: disable=arguments-differ
    def generate(self, mf_network: PreTrainedModel, input_ids: Union[np.ndarray, List[int], List[List[int]]],
                 max_new_tokens=None):
        value_check('mf_network', mf_network, PreTrainedModel)
        value_check('input_ids', input_ids, (np.ndarray, List))
        if max_new_tokens:
            value_check('max_new_tokens', max_new_tokens, int)
        do_sample = self.get_spec('do_sample')
        seq = self.get_spec('seq_length')
        top_p = self.get_spec('top_p')
        top_k = self.get_spec('top_k')
        return mf_network.generate(input_ids, do_sample=do_sample, max_length=seq, max_new_tokens=max_new_tokens,
                                   top_p=top_p, top_k=top_k)

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        raise NotImplementedError


class MFLlama2Helper(MFNetworkHelper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework Llama2 network.

    Args:
        config (MindFormerConfig): MindFormerConfig for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.
    """
    @staticmethod
    def _get_slots(bs, block_size, prefill_max_len, is_prefill, block_tables, valid_length_example):
        """get_slots."""
        slot_mapping = []
        for i in range(bs):
            block_table = block_tables[i]
            if is_prefill:
                slots = [block_table[k // block_size] * block_size + k % block_size
                         for k in range(valid_length_example[i])]
                null_slot_idx = -1
                num_elements_to_add = prefill_max_len - valid_length_example[i]
                for _ in range(num_elements_to_add):
                    slots.append(null_slot_idx)
            else:
                current_idx = valid_length_example[i] - 1
                slots = [block_table[current_idx // block_size] * block_size + current_idx % block_size]
            slot_mapping = slot_mapping + slots

        return np.array(slot_mapping, copy=False, dtype=np.int32)

    @staticmethod
    def _get_pa_inputs(bs, seq, block_size, valid_length):
        """_get_pa_inputs"""
        valid_length_each_example = np.array([valid_length])
        prefill_max_len = max(valid_length_each_example)
        required_block_num = math.ceil(seq / block_size)
        block_tables = np.arange(required_block_num, dtype=np.int32).reshape(bs, -1)
        slot_mapping = MFLlama2Helper._get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                                 valid_length_each_example)
        block_tables = Tensor(block_tables, mstype.int32)
        slot_mapping = Tensor(slot_mapping, mstype.int32)
        return block_tables, slot_mapping

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
        value_check('input_ids', input_ids, np.ndarray)
        shape = input_ids.shape
        if len(shape) > 2:
            raise ValueError(f"Only support two-dimension(bs, seq_length) input_ids, got: {shape}.")
        bs = self.mf_config.model.model_config.batch_size
        seq = self.mf_config.model.model_config.seq_length
        if shape[0] > bs or shape[1] > seq:
            raise ValueError(f"Input input_ids shape({shape}) out of max shape({bs}, {seq}).")
        pad_token_id = self.mf_config.model.model_config.pad_token_id
        use_past = self.mf_config.model.model_config.use_past
        input_ids = np.pad(input_ids, ((0, bs - shape[0]), (0, seq - shape[1])), 'constant',
                           constant_values=pad_token_id)
        t_input_ids = Tensor(input_ids)
        if not use_past:
            return t_input_ids, None, None, None, None, None, None, None, None, None, None, None
        block_size = self.mf_config.model.model_config.block_size

        block_tables, slot_mapping = MFLlama2Helper._get_pa_inputs(bs, seq, block_size, shape[1])
        return t_input_ids, None, None, None, None, None, None, None, None, None, block_tables, slot_mapping

    def get_decoder_layers(self, network: LlamaForCausalLM):
        value_check('network', network, LlamaForCausalLM)
        model: LlamaModel = network.model
        return model.layers

    # pylint: disable=protected-access
    def offload_embedding(self, network: LlamaForCausalLM):
        value_check('network', network, LlamaForCausalLM)
        model: LlamaModel = network.model
        model.casual_mask.lower_triangle_mask._offload()
        model.tok_embeddings.embedding_weight._offload()

    def get_linears(self, decoder_layer: LLamaDecodeLayer):
        value_check('decoder_layer', decoder_layer, LLamaDecodeLayer)
        attention: LLamaAttention = decoder_layer.attention
        if not isinstance(attention, LLamaAttention):
            raise RuntimeError(f"Only support LLamaAttention as attention but got {attention}")
        qkv_concat = attention.qkv_concat
        ffn: LlamaFeedForward = decoder_layer.feed_forward
        if not isinstance(ffn, LlamaFeedForward):
            raise RuntimeError(f"Only support LlamaFeedForward as FFN but got {ffn}")
        ffn_concat = ffn.ffn_concat
        linears = []
        if qkv_concat:
            linears.extend([attention.w_qkv, attention.wo])
        else:
            linears.extend([attention.wq, attention.wk, attention.wv, attention.wo])

        if ffn_concat:
            linears.extend([ffn.w_gate_hidden, ffn.w2])
        else:
            linears.extend([ffn.w1, ffn.w3, ffn.w2])
        return qkv_concat, ffn_concat, linears
