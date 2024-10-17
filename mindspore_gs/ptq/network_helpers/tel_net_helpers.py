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


import math
from collections import OrderedDict
import os
import time
from typing import Union, List
import numpy as np
import mindspore as ms
from mindspore import dtype as mstype
from mindspore import Tensor
from mindspore import Model
from mindformers.tools.register.config import MindFormerConfig
from mindformers.models.modeling_utils import PreTrainedModel
from research.telechat.telechat import TelechatForCausalLM
from research.telechat.telechat_tokenizer import TelechatTokenizer
from research.telechat.telechat_config import TelechatConfig
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers import MindFormerConfig, build_context, AutoModel, build_parallel_config
from research.telechat.telechat_tokenizer import TelechatTokenizer
from mindspore_gs.common.utils import value_check
from .network_helper import NetworkHelper


class TELNetworkHelper(NetworkHelper):
    """
    Network helper for network from MindFormers.

    Args:
        config (MindFormerConfig): MindFormerConfig for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.
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
        """
        Create network of type LlamaForCasualLM.

        Returns:
            Network of type LlamaForCasualLM.
        """
        build_context(self.mf_config)
        network = TelechatForCausalLM(self.mf_config.model.model_config)
        network.set_train(False)
        network.phase = 'predict'
        ckpt_path = self.mf_config.load_checkpoint
        if ckpt_path:
            self._load_ckpt(network)
        ms.ms_memory_recycle()
        return network
    
    def analysis_decoder_groups(self, network):
        raise NotImplementedError

    def get_pre_layer(self, linear_name):
        raise NotImplementedError
    
    def get_spec(self, name: str):
        if name == 'vocab_file':
            return self.mf_config.processor.tokenizer.vocab_file
        model_config = self.mf_config.model.model_config
        if hasattr(model_config, name):
            return getattr(model_config, name)
        raise KeyError(f"Can not find network specific: {name}.")

    def create_tokenizer(self, **kwargs):
        """create_tokenizer."""
        return TelechatTokenizer(vocab_file=self.get_spec('vocab_file'))

    # pylint: disable=arguments-differ
    def generate(self, mf_network, input_ids: Union[np.ndarray, List[int], List[List[int]]],
                 max_new_tokens=None, **kwargs):
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

    def _load_ckpt(self, network):
        raise NotImplementedError


class TELHelper(TELNetworkHelper):
    """
    Derived from 'NetworkHelper', a utility class for the MindFormers framework Llama2 network.

    Args:
        config (MindFormerConfig): MindFormerConfig for network.

    Raises:
        TypeError: If input `config` is not an instance of `MindFormerConfig`.
    """
    def __init__(self, config: Union[str, MindFormerConfig] = None):
        super().__init__(config)
        self._decoder_infos = OrderedDict()
    
    def _load_ckpt(self, network):
        """_load_ckpt"""
        start = time.time()
        model = Model(network)
        input_ids = np.ones(shape=[self.get_spec('batch_size'), self.get_spec('seq_length')], dtype=np.int32)
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        if self.mf_config.use_parallel:
            network.phase = 'infer_predict_layout'
            model.infer_predict_layout(*infer_data)
        transform_and_load_checkpoint(self.mf_config, model, network, infer_data, do_predict=True)
        logger.info(f'Load ckpt cost time is {time.time() - start} s.')


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
        slot_mapping = TELHelper._get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                                 valid_length_each_example)
        block_tables = Tensor(block_tables, mstype.int32)
        slot_mapping = Tensor(slot_mapping, mstype.int32)
        return block_tables, slot_mapping

    def assemble_inputs(self, input_ids: np.ndarray, **kwargs):
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

        block_tables, slot_mapping = TELHelper._get_pa_inputs(bs, seq, block_size, shape[1])
        return t_input_ids, None, None, None, None, None, None, None, None, None, block_tables, slot_mapping
    
    def analysis_decoder_groups(self, network):
        pass

    def get_pre_layer(self, linear_name: str):
        pass

