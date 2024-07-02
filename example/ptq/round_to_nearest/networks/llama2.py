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
"""Llama2Network."""

import time
import math
import numpy as np
from mindspore import log as logger
from mindspore import Tensor
from mindspore import dtype as mstype
from mindformers import LlamaForCausalLM, LlamaTokenizer, MindFormerConfig, LlamaConfig, init_context, \
    TransformerOpParallelConfig
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
from .network import BaseNetwork


class Llama2Network(BaseNetwork):
    """Llama2Network."""
    @staticmethod
    def create_mfconfig(config_path):
        """Create mindformers config for llama2 network for example."""
        config = MindFormerConfig(config_path)
        config.model.model_config = LlamaConfig(**config.model.model_config)

        init_context(use_parallel=config.use_parallel, context_config=config.context, parallel_config=config.parallel)

        parallel_config = TransformerOpParallelConfig(**config.parallel_config)
        config.model.model_config.parallel_config = parallel_config
        return config

    @staticmethod
    def create_network(mindformers_config):
        network = LlamaForCausalLM(mindformers_config.model.model_config)
        network.set_train(False)
        network.phase = 'predict'
        return network

    @staticmethod
    def create_tokenizer(vocab_file):
        return LlamaTokenizer(vocab_file=vocab_file)

    @staticmethod
    def quant_network(network: LlamaForCausalLM, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, **kwargs):
        """Quant llama2 model to w8a16 with RTN algorithm."""
        start = time.time()
        if mode == PTQMode.QUANTIZE:
            logger.info("Use RTN algo to quant network and weight.")
        else:
            logger.info("Use RTN algo to quant network.")
        cfg = PTQConfig(mode=mode, backend=backend, opname_blacklist=["lm_head"])
        ptq = RTN(config=cfg)
        logger.info(f'Create PTQ cost time is {time.time() - start} s.')
        start = time.time()
        mfconfig = kwargs.get("mfconfig", None)
        if not mfconfig:
            raise ValueError("Please provide mfconfig for calibrating.")
        network_helper = MFLlama2Helper(mfconfig)
        network = ptq.apply(network, network_helper)
        logger.info(f'Apply PTQ cost time is {time.time() - start} s.')
        start = time.time()
        network.phase = "quant_convert"
        network = ptq.convert(network)
        logger.info(f'Convert to real quantize cost time is {time.time() - start} s.')
        return network

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
        slot_mapping = Llama2Network._get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                                valid_length_each_example)
        block_tables = Tensor(block_tables, mstype.int32)
        slot_mapping = Tensor(slot_mapping, mstype.int32)
        return block_tables, slot_mapping

    @staticmethod
    def assemble_inputs(input_ids: np.ndarray, config: MindFormerConfig):
        """quant_network."""
        shape = input_ids.shape
        if len(shape) > 2:
            raise ValueError(f"Only support two-dimension(bs, seq_length) input_ids, got: {shape}.")
        bs = config.model.model_config.batch_size
        seq = config.model.model_config.seq_length
        if shape[0] > bs or shape[1] > seq:
            raise ValueError(f"Input input_ids shape({shape}) out of max shape({bs}, {seq}).")
        pad_token_id = config.model.model_config.pad_token_id
        use_past = config.model.model_config.use_past
        input_ids = np.pad(input_ids, ((0, bs - shape[0]), (0, seq - shape[1])), 'constant',
                           constant_values=pad_token_id)
        t_input_ids = Tensor(input_ids)
        if not use_past:
            return t_input_ids, None, None, None, None, None, None, None, None, None, None, None
        block_size = config.model.model_config.block_size

        block_tables, slot_mapping = Llama2Network._get_pa_inputs(bs, seq, block_size, shape[1])
        return t_input_ids, None, None, None, None, None, None, None, None, None, block_tables, slot_mapping
