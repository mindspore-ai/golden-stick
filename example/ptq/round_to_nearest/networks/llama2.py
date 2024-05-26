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
import mindspore as ms
from mindspore import log as logger
from mindformers import LlamaForCausalLM, LlamaTokenizer, MindFormerConfig, LlamaConfig, init_context, \
    TransformerOpParallelConfig
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import RoundToNearest as RTN
from .network import BaseNetwork

class Llama2Network(BaseNetwork):
    """Llama2Network."""
    @staticmethod
    def create_mfconfig(config_path, device, device_id, bs, seq_len, tokenizer_path="", ckpt_path="",
                        ckpt_strategy_file="", model_parallel=1):
        """Create mindformers config for llama2 network for example."""
        config = MindFormerConfig(config_path)
        if model_parallel == -1:
            model_parallel = config.parallel_config.model_parallel
        use_parallel = model_parallel > 1
        if device_id != -1:
            config.context.device_id = device_id
        config.context.device_target = device
        config.model.model_config.batch_size = bs
        if seq_len != -1:
            config.model.model_config.seq_length = seq_len
        config.model.model_config.compute_dtype = ms.float16
        config.model.model_config.layernorm_compute_type = ms.float32
        config.model.model_config.softmax_compute_type = ms.float16
        config.model.model_config.rotary_dtype = ms.float16
        config.model.model_config.param_init_type = ms.float16
        config.processor.tokenizer.vocab_file = tokenizer_path
        config.load_checkpoint = ckpt_path
        config.model.model_config.checkpoint_name_or_path = ckpt_path
        config.src_strategy_path_or_dir = ckpt_strategy_file
        config.use_parallel = use_parallel
        config.parallel_config.model_parallel = model_parallel
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
    def quant_network(network: LlamaForCausalLM, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND):
        """Quant llama2 model to w8a16 with RTN algorithm."""
        start = time.time()
        if mode == PTQMode.QUANTIZE.value:
            logger.info("Use RTN algo to quant network and weight.")
        else:
            logger.info("Use RTN algo to quant network.")
        cfg = PTQConfig(mode=mode, backend=backend)
        ptq = RTN(config=cfg)
        logger.info(f'Create PTQ cost time is {time.time() - start} s.')
        start = time.time()
        qnet = ptq.apply(network.model)
        logger.info(f'Apply PTQ cost time is {time.time() - start} s.')
        start = time.time()
        fake_input_ids = np.ones((1, 2), dtype=np.int64)
        network.generate(fake_input_ids.tolist(), ma_length=2048)
        logger.info(f'Calibrate cost time is {time.time() - start} s.')
        start = time.time()
        qnet = ptq.convert(qnet)
        logger.info(f'Convert to real quantize cost time is {time.time() - start} s.')
        network.model = qnet
        return network

    @staticmethod
    def get_slots(bs, block_size, prefill_max_len, is_prefill, block_tables, valid_length_example):
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
    def gen_fake_inputs(bs, seq, block_size):
        input_seq_len = seq // 2 + 1
        valid_length_each_example = np.array([input_seq_len])
        prefill_max_len = max(valid_length_each_example)
        required_block_num = math.ceil(input_seq_len / block_size)
        block_tables = np.arange(required_block_num, dtype=np.int32).reshape(bs, -1)
        slot_mapping = Llama2Network.get_slots(bs, block_size, prefill_max_len, True, block_tables,
                                               valid_length_each_example)
        input_ids = np.ones(input_seq_len, dtype=np.int64).reshape(bs, -1)

        return [input_ids, None, None, None, None, None, None, None, None, None, block_tables, slot_mapping]
    