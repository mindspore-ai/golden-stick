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
"""QWENNetwork."""
import sys
import os
import time
# pylint: disable=wrong-import-position
sys.path.append(os.path.abspath("/path/to/mindformers/research/qwen"))
import mindspore as ms
from mindspore import log as logger
from mindformers import MindFormerConfig, init_context, TransformerOpParallelConfig
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import RoundToNearest as RTN
from qwen_model import QwenForCausalLM
from qwen_config import QwenConfig
from qwen_tokenizer import QwenTokenizer
from .network import BaseNetwork


class QWENNetwork(BaseNetwork):
    """QWENNetwork."""
    @staticmethod
    def create_mfconfig(config_path, device, device_id, bs, seq_len, tokenizer_path="", ckpt_path="",
                        ckpt_strategy_file="", model_parallel=1):
        """Create mindformers config for llama2 network for example."""
        if model_parallel > 1:
            # MS parallel not support bfloat16 now.
            compute_dtype = ms.float16
            use_parallel = True
        else:
            compute_dtype = ms.float16
            use_parallel = False
            model_parallel = 1
        config = MindFormerConfig(config_path)
        if device_id != -1:
            config.context.device_id = device_id
        config.context.device_target = device
        config.model.model_config.batch_size = bs
        config.model.model_config.seq_length = seq_len
        config.model.model_config.compute_dtype = compute_dtype
        config.model.model_config.layernorm_compute_type = ms.float32
        config.model.model_config.softmax_compute_type = ms.float16
        config.model.model_config.rotary_dtype = ms.float16
        config.model.model_config.param_init_type = ms.float16
        config.processor.tokenizer.vocab_file = tokenizer_path
        config.load_checkpoint = ckpt_path
        config.model.model_config.checkpoint_name_or_path = ckpt_path
        config.use_parallel = use_parallel
        config.parallel_config.model_parallel = model_parallel
        config.model.model_config = QwenConfig(**config.model.model_config)

        init_context(use_parallel=config.use_parallel, context_config=config.context, parallel_config=config.parallel)

        parallel_config = TransformerOpParallelConfig(**config.parallel_config)
        config.model.model_config.parallel_config = parallel_config
        return config

    @staticmethod
    def create_network(mindformers_config):
        network = QwenForCausalLM(mindformers_config.model.model_config)
        network.set_train(False)
        network.phase = 'predict'
        return network

    @staticmethod
    def create_tokenizer(vocab_file):
        return QwenTokenizer(vocab_file=vocab_file)

    @staticmethod
    def quant_network(network: QwenForCausalLM, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND):
        """Quant llama2 model to w8a16 with RTN algorithm."""
        if mode == PTQMode.QUANTIZE.value:
            logger.info("Use RTN algo to quant network and weight.")
        else:
            logger.info("Use RTN algo to quant network.")
        cfg = PTQConfig(mode=mode, backend=backend)
        ptq = RTN(config=cfg)
        start = time.time()
        qnet = ptq.apply(network.transformer)
        end = time.time()
        logger.info(f'fake quantize cost time is {end - start}')

        start = time.time()
        qnet = ptq.convert(qnet)
        end = time.time()
        logger.info(f'convert to real quantize cost time is {end - start}')
        network.transformer = qnet
        return network
