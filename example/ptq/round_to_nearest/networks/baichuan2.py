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
"""BaiChuanNetwork."""
import sys
import os
import time
import numpy as np
# pylint: disable=wrong-import-position
sys.path.append(os.path.abspath("/path/to/mindformers/research/baichuan2"))
from mindspore import log as logger
from mindspore_gs.ptq import PTQConfig, PTQMode
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import RoundToNearest as RTN
from mindformers import MindFormerConfig
from baichuan2_13b import Baichuan13BV2ForCausalLM
from baichuan2_tokenizer import Baichuan2Tokenizer
from .llama2 import Llama2Network


class BaiChuanNetwork(Llama2Network):
    """BaiChuanNetwork"""
    @staticmethod
    def create_network(mindformers_config):
        network = Baichuan13BV2ForCausalLM(mindformers_config.model.model_config)
        network.set_train(False)
        network.phase = 'predict'
        return network

    @staticmethod
    def create_tokenizer(vocab_file):
        return Baichuan2Tokenizer(vocab_file=vocab_file)

    @staticmethod
    def quant_network(network: Baichuan13BV2ForCausalLM, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, **kwargs):
        """Quant llama2 model to w8a16 with RTN algorithm."""
        if mode == PTQMode.QUANTIZE.value:
            logger.info("Use RTN algo to quant network and weight.")
        else:
            logger.info("Use RTN algo to quant network.")
        cfg = PTQConfig(mode=mode, backend=backend)
        ptq = RTN(config=cfg)
        start = time.time()
        qnet = ptq.apply(network.model)
        end = time.time()
        logger.info(f'fake quantize cost time is {end - start}')

        start = time.time()
        qnet = ptq.convert(qnet)
        end = time.time()
        logger.info(f'convert to real quantize cost time is {end - start}')
        network.model = qnet
        return network

    @staticmethod
    def assemble_inputs(input_ids: np.ndarray, config: MindFormerConfig):
        raise NotImplementedError
