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
"""BaseNetwork."""


from mindformers import BaseModel
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget


class BaseNetwork:
    """BaseNetwork."""
    @staticmethod
    def create_mfconfig(config_path, device="", device_id=-1, bs=-1, seq_len=-1, tokenizer_path="", ckpt_path="",
                        ckpt_strategy_file="", model_parallel=1):
        """create_mfconfig."""
        raise NotImplementedError

    @staticmethod
    def create_network(mindformers_config):
        """create_network."""
        raise NotImplementedError

    @staticmethod
    def create_tokenizer(vocab_file):
        """create_tokenizer."""
        raise NotImplementedError

    @staticmethod
    def quant_network(network: BaseModel, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND):
        """quant_network."""
        raise NotImplementedError

    @staticmethod
    def gen_fake_inputs(bs, seq, block_size):
        """gen_fake_inputs."""
        return None


class NetworkRegister:
    """NetworkRegister."""
    _instance = None

    @staticmethod
    def instance():
        """instance."""
        if NetworkRegister._instance is None:
            NetworkRegister._instance = NetworkRegister()
        return NetworkRegister._instance

    def __init__(self):
        self._map = {}

    def reg(self, type_: str, clazz):
        """reg."""
        self._map[type_] = clazz

    def get(self, type_: str):
        """get."""
        base_network = self._map.get(type_)
        if base_network is None:
            raise RuntimeError(f"Unsupported network: {type_}, available: llama2_7b, llama2_13b, llama2_70b, "
                               "baichuan2_13b, qwen_14b.")
        return base_network
