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
"""base class of quant models"""


import time
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common.utils import offload_network
from .base_model import BaseQuantForCausalLM
from .distributed_parameter import DistributedParameter


class BaseQuantForCausalLMImpl(BaseQuantForCausalLM):
    """BaseQuantForCausalLMImpl"""
    def forward(self, input_ids, max_new_tokens=1):
        """forward"""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, **kwargs):
        """from_pretrained"""
        raise NotImplementedError

    def save_quantized(self, save_path):
        """save_pretrained"""
        raise NotImplementedError

    @staticmethod
    def _get_num_str(index, length=5):
        if index < 0:
            raise RuntimeError(f"index should be bigger than 0, but got {index}.")
        for i in range(length):
            threshold = 10^(i + 1)
            if index < threshold:
                return f"{'0' * (length - 1)}{index}"
        raise RuntimeError(f"index should be smaller than {10^length}, but got {index}.")

    def get_description_file(self, network):
        """
        Obtain the description of quantization type for each parameter in each layer of the network.
        Such as W8A8 or W4A8_DYNAMIC
        """
        raise NotImplementedError

    def parameters_dict(self, scope="") -> dict[str, DistributedParameter]:
        """parameters_dict"""
        raise NotImplementedError

    def _network(self):
        """_network"""
        raise NotImplementedError

    def _transformer_layers(self) -> tuple[type]:
        """_transformer_layers"""
        raise NotImplementedError

    def calibrate(self, ptq_config, layers_policy, datasets):
        """calibrate"""
        logger.info("Use ptq algo to quant network and weight.")
        net = self._network()
        ptq = PTQ(config=ptq_config, layer_policies=layers_policy)
        # pylint: disable=protected-access
        ptq._config.experimental = True
        ptq._config.use_fake_quant = True
        transformer_layers = self._transformer_layers()
        _ = [ptq.decoder_layer_types.append(layer) for layer in transformer_layers]
        quant_start = time.time()
        logger.info('Quantize-ing network...')
        start_time = time.time()
        ptq.apply(net, datasets=datasets)
        offload_network(net)
        logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
        start_time = time.time()
        logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
        logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')

    def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
        raise NotImplementedError
