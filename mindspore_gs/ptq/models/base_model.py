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


import os
import time
import json
import mindspore as ms
from mindspore import Parameter
from mindspore.communication import get_rank
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common.utils import offload_network


class BaseModel:
    """BaseModel"""
    def forward(self, input_ids, max_new_tokens=1):
        """forward"""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, **kwargs):
        """from_pretrained"""
        raise NotImplementedError

    @staticmethod
    def _cal_size(param: Parameter):
        """_cal_size"""
        shape = param.shape
        dtype = param.dtype
        return shape, dtype

    def save_pretrained(self, save_path, safetensor_name, json_name) -> str:
        """save_pretrained"""
        save_safetensor_path = self._save_safetenors(save_path, safetensor_name)
        _ = self._save_desc_json(save_path, safetensor_name, json_name)
        return save_safetensor_path

    def _save_safetenors(self, save_path, safetensor_name) -> str:
        """_save_safetenors"""
        start = time.time()
        logger.info(f"Saving checkpoint...", flush=True)
        param_dict = self.parameters_dict()
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        save_safetensor_path = os.path.join(save_path,
                                            f"{safetensor_name}_quant_safetensors")
        save_path = os.path.join(save_safetensor_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        final_path = os.path.join(save_path, safetensor_name)
        ms.save_checkpoint(param_dict, final_path, format="safetensors")
        logger.info(f'Checkpoint saved to {final_path}', flush=True)
        logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
        return save_safetensor_path

    def _save_desc_json(self, save_path, safetensor_name, json_name) -> str:
        """_save_desc_json"""
        start = time.time()
        logger.info(f"Saving describle json file...", flush=True)
        desc_info = self.get_description_file(self._network())
        save_safetensor_path = os.path.join(save_path,
                                            f"{safetensor_name}_quant_safetensors")
        save_json_path = os.path.join(save_safetensor_path,
                                      f"quant_model_description_{json_name}.json")
        os.makedirs(save_path, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(desc_info, f, ensure_ascii=False, indent=4)
        logger.info(f'Describle json file saved to {save_json_path}', flush=True)
        logger.info(f'Save describle json cost time is {time.time() - start} s.')
        return save_json_path

    def get_description_file(self, network):
        """
        Obtain the description of quantization type for each parameter in each layer of the network.
        Such as W8A8 or W4A8_DYNAMIC
        """
        raise NotImplementedError

    def parameters_dict(self, scope="") -> dict:
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
