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
"""base class of mindformers quant model"""


import os
from mindspore.nn.utils import no_init_parameters
from mindspore.communication import get_rank
from mindspore import load_param_into_net, load_checkpoint
from mindformers import AutoModel, MindFormerConfig, build_context, build_parallel_config
from mindspore_gs.ptq.models.base_model import BaseModel
from mindspore_gs.common import logger


class MFModel(BaseModel):
    """MFModel"""
    _model_registry: dict[str, type] = {}

    @staticmethod
    def _reg_model(name, model_clazz):
        cur = MFModel._model_registry.get(name)
        if cur:
            raise RuntimeError(f"Duplicated model reg, name: {name}, already reg class: {cur}, "
                               f"current reg class:{model_clazz}")
        logger.info(f"Register name {name} to model {model_clazz}")
        MFModel._model_registry[name] = model_clazz

    @staticmethod
    def reg_model(alias=None):
        def decorator(cls):
            """decorator"""
            register_key = alias if alias is not None else cls.__name__
            MFModel._reg_model(register_key, cls)
            return cls

        return decorator

    def __init__(self, yaml_path):
        config = MindFormerConfig(yaml_path)
        build_context(config)
        build_parallel_config(config)
        with no_init_parameters():
            self.network = AutoModel.from_config(yaml_path)

        if config.load_checkpoint:
            self.network.load_weights(config.load_checkpoint)

    # pylint: disable=arguments-differ
    @classmethod
    def from_pretrained(cls, yaml_path):
        # todo check
        logger.info('Creating mindformers network...', flush=True)
        config = MindFormerConfig(yaml_path)
        if not hasattr(config, 'trainer') or not hasattr(config.trainer, 'model_name'):
            raise ValueError(f"Not contain trainer.model_name in yaml-file: {yaml_path}")
        model_name = config.trainer.model_name
        model_cls = MFModel._model_registry.get(model_name, None)
        if model_cls is None:
            raise ValueError(f"Not supported model_name: {model_name} from yaml: {yaml_path}")
        return model_cls(yaml_path)

    def forward(self, input_ids, max_new_tokens=1):
        return self.network.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)

    def parameters_dict(self, scope="") -> dict:
        param_dict = self.network.parameters_dict()
        param_dict = self._process_params_dict_before_save(param_dict)
        return param_dict

    def _network(self):
        return self.network

    def _transformer_layers(self) -> tuple[type]:
        """_transformer_layers"""
        from mindformers.parallel_core.inference.transformer.transformer_layer import TransformerLayer
        return [TransformerLayer]

    @staticmethod
    def _find_unique_file(directory, suffix):
        """_find_unique_file"""
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"directory not exist: {directory}")

        matching_files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.endswith(suffix):
                matching_files.append(file_path)

        if not matching_files:
            raise ValueError(f"not found any 'xxx.{suffix}' file under {directory}")
        if len(matching_files) > 1:
            error_msg = f"found multi 'xxx.{suffix}' file under {directory}: {matching_files}"
            raise ValueError(error_msg)

        return matching_files[0]

    def _load_tp_splited_safetensors(self, quant_safetensors_path):
        """_load_tp_splited_safetensors"""
        if not quant_safetensors_path:
            return
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        param_dict_path = os.path.join(quant_safetensors_path, f"rank_{rank_id}")
        param_dict_path = MFModel._find_unique_file(param_dict_path, ".safetensors")
        param_dict = load_checkpoint(param_dict_path, format="safetensors")
        new_param_dict = self._process_params_dict_before_load(param_dict)
        param_not_load, ckpt_not_load = load_param_into_net(self.network, new_param_dict)
        logger.info(f"Network has but not in ckpt: {param_not_load}", flush=True)
        logger.info(f"CKPT has but not in network: {ckpt_not_load}", flush=True)

    def _process_params_dict_before_save(self, param_dict) -> dict:
        raise NotImplementedError

    def _process_params_dict_before_load(self, param_dict) -> dict:
        return param_dict

    def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
        raise NotImplementedError
