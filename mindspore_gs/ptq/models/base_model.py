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


from mindspore_gs.common import logger


class BaseModel:
    """BaseModel"""
    _model_hub_registry: dict[str, type] = {}

    @staticmethod
    def _reg_model_hub(name, model_clazz):
        cur = BaseModel._model_hub_registry.get(name)
        if cur:
            raise RuntimeError(f"Duplicated model-hub reg, name: {name}, already reg class: {cur}, "
                               f"current reg class:{model_clazz}")
        logger.info(f"Register name {name} to model {model_clazz}")
        BaseModel._model_hub_registry[name] = model_clazz

    @staticmethod
    def reg_model_hub(alias=None):
        def decorator(cls):
            """decorator"""
            register_key = alias if alias is not None else cls.__name__
            BaseModel._reg_model_hub(register_key, cls)
            return cls

        return decorator

    @staticmethod
    def get_model_hub_registry():
        return BaseModel._model_hub_registry

    @classmethod
    def from_pretrained(cls, **kwargs):
        """from_pretrained"""
        raise NotImplementedError

    def forward(self, input_ids, max_new_tokens=1):
        """forward"""
        raise NotImplementedError

    def calibrate(self, ptq_config, layers_policy, datasets):
        """calibrate"""
        raise NotImplementedError

    def save_quantized(self, save_path):
        """save_pretrained"""
        raise NotImplementedError

    def fake_quant(self, ptq_config, layers_policy, quant_safetensors_path: str = ""):
        raise NotImplementedError
