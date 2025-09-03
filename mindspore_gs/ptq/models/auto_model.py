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
"""auto model"""


from mindspore_gs.common import logger
from mindspore_gs.ptq.models.base_model import BaseQuantForCausalLM


class AutoQuantForCausalLM:
    """AutoModel"""
    @staticmethod
    def from_pretrained(pretained) -> BaseQuantForCausalLM:
        model_hubs = BaseQuantForCausalLM.get_model_hub_registry()
        for name, model_hub in model_hubs.items():
            try:
                model = model_hub.from_pretrained(pretained)
                logger.info(f"Create model from {name}")
                return model
            except ValueError:
                pass
