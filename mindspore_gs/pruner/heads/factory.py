# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
define the head pruner factory class
"""

from mindspore_gs.common.validator import Validator
from .lrp.bert.head_pruner_bert import HeadPrunerBertLRP
from .lrp.gpt.head_pruner_gpt import HeadPrunerGPTLRP
from .type import PruningType
from .supported import SupportedModels


class HeadPruningFactory:
    """
    Factory design pattern for object creation
    """

    @staticmethod
    def get_pruner(config):
        """
        Creates and return requested pruner by type.

        Args:
            config: configuration parameters as a dictionary. contain prune type, arch type and model config.
        """
        Validator.check_value_type("config", config, [dict])
        if config['prune_type'] == PruningType.LRP:
            if config['arch_type'] == SupportedModels.BERT:
                return HeadPrunerBertLRP(config['config'])

            if config['arch_type'] == SupportedModels.GPT:
                return HeadPrunerGPTLRP(config['config'])

            raise NotImplementedError("Provided architecture - {0} doesn't exist!!".format(config['arch_type']))

        raise NotImplementedError("Provided head pruning method - {0} doesn't exist!!".format(config['prune_type']))
