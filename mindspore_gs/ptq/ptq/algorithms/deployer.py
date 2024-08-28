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
"""anti-outliers algorithm."""

from functools import partial
from mindspore_gs.common import logger
from mindspore_gs.ptq.processor import network_replace
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.algorithm import Algorithm
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell


class Deployer(Algorithm):
    """quanter for linear and PageAttentionMgr"""

    _layer_map = {}

    def __init__(self, config=None):
        super().__init__()
        if not isinstance(config, InnerPTQConfig):
            raise TypeError(f'Shall init Deployer with InnerPTQConfig, bug got {type(config)}')
        self._config = config

    @staticmethod
    def reg_layer_map(layer_type, deploy_layer_type):
        if not issubclass(deploy_layer_type, WrapperCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(WrapperCell)}, "
                               f"but got {deploy_layer_type}.")
        Deployer._layer_map[layer_type] = deploy_layer_type

    @staticmethod
    def load_mindformers_plugin():
        # pylint: disable=unused-import
        import mindspore_gs.ptq.ptq.wrappers.mindformers

    def process(self, decoder_layer_name: str, decoder_layer, args_list, kwargs_list, network_helper: NetworkHelper):
        _, _, linears = network_helper.get_linears(decoder_layer)
        linear_type = [type(linears[k]) for k in range(len(linears))]
        logger.info("Replacing Linear with deploy linear.")
        deploy_linear_type = Deployer._layer_map.get(linear_type[0])
        if not issubclass(deploy_linear_type, WrapperCell):
            raise RuntimeError(f"Not support linear type: {linear_type[0]}.")
        deploy_linear_creator = partial(deploy_linear_type, cfg=self._config, network_helper=network_helper)
        network_replace(decoder_layer, tuple(linear_type), deploy_linear_type, deploy_linear_creator,
                        self._config.opname_blacklist, decoder_layer_name)
