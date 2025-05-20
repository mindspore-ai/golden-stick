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
"""Algorithm base class."""
import re
import copy
from typing import Tuple

from mindspore import dtype as msdtype
from mindspore.nn import Cell

from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.processor import Processor, transform_network_inplace
from mindspore_gs.ptq.ptq_config import OutliersSuppressionType
from mindspore_gs.ptq.context import LayerQuantizeAlgo
from mindspore_gs.common import logger
from .wrapper_cell import WrapperCell


class Algorithm:
    """Algorithm"""
    def __init__(self, net_config=None, layer_policies=None):
        if not isinstance(net_config, InnerPTQConfig):
            raise TypeError(f'net_config should be InnerPTQConfig, bug got {type(net_config)}')
        self.net_config = net_config
        if layer_policies:
            self.layer_policies = layer_policies
        else:
            self.layer_policies = {}
        for config in self.layer_policies:
            if not config and not isinstance(config, InnerPTQConfig):
                raise TypeError(f'layer_policies should be InnerPTQConfig, bug got {type(config)}')

    @staticmethod
    def _get_fallback_config(fallback_algo, origin_config):
        """get fallback config"""
        new_config = copy.deepcopy(origin_config)
        if fallback_algo == LayerQuantizeAlgo.A16W8:
            new_config.weight_quant_dtype = msdtype.int8
            new_config.act_quant_dtype = None
            new_config.kvcache_quant_dtype = None
            new_config.outliers_suppression = OutliersSuppressionType.NONE
        else:
            raise ValueError("Only support fallback layer quantization algorithm to A16w8 Now.")
        return new_config

    def target_layer_type(self) -> tuple:
        raise NotImplementedError

    def get_layer_policy(self, layer_name) -> InnerPTQConfig:
        """get_layer_policy"""
        layer_policy = None
        found = ''
        for name, config in self.layer_policies.items():
            tmp = re.match(name, layer_name)
            if tmp:
                if found:
                    raise RuntimeError(f"layer_policy '{found}' and '{name}' conflict, matching same layer "
                                       f"'{layer_name}'.")
                found = name
                layer_policy = config
        if not found:
            logger.debug(f"{layer_name} does not find available layer_policy, use network config.")
            layer_policy = self.net_config
        logger.debug(f"{layer_name} layer policy: {layer_policy}.")
        return layer_policy

    def load_mindformers_plugin(self):
        """load_mindformers_plugin"""
        raise NotImplementedError

    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """replace"""
        raise NotImplementedError

    @classmethod
    def class_name(cls):
        """class name"""
        return cls.__name__

    def process(self, decoder_layer_name: str, decoder_layer):
        """process"""
        def transform_fn(cell_name, cell):
            logger.info(f"process {cell_name} in {self.class_name()}")
            cell.process()

        transform_network_inplace(decoder_layer, WrapperCell, transform_fn, decoder_layer_name)

    def deploy(self, decoder_layer_name, decoder_layer):
        """deploy"""
        class Deployer(Processor):
            """A network iterator for transform fq-network to quant-network."""
            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, WrapperCell):
                    return cell, False
                deploy_cell = cell.deploy()
                logger.info(f"convert {cell_name} to real-quant cell({type(deploy_cell)}).")
                return deploy_cell, True

        Deployer().process(decoder_layer, decoder_layer_name)
