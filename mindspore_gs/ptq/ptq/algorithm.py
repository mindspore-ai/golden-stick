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
import warnings
import copy
from typing import Tuple

from mindspore import dtype as msdtype
from mindspore.nn import Cell
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.ptq_config import LayerQuantizeAlgo, OutliersSuppressionType
from mindspore_gs.common import logger
from .wrapper_cell import WrapperCell


class Algorithm:
    """Algorithm"""
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

    def replace(self, decoder_layer_name: str, decoder_layer, network_helper: NetworkHelper = None):
        """replace"""
        raise NotImplementedError

    def process(self, decoder_layer_name: str, decoder_layer, network_helper: NetworkHelper = None):
        """process"""
        raise NotImplementedError

    def deploy(self, decoder_layer_name, decoder_layer):
        """deploy"""
        class Deployer(Processor):
            """A network iterator for transform fq-network to quant-network."""
            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, WrapperCell):
                    return cell, False
                deploy_cell = cell.deploy()
                logger.info(f"convert {cell_name} to real-quant cell({type(deploy_cell)}).")
                nonlocal changed
                changed = True
                return deploy_cell, True

        changed = False
        Deployer().process(decoder_layer, decoder_layer_name)
        if not changed:
            warn_str = "No layer found in network is suitable for quantization, please check network and " \
                       "opname_blacklist, and make sure call apply before convert."
            warnings.warn(warn_str, RuntimeWarning)
