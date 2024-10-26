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
"""quantizer algorithm."""
from typing import Tuple

from mindspore.nn import Cell
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.algorithm import Algorithm
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell, Checker
from mindspore_gs.ptq.processor import Processor


class Quantizer(Algorithm):
    """quanter for linear and PageAttentionMgr"""

    layer_map = {}

    def __init__(self, config=None):
        super().__init__()
        if not isinstance(config, InnerPTQConfig):
            raise TypeError(f'Shall init Quantizer with InnerPTQConfig, bug got {type(config)}')
        self._config = config

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        if not issubclass(quant_layer_type, WrapperCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(WrapperCell)}, "
                               f"but got {quant_layer_type}.")
        if not Quantizer.layer_map.get(layer_type):
            Quantizer.layer_map[layer_type] = [(checker, quant_layer_type)]
        else:
            Quantizer.layer_map[layer_type].append((checker, quant_layer_type))

    @staticmethod
    def get_wrapper_layer(layer_type, config: InnerPTQConfig):
        wrappers = Quantizer.layer_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def load_mindformers_plugin(self):
        # pylint: disable=unused-import
        import mindspore_gs.ptq.ptq.wrappers.mindformers

    def replace(self, decoder_layer_name: str, decoder_layer, network_helper: NetworkHelper = None):

        class Replacer(Processor):
            """Replacer"""
            def __init__(self, inner_config):
                self._inner_config = inner_config

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                for opname in self._inner_config.opname_blacklist:
                    if opname in cell_name:
                        logger.info(f"{cell_name} is in blacklist, keep not being quant.")
                        return cell, True

                op_config = self._inner_config
                for fallback_name in self._inner_config.fallback_blacklist.keys():
                    if fallback_name in cell_name:
                        fallback_algo = self._inner_config.fallback_blacklist.get(fallback_name)
                        logger.info(f"fallback {cell_name} to {fallback_algo}.")
                        # pylint: disable=W0212
                        op_config = Quantizer._get_fallback_config(fallback_algo, op_config)
                        break

                wrapper_cell_type = Quantizer.get_wrapper_layer(type(cell), op_config)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, WrapperCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {WrapperCell}.")
                nonlocal changed
                wrapper_cell = wrapper_cell_type(cell_name, cell, cfg=self._inner_config, network_helper=network_helper)
                logger.info(f"Replacing {cell_name} with quant cell {wrapper_cell_type}.")
                changed = True
                return wrapper_cell, True

        changed = False
        Replacer(self._config).process(decoder_layer, decoder_layer_name)
        if not changed:
            warn_str = f"No layer found in network is suitable to quantize, please check network and opname_blacklist" \
                       f"({self._config.opname_blacklist})."
            logger.warning(warn_str)
