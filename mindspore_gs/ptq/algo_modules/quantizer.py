# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
from mindspore import dtype as msdtype
from mindspore_gs.common import logger
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell, Checker
from mindspore_gs.ptq.basic_functions.processor import Processor
from .algo_module import AlgoModule


class Quantizer(AlgoModule):
    """quanter for linear and PageAttentionMgr"""

    layer_map = {}
    fake_quant_layer_map = {}

    def target_layer_type(self) -> tuple:
        return tuple(self.layer_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        """register layer map"""
        if not issubclass(quant_layer_type, QuantCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(QuantCell)}, "
                               f"but got {quant_layer_type}.")
        logger.info(f"Register quant_cell {layer_type} with {quant_layer_type} to Quantizer, checker: {checker}")
        if not Quantizer.layer_map.get(layer_type):
            Quantizer.layer_map[layer_type] = [(checker, quant_layer_type)]
        else:
            Quantizer.layer_map[layer_type].append((checker, quant_layer_type))

    @staticmethod
    def reg_fake_quant_layer_map(layer_type, quant_layer_type, checker: Checker):
        """reg_fake_quant_layer_map"""
        if not Quantizer.fake_quant_layer_map.get(layer_type):
            Quantizer.fake_quant_layer_map[layer_type] = [(checker, quant_layer_type)]
        else:
            Quantizer.fake_quant_layer_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = Quantizer.fake_quant_layer_map.get(layer_type) if self.is_fake_quant else \
                   Quantizer.layer_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):

        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            @staticmethod
            def _is_quant(config):
                act_support_dtype = [msdtype.int8]
                weight_support_dtype = [msdtype.int8, msdtype.qint4x2]
                kvcache_support_dtype = [msdtype.int8]
                return (config.act_quant_dtype in act_support_dtype or
                        config.weight_quant_dtype in weight_support_dtype or
                        config.kvcache_quant_dtype in kvcache_support_dtype)

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not self.handler.is_fake_quant and not Quantizer.layer_map.get(type(cell)):
                    return cell, False
                if self.handler.is_fake_quant and not Quantizer.fake_quant_layer_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if not layer_policy or not self._is_quant(layer_policy):
                    return cell, False
                if any(opname in cell_name for opname in layer_policy.opname_blacklist):
                    logger.info(f"{cell_name} is in blacklist, keep not being quant.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = self.handler.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, QuantCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {QuantCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy)
                logger.info(f"Replacing {cell_name} with quant cell {wrapper_cell_type}.")
                return wrapper_cell, True

        Replacer(self).process(decoder_layer, decoder_layer_name)
