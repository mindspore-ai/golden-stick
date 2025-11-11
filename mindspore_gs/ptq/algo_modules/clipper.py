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
"""anti-outliers algorithm."""
from typing import Tuple

from mindspore.nn import Cell
from mindspore_gs.common import logger
from mindspore_gs.ptq.basic_functions.processor import Processor
from mindspore_gs.ptq.context import InnerPTQConfig, OutliersSuppressionType
from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell, Checker
from .algo_module import AlgoModule


class LinearClipper(AlgoModule):
    """clip got lienar"""

    linear_map = {}
    fake_quant_linear_map = {}

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        """register layer map"""
        if not issubclass(quant_layer_type, QuantCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(QuantCell)}, "
                               f"but got {quant_layer_type}.")
        logger.info(f"Register quant_cell {layer_type} with {quant_layer_type} to LinearClipper, checker: {checker}")
        if not LinearClipper.linear_map.get(layer_type):
            LinearClipper.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            LinearClipper.linear_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = LinearClipper.linear_map.get(layer_type) if not self.is_fake_quant else \
            LinearClipper.fake_quant_linear_map.get(layer_type)
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

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not self.handler.is_fake_quant and not LinearClipper.linear_map.get(type(cell)):
                    return cell, False
                if self.handler.is_fake_quant and not LinearClipper.fake_quant_linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                is_inner_osp = layer_policy.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS \
                            and layer_policy.use_inner_osp
                is_satisfied = layer_policy.outliers_suppression == OutliersSuppressionType.AWQ or \
                    layer_policy.outliers_suppression == OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE or \
                        is_inner_osp
                if is_satisfied:
                    layer_policy.weight_clip = True
                if not layer_policy or not layer_policy.weight_clip:
                    return cell, False
                if (any(opname in cell_name for opname in layer_policy.opname_blacklist) or
                        any(opname in cell_name for opname in clip_skip_layer)):
                    logger.info(f"{cell_name} is in blacklist, keep not being clip.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = self.handler.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, QuantCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {QuantCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                return wrapper_cell, True

        clip_skip_layer = ["wq", "wk", "w_qkv", "routed_experts.ffn.w2"]
        Replacer(self).process(decoder_layer, decoder_layer_name)
