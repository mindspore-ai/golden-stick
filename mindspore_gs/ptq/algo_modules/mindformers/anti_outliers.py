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
from mindspore_gs.ptq.ptq.quant import PTQ
from mindspore_gs.ptq.basic_functions.processor import Processor
from mindspore_gs.ptq.context import InnerPTQConfig, OutliersSuppressionType
from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell, Checker, SearchInputs
from mindspore_gs.ptq.algo_modules.algo_module import AlgoModule


class LinearSmoother(AlgoModule):
    """LinearSmoother"""
    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        raise NotImplementedError


class LinearSmoothQuant(LinearSmoother):
    """smoother for linear"""

    linear_map = {}
    fake_quant_linear_map = {}

    @staticmethod
    def reg_self():
        """register self"""
        # Check if already registered to avoid duplicate registration
        if LinearSmoothQuant not in PTQ.pipeline:
            PTQ.pipeline.append(LinearSmoothQuant)
        logger.info(f"Register algo_module {LinearSmoothQuant} to {PTQ.__name__} pipeline.")
        # Add layer types that are not already in target_layer_type
        new_layer_types = tuple(set(LinearSmoothQuant.linear_map.keys()) - set(PTQ.target_layer_type))
        if new_layer_types:
            PTQ.target_layer_type += new_layer_types

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        """register layer map"""
        if not issubclass(quant_layer_type, QuantCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(QuantCell)}, "
                               f"but got {quant_layer_type}.")
        logger.info(
            f"Register quant_cell {layer_type} with {quant_layer_type} to LinearSmoothQuant, "
            f"checker: {checker}"
        )
        if not LinearSmoothQuant.linear_map.get(layer_type):
            LinearSmoothQuant.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            LinearSmoothQuant.linear_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = LinearSmoothQuant.linear_map.get(layer_type) if not self.is_fake_quant else \
            LinearSmoothQuant.fake_quant_linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """infer_and_cache"""
        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process cell"""
                if not self.handler.is_fake_quant and not LinearSmoothQuant.linear_map.get(type(cell)):
                    return cell, False
                if self.handler.is_fake_quant and not LinearSmoothQuant.fake_quant_linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if not layer_policy or layer_policy.outliers_suppression != OutliersSuppressionType.SMOOTH:
                    return cell, False
                if any(opname in cell_name for opname in layer_policy.opname_blacklist):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
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

        Replacer(self).process(decoder_layer, decoder_layer_name)


class LinearAWQ(LinearSmoother):
    """smoother for linear"""

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
        logger.info(f"Register quant_cell {layer_type} with {quant_layer_type} to LinearAWQ, checker: {checker}")
        if not LinearAWQ.linear_map.get(layer_type):
            LinearAWQ.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            LinearAWQ.linear_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = LinearAWQ.linear_map.get(layer_type) if not self.is_fake_quant else \
            LinearAWQ.fake_quant_linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, **kwargs):
        """infer_and_cache"""
        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process cell"""
                layer_policy = self.handler.get_layer_policy(cell_name)
                if not layer_policy or layer_policy.outliers_suppression != OutliersSuppressionType.AWQ:
                    return cell, False
                if any(opname in cell_name for opname in layer_policy.opname_blacklist):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, True
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

        Replacer(self).process(decoder_layer, decoder_layer_name)


class LinearAutoSmoother(LinearSmoother):
    """LinearAutoSmoother"""

    linear_map = {}
    fake_quant_linear_map = {}

    @staticmethod
    def reg_self():
        """register self"""
        # Check if already registered to avoid duplicate registration
        if LinearAutoSmoother not in PTQ.pipeline:
            PTQ.pipeline.append(LinearAutoSmoother)
        logger.info(f"Register algo_module {LinearAutoSmoother} to {PTQ.__name__} pipeline.")
        # Add layer types that are not already in target_layer_type
        new_layer_types = tuple(set(LinearAutoSmoother.linear_map.keys()) - set(PTQ.target_layer_type))
        if new_layer_types:
            PTQ.target_layer_type += new_layer_types

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        """register layer map"""
        if not issubclass(quant_layer_type, QuantCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(QuantCell)}, "
                               f"but got {quant_layer_type}.")
        logger.info(f"Register quant_cell {layer_type} with {quant_layer_type} " \
                    f"to LinearAutoSmoother, checker: {checker}")
        if not LinearAutoSmoother.linear_map.get(layer_type):
            LinearAutoSmoother.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            LinearAutoSmoother.linear_map[layer_type].append((checker, quant_layer_type))

    def get_wrapper_layer(self, layer_type, config: InnerPTQConfig):
        """get wrapper layer"""
        wrappers = LinearAutoSmoother.linear_map.get(layer_type) if not self.is_fake_quant else \
            LinearAutoSmoother.fake_quant_linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    # pylint: disable=arguments-differ
    def replace(self, decoder_layer_name: str, decoder_layer, search_inputs: SearchInputs = None, **kwargs):
        """infer_and_cache"""
        search_layer = search_inputs.layer if search_inputs else None
        search_args = search_inputs.layer_args if search_inputs else None
        search_kwargs = search_inputs.layer_kwargs if search_inputs else None

        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                """process cell"""
                if not LinearAutoSmoother.linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if (not layer_policy or
                        layer_policy.outliers_suppression not in (OutliersSuppressionType.AWQ,
                                                                  OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                                                                  OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS)):
                    return cell, False
                if any(opname in cell_name for opname in layer_policy.opname_blacklist):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = self.handler.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, QuantCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {QuantCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy,
                                                 decoder_layer=search_layer, layer_args=search_args,
                                                 layer_kwargs=search_kwargs)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                return wrapper_cell, True

        Replacer(self).process(decoder_layer, decoder_layer_name)
