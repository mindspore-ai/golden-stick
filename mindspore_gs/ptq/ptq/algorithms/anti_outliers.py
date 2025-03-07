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
from typing import Tuple

from mindspore.nn import Cell
from mindspore_gs.common import logger
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.context import InnerPTQConfig, OutliersSuppressionType
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.algorithm import Algorithm
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell, Checker, SearchInputs


class LinearSmoother(Algorithm):
    """LinearSmoother"""
    def load_mindformers_plugin(self):
        # pylint: disable=unused-import
        import mindspore_gs.ptq.ptq.wrappers.mindformers

    def replace(self, decoder_layer_name: str, decoder_layer, network_helper: NetworkHelper = None, **kwargs):
        raise NotImplementedError


class LinearSmoothQuant(LinearSmoother):
    """smoother for linear"""

    linear_map = {}

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        if not issubclass(quant_layer_type, WrapperCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(WrapperCell)}, "
                               f"but got {quant_layer_type}.")
        if not LinearSmoothQuant.linear_map.get(layer_type):
            LinearSmoothQuant.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            LinearSmoothQuant.linear_map[layer_type].append((checker, quant_layer_type))

    @staticmethod
    def get_wrapper_layer(layer_type, config: InnerPTQConfig):
        wrappers = LinearSmoothQuant.linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, network_helper: NetworkHelper = None, **kwargs):
        """infer_and_cache"""
        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not LinearSmoothQuant.linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if (not layer_policy or layer_policy.outliers_suppression == OutliersSuppressionType.NONE or
                        any(opname in cell_name for opname in layer_policy.opname_blacklist)):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = LinearSmoothQuant.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, WrapperCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {WrapperCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy,
                                                 network_helper=network_helper)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                nonlocal changed
                changed = True
                return wrapper_cell, True

        changed = False
        Replacer(self).process(decoder_layer, decoder_layer_name)
        if not changed:
            warn_str = f"No layer found in network is suitable to suppress, please check network and ptq-config."
            logger.warning(warn_str)


class LinearAWQ(LinearSmoother):
    """smoother for linear"""

    linear_map = {}

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        if not issubclass(quant_layer_type, WrapperCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(WrapperCell)}, "
                               f"but got {quant_layer_type}.")
        if not LinearAWQ.linear_map.get(layer_type):
            LinearAWQ.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            LinearAWQ.linear_map[layer_type].append((checker, quant_layer_type))

    @staticmethod
    def get_wrapper_layer(layer_type, config: InnerPTQConfig):
        wrappers = LinearAWQ.linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    def replace(self, decoder_layer_name: str, decoder_layer, network_helper: NetworkHelper = None, **kwargs):
        """infer_and_cache"""
        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                layer_policy = self.handler.get_layer_policy(cell_name)
                if (not layer_policy or layer_policy.outliers_suppression == OutliersSuppressionType.NONE or
                        any(opname in cell_name for opname in layer_policy.opname_blacklist)):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, True
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = LinearAWQ.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, WrapperCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {WrapperCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy,
                                                 network_helper=network_helper)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                nonlocal changed
                changed = True
                return wrapper_cell, True

        changed = False
        Replacer(self).process(decoder_layer, decoder_layer_name)
        if not changed:
            warn_str = f"No layer found in network is suitable to suppress, please check network and ptq-config."
            logger.warning(warn_str)


class LinearAutoSmoother(LinearSmoother):
    """LinearAutoSmoother"""

    linear_map = {}

    def target_layer_type(self) -> tuple:
        return tuple(self.linear_map.keys())

    @staticmethod
    def reg_layer_map(layer_type, quant_layer_type, checker: Checker):
        if not issubclass(quant_layer_type, WrapperCell):
            raise RuntimeError(f"Quantize linear type should be a subclass of {id(WrapperCell)}, "
                               f"but got {quant_layer_type}.")
        if not LinearAutoSmoother.linear_map.get(layer_type):
            LinearAutoSmoother.linear_map[layer_type] = [(checker, quant_layer_type)]
        else:
            LinearAutoSmoother.linear_map[layer_type].append((checker, quant_layer_type))

    @staticmethod
    def get_wrapper_layer(layer_type, config: InnerPTQConfig):
        wrappers = LinearAutoSmoother.linear_map.get(layer_type)
        if not wrappers:
            return None
        for checker_wrapper in wrappers:
            if not checker_wrapper[0].check(config):
                continue
            return checker_wrapper[1]
        return None

    # pylint: disable=arguments-differ
    def replace(self, decoder_layer_name: str, decoder_layer, network_helper: NetworkHelper = None,
                search_inputs: SearchInputs = None, **kwargs):
        """infer_and_cache"""
        search_layer = search_inputs.layer if search_inputs else None
        search_args = search_inputs.layer_args if search_inputs else None
        search_kwargs = search_inputs.layer_kwargs if search_inputs else None

        class Replacer(Processor):
            """Replacer"""
            def __init__(self, algorithm):
                self.handler = algorithm

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not LinearAutoSmoother.linear_map.get(type(cell)):
                    return cell, False
                layer_policy = self.handler.get_layer_policy(cell_name)
                if (not layer_policy or layer_policy.outliers_suppression == OutliersSuppressionType.NONE or
                        any(opname in cell_name for opname in layer_policy.opname_blacklist)):
                    logger.info(f"{cell_name} is in blacklist, keep not being suppressed.")
                    return cell, False
                logger.debug(f"{cell_name} layer policy: {layer_policy}.")
                wrapper_cell_type = LinearAutoSmoother.get_wrapper_layer(type(cell), layer_policy)
                if not wrapper_cell_type:
                    return cell, False
                if not issubclass(wrapper_cell_type, WrapperCell):
                    raise RuntimeError(f"Registered wrapper cell for {type(cell)} is {wrapper_cell_type} which is not "
                                       f"a subclass of {WrapperCell}.")
                wrapper_cell = wrapper_cell_type(cell_name, cell, context=self.handler.net_config, cfg=layer_policy,
                                                 network_helper=network_helper, decoder_layer=search_layer,
                                                 layer_args=search_args, layer_kwargs=search_kwargs)
                logger.info(f"Replacing {cell_name} with cell {wrapper_cell_type}.")
                nonlocal changed
                changed = True
                return wrapper_cell, True

        changed = False
        Replacer(self).process(decoder_layer, decoder_layer_name)
        if not changed:
            warn_str = f"No layer found in network is suitable to suppress, please check network and ptq-config."
            logger.warning(warn_str)
