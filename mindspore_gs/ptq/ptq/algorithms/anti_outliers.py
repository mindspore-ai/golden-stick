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
import warnings

from typing import Tuple

from mindspore.nn import Cell
from mindspore_gs.common import logger
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.algorithm import Algorithm
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperLinearCell


class LinearSmoother(Algorithm):
    """smoother for linear"""

    _linear_map = {}

    def __init__(self, config=None):
        super().__init__()
        if not isinstance(config, InnerPTQConfig):
            raise TypeError(f'Shall init LinearSmoother with InnerPTQConfig, bug got {type(config)}')
        self._config = config

    @staticmethod
    def reg_linear_map(linear_type, smooth_linear_type):
        if not issubclass(smooth_linear_type, WrapperLinearCell):
            raise RuntimeError(f"Smooth linear type should be a subclass of {id(WrapperLinearCell)}, "
                               f"but got {smooth_linear_type}.")
        LinearSmoother._linear_map[linear_type] = smooth_linear_type

    @staticmethod
    def load_mindformers_plugin():
        # pylint: disable=unused-import
        import mindspore_gs.ptq.ptq.wrappers.mindformers

    def _replace(self, decoder_layer_name: str, decoder_layer, linear_type, smooth_linear_type, network_helper):
        """_replace"""
        class Replacer(Processor):
            """A network iterator for transform fq-network to quant-network."""
            def __init__(self, config):
                if not smooth_linear_type or not issubclass(smooth_linear_type, WrapperLinearCell):
                    raise RuntimeError(f"Not support linear type: {linear_type} with wrapper type "
                                       f"{smooth_linear_type}.")
                self._inner_config = config

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, linear_type):
                    return cell, False
                if isinstance(cell, smooth_linear_type):
                    return cell, True
                smooth_linear = smooth_linear_type(cell_name, cell, self._inner_config, network_helper)
                logger.info(f"replacing {cell_name} with smooth cell({id(smooth_linear)}).")
                nonlocal changed
                changed = True
                return smooth_linear, True

        changed = False
        Replacer(self._config).process(decoder_layer, decoder_layer_name)
        if not changed:
            warn_str = f"No layer found in network is suitable to smooth, please check network and opname_blacklist" \
                       f"({self._config.opname_blacklist})."
            warnings.warn(warn_str, RuntimeWarning)

    def process(self, decoder_layer_name: str, decoder_layer, args_list, kwargs_list, network_helper: NetworkHelper):
        """process"""
        _, _, linears = network_helper.get_linears(decoder_layer)
        linear_type = type(linears[0])
        logger.info("Replacing Linear with Smooth linear.")
        smooth_linear_type = LinearSmoother._linear_map.get(linear_type)
        self._replace(decoder_layer_name, decoder_layer, linear_type, smooth_linear_type, network_helper)
        _, _, linears = network_helper.get_linears(decoder_layer)

        logger.info("Catching inputs of all Linear in current decoder layer.")
        for j in range(len(args_list)):
            cur_args = args_list[j]
            cur_kwargs = kwargs_list[j]
            decoder_layer(*cur_args, **cur_kwargs)

        for linear in linears:
            if isinstance(linear, smooth_linear_type):
                logger.info(f"Smooth Linear {linear.linear_name}")
                linear.process()
