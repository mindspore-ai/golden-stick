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

    def process(self, decoder_layer_name: str, decoder_layer, args_list, kwargs_list, network_helper: NetworkHelper):
        """process"""
        _, _, linears = network_helper.get_linears(decoder_layer)
        linear_type = type(linears[0])
        logger.info("Replacing Linear with Smooth linear.")
        smooth_linear_type = LinearSmoother._linear_map.get(linear_type)
        if not smooth_linear_type or not issubclass(smooth_linear_type, WrapperLinearCell):
            raise RuntimeError(f"Not support linear type: {linear_type}.")
        smooth_linear_creator = partial(smooth_linear_type, cfg=self._config, net_helper=network_helper)
        network_replace(decoder_layer, linear_type, smooth_linear_type, smooth_linear_creator,
                        self._config.opname_blacklist, decoder_layer_name)
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
        logger.info("Take back Linear from SmoothQuantLinearCell.")
        get_out = lambda cell_name, cell: cell.linear
        network_replace(decoder_layer, smooth_linear_type, None, get_out, [])
