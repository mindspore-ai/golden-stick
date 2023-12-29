# Copyright 2023 Huawei Technologies Co., Ltd
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
"""RoundToNearestPTQ."""
import os
from typing import Tuple

from mindspore.nn import Cell
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindformers.modules import Linear

from mindspore_gs.ptq.processor import Processor
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQApproach
from mindspore_gs.ptq.quant_cells import SQLinearWrapper
from mindspore_gs.common.register import cell_type_dicts
from mindspore_gs.ptq.convert_utils import QuantCell


class SmoothQuant(CompAlgo):
    """smooth quant for ptq"""

    def __init__(self, config=None):
        super().__init__(config)
        if config is not None:
            if not isinstance(config, PTQConfig):
                raise TypeError(f'Shall init SmoothQuant with PTQConfig, bug got {type(config)}')
            self._config = config
        else:
            self._config = PTQConfig()
        # convert PTQConfig to InnerConfig to add inner parameters
        self._config = InnerPTQConfig.inner_config(self._config)
        self._init_config_with_dict()
        self._op_types = {cell_type_dicts[item] for item in self._config.op_types}

    def apply(self, network: Cell) -> Cell:
        """Apply"""

        def _replace(root: Cell):
            if root is None:
                return
            for name, cell in root.name_cells().items():
                if type(cell) in self._op_types:
                    # todo: would add support for conv2d and mindspore.Linear
                    cell_wrapper = self._create_cell_wrapper(cell)
                    root.insert_child_to_cell(name, cell_wrapper)
                else:
                    _replace(cell)

        _replace(network)
        network.update_parameters_name()
        return network

    @staticmethod
    def fix_param_after_load_ckpt(network):
        """
        Fix quant param after loaded checkpoint for some quant parameter is store in attribute of primitive. QuantCell
        is an example who's quant parameter is an attribute.
        """

        class FixProcessor(Processor):
            """A network iterator for fix parameter after load ckpt."""

            def process_cell(self, cell: Cell) -> Tuple[Cell, bool]:
                if isinstance(cell, SQLinearWrapper):
                    return cell, False
                if isinstance(cell, QuantCell):
                    cell.update_ascend_quant()
                return cell, True

        FixProcessor().process(network)
        network.update_parameters_name()

    def convert(self, net_opt: Cell, ckpt_path="", backend=None) -> Cell:
        """convert"""

        if not isinstance(net_opt, Cell):
            raise TypeError(
                f'The parameter `net_opt` must be isinstance of Cell, but got {type(net_opt)}.')
        if not isinstance(ckpt_path, str):
            raise TypeError(
                f'The parameter `ckpt_path` must be isinstance of str, but got {type(ckpt_path)}.')
        real_path = os.path.realpath(ckpt_path)
        if ckpt_path != "":
            if os.path.isfile(real_path):
                param_dict = load_checkpoint(ckpt_path)
                load_param_into_net(net_opt, param_dict)
            else:
                raise ValueError(
                    f'The parameter `ckpt_path` can only be empty or a valid file, but got {real_path}.')

        def _convert(root: Cell):
            if root is None:
                return
            for _, cell in root.name_cells().items():
                if isinstance(cell, SQLinearWrapper):
                    cell.convert()
                else:
                    _convert(cell)

        _convert(net_opt)
        net_opt.update_parameters_name()
        return net_opt

    def _create_cell_wrapper(self, cell):
        if isinstance(cell, Linear):
            return SQLinearWrapper(cell, cfg=self._config)
        else:
            raise RuntimeError(f'{type(cell)} is not supported yet!')

    def set_deploy(self, is_deploy: bool = False):
        self._config.algo_args['is_deploy'] = is_deploy
