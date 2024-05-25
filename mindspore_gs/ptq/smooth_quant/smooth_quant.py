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
"""SmoothQuant algorithm."""
import os
import copy
from typing import Tuple

from mindspore.nn import Cell
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindspore_gs.ptq.processor import Processor
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig
from mindspore_gs.ptq.quant_cells import SQLinearWrapper
from mindspore_gs.common.register import cell_type_dicts
from mindspore_gs.ptq.smooth_quant.sq_net_policy import SQNetPolicy


class SmoothQuant(CompAlgo):
    """smooth quant for ptq"""

    def __init__(self, config=None):
        super().__init__()
        if config is not None:
            if not isinstance(config, PTQConfig):
                raise TypeError(f'Shall init SmoothQuant with PTQConfig, bug got {type(config)}')
            self._config = config
        else:
            self._config = PTQConfig()
        # convert PTQConfig to InnerConfig to add inner parameters
        self._config = InnerPTQConfig.inner_config(self._config)
        self._ptq_policy = SmoothQuant._init_net_policy(self._config)
        self._op_types = tuple({cell_type_dicts[item] for item in self._config.op_types})

    @staticmethod
    def _init_net_policy(config):
        return SQNetPolicy(config)

    def apply(self, network: Cell) -> Cell:
        """Apply"""

        if not isinstance(self._ptq_policy, SQNetPolicy):
            raise RuntimeError("Derived class should provide net policy")
        self._ptq_policy.build()

        class ApplyProcessor(Processor):
            """A network iterator for applying algorithm on network."""

            def __init__(self, ptq_policy):
                self._ptq_policy = ptq_policy

            def process_cell(self, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, Cell):
                    return cell, True
                layer_policy = self._ptq_policy.get_layer_policy(type(cell))
                if layer_policy:
                    new_layer_policy = copy.deepcopy(layer_policy)
                    return new_layer_policy.wrap_cell(cell), True
                return cell, False

        ApplyProcessor(self._ptq_policy).process(network)
        network.update_parameters_name()
        return network

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
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

    def set_deploy(self, is_deploy: bool = False):
        self._config.algo_args['is_deploy'] = is_deploy
