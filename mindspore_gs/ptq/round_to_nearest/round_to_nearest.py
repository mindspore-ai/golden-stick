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
import copy
import os
from typing import Tuple

from mindspore.nn import Cell
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.log as logger

from mindspore_gs import CompAlgo
from mindspore_gs.quantization.net_policy import NetPolicy
from mindspore_gs.ptq.quant_cells import PTQCell
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.convert_utils import QuantCell
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQMode
from .rtn_net_policy import RTNNetPolicy


class RoundToNearest(CompAlgo):
    """
    Native implementation for post training quantization based on min/max statistic values.
    Args:
        config(PTQConfig): config for RoundToNearst, default is ``None``.
    Raises:
        TypeError: If `config` type is not PTQConfig when it's not ``None``.
    """

    def __init__(self, config=None):
        super(RoundToNearest, self).__init__()
        if config is not None:
            if not isinstance(config, PTQConfig):
                raise TypeError(f'Shall init RTN with PTQConfig, bug got {type(config)}')
            self._config = config
        else:
            self._config = PTQConfig()
        # convert PTQConfig to InnerConfig to add inner parameters
        self._config = InnerPTQConfig.inner_config(self._config)
        self._qat_policy = RoundToNearest._init_net_policy(self._config)
        self._custom_transforms = {}
        self._custom_layer_policy_map = {}
        self._is_deploy: bool = self._config.mode == PTQMode.DEPLOY
        if hasattr(config, 'custom_transforms'):
            self._custom_transforms = config.custom_transforms
        if hasattr(config, 'custom_policies'):
            self._custom_layer_policy_map = config.custom_policies

    @staticmethod
    def _init_net_policy(config):
        return RTNNetPolicy(config)

    def _create_config(self):
        """Create SimulatedQuantizationConfig."""
        self._config = PTQConfig()

    def apply(self, network: Cell) -> Cell:
        """Apply"""

        if not isinstance(self._qat_policy, NetPolicy):
            raise RuntimeError("Derived class should provide net policy")
        self._qat_policy.build()

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

        ApplyProcessor(self._qat_policy).process(network)
        if not self._is_deploy and self._config.weight_only:
            network = self.calibrate(network)
        network.update_parameters_name()
        return network

    def calibrate(self, network):
        """
        Start calibrating network and statistic quant parameters.
        """
        def _process(root: Cell):
            if root is None:
                return
            for _, cell in root.name_cells().items():
                if not isinstance(cell, PTQCell):
                    _process(cell)
                    continue
                cell.calibrate()

        _process(network)
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
                if not isinstance(cell, QuantCell):
                    return cell, True
                cell.update_ascend_quant()
                return cell, False

        FixProcessor().process(network)
        network.update_parameters_name()

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """Implement method convert of super class."""
        if not isinstance(net_opt, Cell):
            raise TypeError(
                f'The parameter `net_opt` must be isinstance of Cell, but got {type(net_opt)}.')
        if not isinstance(ckpt_path, str):
            raise TypeError(
                f'The parameter `ckpt_path` must be isinstance of str, but got {type(ckpt_path)}.')
        if ckpt_path:
            logger.warning('ckpt_path in convert would be deprecated in next version')
        real_path = os.path.realpath(ckpt_path)
        if ckpt_path != "":
            if os.path.isfile(real_path):
                param_dict = load_checkpoint(ckpt_path)
                load_param_into_net(net_opt, param_dict)
            else:
                raise ValueError(
                    f'The parameter `ckpt_path` can only be empty or a valid file, but got {real_path}.')

        class ConvertProcessor(Processor):
            """A network iterator for converting network to deploy network."""
            def __init__(self, ptq_policy, is_deploy, backend):
                self._ptq_policy = ptq_policy
                self._is_deploy = is_deploy
                self._backend = backend

            def process_cell(self, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, Cell):
                    return cell, True
                if isinstance(cell, PTQCell):
                    cell.convert(self._backend, self._is_deploy)
                    return cell, True
                return cell, False

        ConvertProcessor(self._qat_policy, self._is_deploy, self._config.backend).process(net_opt)
        net_opt.update_parameters_name()
        return net_opt
