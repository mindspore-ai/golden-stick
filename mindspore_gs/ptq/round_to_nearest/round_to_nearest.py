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
import numpy as np

from mindspore.nn import Cell
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.log as logger

from mindspore_gs import CompAlgo
from mindspore_gs.quantization.net_policy import NetPolicy
from mindspore_gs.ptq.quant_cells import PTQCell
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.convert_utils import QuantCell
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQMode, PTQApproach, BackendTarget
from mindspore_gs.ptq.network_helpers import NetworkHelper
from .rtn_net_policy import RTNNetPolicy
from .quant_cells import LinearQuant


class RoundToNearest(CompAlgo):
    """
    Native implementation for post training quantization based on min/max statistic values.

    Args:
        config(:class:`mindspore_gs.ptq.PTQConfig`): config for RoundToNearst, default is ``None``.

    Raises:
        TypeError: If `config` type is not PTQConfig when it's not ``None``.
        ValueError: If backend in config is not `BackendTarget.ASCEND`.

    Examples:
        >>> import mindspore_gs
        >>> from mindspore_gs import ptq
        >>> from mindspore_gs.ptq import RoundToNearest as rtn
        >>> from mindspore_gs.ptq import PTQConfig
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/code/lenet.py
        >>> ptq = rtn()
        >>> network = LeNet5()
        >>> fake_quant_net = ptq.apply(net_work)
        >>> quant_net = ptq.convert(fake_quant_net)
    """

    def __init__(self, config=None):
        super(RoundToNearest, self).__init__()
        if config is not None:
            if not isinstance(config, PTQConfig):
                raise TypeError(f'Shall init RTN with PTQConfig, bug got {type(config)}')
            self._config = config
        else:
            self._create_config()
        # convert PTQConfig to InnerConfig to add inner parameters
        self._config = InnerPTQConfig.inner_config(self._config, approach=PTQApproach.RTN)
        if self._config.backend != BackendTarget.ASCEND:
            raise ValueError("RoundToNearest only support ASCEND as BackendTarget now, "
                             f"but got {self._config.backend}.")
        self._ptq_policy = RoundToNearest._init_net_policy(self._config)
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

    @staticmethod
    def _convert2list(name, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            value = [value]
        elif len(value) > 2:
            raise ValueError("input `{}` len should be less than 3".format(name))
        return value

    def _update_config_from_dict(self, config: dict):
        """Create RoundToNearestPTQ `config` from a dict"""

    def _calibrate(self, network):
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

        restore_device_target = context.get_context("device_target")
        context.set_context(device_target="CPU")
        _process(network)
        context.set_context(device_target=restore_device_target)
        return network

    @staticmethod
    def _fix_param_after_load_ckpt(network):
        """
        Fix quant param after loaded checkpoint for some quant parameter is store in attribute of primitive. QuantCell
        is an example who's quant parameter is an attribute.
        """
        class FixProcessor(Processor):
            """A network iterator for fix parameter after load ckpt."""
            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, QuantCell):
                    return cell, True
                return cell, False

        FixProcessor().process(network)
        network.update_parameters_name()

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, network_helper: NetworkHelper = None) -> Cell:
        """
        Define how to add fake quantizer to `network`.

        Args:
            network (Cell): Network to be fake quantized.
            network_helper (NetworkHelper): Utils for decoupling algorithm with network framework.

        Raises:
            RuntimeError: If RoundToNearest is not well inited.

        Returns:
            fake quantized network.
        """
        if not isinstance(self._ptq_policy, NetPolicy):
            raise RuntimeError("Derived class should provide net policy")
        self._ptq_policy.build()

        class ApplyProcessor(Processor):
            """A network iterator for applying algorithm on network."""

            def __init__(self, ptq_policy, config):
                self._ptq_policy = ptq_policy
                self._config = config

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, Cell):
                    return cell, True
                for exclude_name in self._config.opname_blacklist:
                    if exclude_name in cell_name:
                        logger.info(f"Setting {cell_name} being no-quant")
                        return cell, True
                layer_policy = self._ptq_policy.get_layer_policy(type(cell))
                if layer_policy:
                    new_layer_policy = copy.deepcopy(layer_policy)
                    return new_layer_policy.wrap_cell(cell), True
                return cell, False

        ApplyProcessor(self._ptq_policy, self._config).process(network)
        network.update_parameters_name()
        if self._is_deploy:
            return network
        if network_helper:
            bs = network_helper.get_spec("batch_size") if network_helper.get_spec("batch_size") else 1
            seq = network_helper.get_spec("seq_length") if network_helper.get_spec("seq_length") else 4096
            network(*(network.prepare_inputs_for_predict_layout(input_ids=np.ones([bs, seq], dtype=np.int32))))
        return network

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a compressed network to a standard network before exporting.

        Args:
            net_opt (Cell): Network to be converted which is transformed by `RoundToNearest.apply`.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is ``""``, which means not loading
                checkpoint file to `net_opt`.

        Returns:
            An instance of Cell represents converted network.

        Raises:
            TypeError: If `net_opt` is not Cell.
            TypeError: If `ckpt_path` is not string.
            ValueError: If `ckpt_path` is not empty and invalid.
        """
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

        def _convert(root: Cell):
            if root is None:
                return
            for _, cell in root.name_cells().items():
                if isinstance(cell, LinearQuant):
                    cell.convert(self._config.backend, self._is_deploy)
                else:
                    _convert(cell)

        _convert(net_opt)
        net_opt.update_parameters_name()
        return net_opt
