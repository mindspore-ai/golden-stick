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
from typing import Tuple
import copy
import numpy as np

from mindspore.nn import Cell
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq import PTQMode
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, BackendTarget
from mindspore_gs.ptq.smooth_quant.quant_cells import SQLinearActObserver, SQLinearWeightObserver, SQLinearWrapper
from mindspore_gs.common.register import cell_type_dicts
from mindspore_gs.common import logger
from mindspore_gs.ptq.smooth_quant.sq_net_policy import SQNetPolicy
from mindspore_gs.ptq.network_helpers import NetworkHelper


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
        if self._config.backend != BackendTarget.ASCEND:
            raise ValueError("SmoothQuant only support ASCEND as BackendTarget now, "
                             f"but got {self._config.backend}.")
        self._ptq_policy = SmoothQuant._init_net_policy(self._config)
        self._op_types = tuple({cell_type_dicts[item] for item in self._config.op_types})
        mode = self._config.mode
        self._is_deploy = mode == PTQMode.DEPLOY

    @staticmethod
    def _init_net_policy(config):
        return SQNetPolicy(config)

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, network_helper: NetworkHelper = None, ds=None, **kwargs) -> Cell:
        """Apply"""
        if not isinstance(self._ptq_policy, SQNetPolicy):
            raise RuntimeError("Derived class should provide net policy")

        class InsertActObserver(Processor):
            """A network iterator for applying algorithm on network."""

            def __init__(self, ptq_policy, config):
                self._ptq_policy = ptq_policy
                self._config = config

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, Cell):
                    return cell, True
                for exclude_name in self._config.opname_blacklist:
                    if exclude_name in cell_name:
                        logger.info(f"Setting {cell_name} being no-quant.")
                        return cell, True
                layer_policy = self._ptq_policy.get_layer_policy(type(cell))
                if layer_policy:
                    new_layer_policy = copy.deepcopy(layer_policy)
                    return new_layer_policy.wrap_cell(cell), True
                return cell, False

        def insert_weight_observer_and_quant(root: Cell):
            if root is None:
                return
            for name, cell in root.name_cells().items():
                if isinstance(cell, SQLinearActObserver):
                    root.insert_child_to_cell(name, SQLinearWeightObserver(cell))
                else:
                    insert_weight_observer_and_quant(cell)

        def insert_act_quant(root: Cell):
            if root is None:
                return
            for name, cell in root.name_cells().items():
                if isinstance(cell, SQLinearWeightObserver):
                    root.insert_child_to_cell(name, SQLinearWrapper(cell))
                else:
                    insert_act_quant(cell)

        self._ptq_policy.build()
        InsertActObserver(self._ptq_policy, self._config).process(network)
        network.update_parameters_name()
        if self._is_deploy:
            return network
        if ds:
            if not network_helper:
                raise ValueError("Please provide network_helper when datasets is given for calibrating.")
            total_count = ds.get_dataset_size()
            os.environ['NETWORK_PHASE'] = "actobs"
            network.phase = "prefill_actobs"
            data_count = 1
            for _, ds_item in enumerate(ds.create_dict_iterator()):
                logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
                input_ids = ds_item['input_ids'].asnumpy()
                network_helper.generate(network, input_ids, max_new_tokens=1)
                data_count += 1
        insert_weight_observer_and_quant(network)
        network.update_parameters_name()
        if ds:
            os.environ['NETWORK_PHASE'] = "weightobs"
            network.phase = "prefill_weightobs"
            bs = network_helper.get_spec('batch_size')
            network_helper.generate(network, np.ones([bs, 1], dtype=np.int32), max_new_tokens=1)
        insert_act_quant(network)
        network.update_parameters_name()
        if ds:
            os.environ['NETWORK_PHASE'] = "quant"
            network.phase = "prefill_quant"
            data_count = 1
            for _, ds_item in enumerate(ds.create_dict_iterator()):
                logger.info(f"Calibrating: dataset count: {data_count}/{total_count}")
                input_ids = ds_item['input_ids'].asnumpy()
                network_helper.generate(network, input_ids, max_new_tokens=1)
                data_count += 1
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
