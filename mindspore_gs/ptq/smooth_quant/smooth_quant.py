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
import warnings

import os
from typing import Tuple
import copy
import numpy as np

from mindspore import dtype as msdtype
from mindspore.nn import Cell
from mindspore.dataset import Dataset
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.comp_algo import CompAlgo
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq import PTQMode
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQApproach, BackendTarget
from mindspore_gs.ptq.quant_cell import PTQCell
from mindspore_gs.ptq.smooth_quant.sq_cell import SQCell
from mindspore_gs.common import logger
from mindspore_gs.common.utils import value_check
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
        self._config = InnerPTQConfig.inner_config(self._config, approach=PTQApproach.SMOOTH_QUANT)
        self._config.act_dtype = msdtype.int8
        self._config.weight_dtype = msdtype.int8
        self._config.kvcache_dtype = msdtype.float_
        if self._config.backend != BackendTarget.ASCEND:
            raise ValueError("SmoothQuant only support ASCEND as BackendTarget now, "
                             f"but got {self._config.backend}.")
        self._ptq_policy = SmoothQuant._init_net_policy(self._config)
        mode = self._config.mode
        self._is_deploy = mode == PTQMode.DEPLOY

    @staticmethod
    def load_mindformers_plugin():
        """
        Load quant cells, layer policy for MindFormers as plugin so that `SmoothQuant` can support network from
        MindFormers. Invoking this static method before creating `SmoothQuant`.
        """
        # pylint: disable=unused-import
        import mindspore_gs.ptq.smooth_quant.quant_cells.mindformers

    @staticmethod
    def _init_net_policy(config):
        SmoothQuant.load_mindformers_plugin()
        return SQNetPolicy(config)

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, network_helper: NetworkHelper = None, datasets: Dataset = None) -> Cell:
        """
        Define how to add fake quantizer to `network`.

        Args:
            network (Cell): Network to be fake quantized.
            network_helper (NetworkHelper): Utils for decoupling algorithm with network framework.
            datasets (Dataset): Datasets for calibrating.

        Raises:
            RuntimeError: If SmoothQuant is not well inited.
            TypeError: If input `network` is not a Cell.
            TypeError: If input `network_helper` is not a NetworkHelper if mode is `PTQMode.QUANTIZE`.
            TypeError: If input `datasets` is not a GeneratorDataset if mode is `PTQMode.QUANTIZE`.

        Returns:
            fake quantized network.
        """
        value_check('network', network, Cell)
        if not self._is_deploy:
            value_check('network_helper', network_helper, NetworkHelper)
            value_check('datasets', datasets, Dataset)
        if not isinstance(self._ptq_policy, SQNetPolicy):
            raise RuntimeError("Derived class should provide net policy")

        class InsertFQCell(Processor):
            """A network iterator for applying algorithm on network."""

            def __init__(self, ptq_policy, config):
                self._ptq_policy = ptq_policy
                self._config = config
                self.changed = False

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
                    fq_cell = new_layer_policy.wrap_cell(cell)
                    if fq_cell:
                        self.changed = True
                        logger.info(f"replace {cell_name} with fake-quant cell {type(fq_cell)}.")
                        return fq_cell, True
                return cell, False

        class ToNextPhase(Processor):
            """A network iterator for applying algorithm on network."""
            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, SQCell):
                    return cell, False
                next_phase_cell = cell.to_next_phase()
                logger.info(f"replace {cell_name} with fake-quant cell {type(next_phase_cell)}.")
                return next_phase_cell, True

        self._ptq_policy.build()
        # act smooth observer
        replacer = InsertFQCell(self._ptq_policy, self._config)
        replacer.process(network)
        if not replacer.changed:
            warn_str = "No layer found in network is suitable for quantization, please check network and " \
                       "opname_blacklist."
            warnings.warn(warn_str, RuntimeWarning)
            return network
        network.update_parameters_name()
        if self._is_deploy:
            return network
        total_count = datasets.get_dataset_size()
        os.environ['NETWORK_PHASE'] = "actobs"
        network.phase = "prefill_actobs"
        data_count = 1
        tokenizer = network_helper.create_tokenizer()
        for _, ds_item in enumerate(datasets.create_dict_iterator()):
            logger.info(f"Calibrating, act smooth obs phase: dataset count: {data_count}/{total_count}")
            input_ids = ds_item['input_ids'].asnumpy()
            output = network_helper.generate(network, input_ids, max_new_tokens=1)
            data_count += 1
            if tokenizer is not None:
                logger.info(f"Input: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
                logger.info(f"Output: {tokenizer.decode(output, skip_special_tokens=True)}")
        # weight smooth observer; smooth; weight quant observer
        network_to_next_phase = ToNextPhase()
        network_to_next_phase.process(network)
        network.update_parameters_name()
        os.environ['NETWORK_PHASE'] = "weightobs"
        network.phase = "prefill_weightobs"
        bs = network_helper.get_spec('batch_size')
        network_helper.generate(network, np.ones([bs, 1], dtype=np.int32), max_new_tokens=1)
        # act quant observer
        network_to_next_phase.process(network)
        network.update_parameters_name()
        os.environ['NETWORK_PHASE'] = "quant"
        network.phase = "prefill_quant"
        data_count = 1
        for _, ds_item in enumerate(datasets.create_dict_iterator()):
            logger.info(f"Calibrating, act quant obs phase: dataset count: {data_count}/{total_count}")
            input_ids = ds_item['input_ids'].asnumpy()
            output = network_helper.generate(network, input_ids, max_new_tokens=1)
            data_count += 1
            if tokenizer is not None:
                logger.info(f"Input: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
                logger.info(f"Output: {tokenizer.decode(output, skip_special_tokens=True)}")
        return network

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a compressed network to a standard network before exporting.

        Args:
            net_opt (Cell): Network to be converted which is transformed by `RoundToNearest.apply`.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is ``""``, which means not loading
                checkpoint file to `net_opt`.

        Returns:
            An instance of Cell represents quantized network.

        Raises:
            TypeError: If `net_opt` is not Cell.
            TypeError: If `ckpt_path` is not string.
            ValueError: If `ckpt_path` is not empty and invalid.
        """
        value_check('net_opt', net_opt, Cell)
        value_check('ckpt_path', ckpt_path, str)
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
            for name, cell in root.name_cells().items():
                if isinstance(cell, PTQCell):
                    logger.info(f"convert {name} to real-quant cell.")
                    cell.convert()
                    nonlocal changed
                    changed = True
                else:
                    _convert(cell)

        changed = False
        _convert(net_opt)
        net_opt.update_parameters_name()
        if not changed and self._config.mode == PTQMode.QUANTIZE:
            warn_str = "No layer found in network is suitable for quantization, please check network and " \
                       "opname_blacklist, and make sure call apply before convert."
            warnings.warn(warn_str, RuntimeWarning)
        return net_opt

    def set_deploy(self, is_deploy: bool = False):
        self._config.algo_args['is_deploy'] = is_deploy
