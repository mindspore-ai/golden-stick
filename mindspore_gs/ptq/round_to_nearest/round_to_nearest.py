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
import warnings

import copy
import os
from typing import Tuple
import numpy as np

from mindspore.nn import Cell
from mindspore.dataset import Dataset
from mindspore import dtype as msdtype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore_gs.common import logger

from mindspore_gs import CompAlgo
from mindspore_gs.quantization.net_policy import NetPolicy
from mindspore_gs.ptq.quant_cell import PTQCell
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig, PTQMode, PTQApproach, BackendTarget
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.common.utils import value_check
from .rtn_net_policy import RTNNetPolicy


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

    @staticmethod
    def load_mindformers_plugin():
        """
        Load quant cells, layer policy for MindFormers as plugin so that `RoundToNearest` can support network from
        MindFormers. Invoking this static method before creating `RoundToNearest`.
        """
        # pylint: disable=unused-import
        import mindspore_gs.ptq.round_to_nearest.quant_cells.mindformers

    @staticmethod
    def _init_net_policy(config):
        RoundToNearest.load_mindformers_plugin()
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

    # pylint: disable=arguments-differ
    def apply(self, network: Cell, network_helper: NetworkHelper = None, datasets: Dataset = None) -> Cell:
        """
        Define how to add fake quantizer to `network`.

        Args:
            network (Cell): Network to be fake quantized.
            network_helper (NetworkHelper): Utils for decoupling algorithm with network framework.
            datasets (Dataset): Datasets for calibrating.

        Raises:
            RuntimeError: If RoundToNearest is not well inited.
            TypeError: If input `network` is not a Cell.
            TypeError: If input `network_helper` is not None and is not a NetworkHelper.
            ValueError: if `network_helper` is None when kvcache_quant_dtype is `mindspore.int8`.

        Returns:
            fake quantized network.
        """
        value_check('network', network, Cell)
        if network_helper:
            value_check('network_helper', network_helper, NetworkHelper)
        if not isinstance(self._ptq_policy, NetPolicy):
            raise RuntimeError("Derived class should provide net policy")
        self._ptq_policy.build()

        class ApplyProcessor(Processor):
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
                        logger.info(f"Setting {cell_name} being no-quant")
                        return cell, True
                layer_policy = self._ptq_policy.get_layer_policy(type(cell))
                if layer_policy:
                    new_layer_policy = copy.deepcopy(layer_policy)
                    fq_cell = new_layer_policy.wrap_cell(cell)
                    if fq_cell:
                        logger.info(f"replace {cell_name} with fake-quant cell {type(fq_cell)}.")
                        self.changed = True
                        return fq_cell, True
                return cell, False

        replacer = ApplyProcessor(self._ptq_policy, self._config)
        replacer.process(network)
        network.update_parameters_name()
        if not replacer.changed:
            warn_str = "No layer found in network is suitable for quantization, please check network and " \
                       "opname_blacklist."
            warnings.warn(warn_str, RuntimeWarning)
            return network
        if self._is_deploy:
            return network
        if network_helper and self._config.kvcache_quant_dtype != msdtype.int8:
            bs = network_helper.get_spec("batch_size") if network_helper.get_spec("batch_size") else 1
            network_helper.generate(network, input_ids=np.ones([bs, 1], dtype=np.int32))
        if datasets and self._config.kvcache_quant_dtype == msdtype.int8:
            if not network_helper:
                raise ValueError("Please provide network_helper when datasets is given for calibrating.")
            total_count = datasets.get_dataset_size()
            os.environ['NETWORK_PHASE'] = "kvcacheobs"
            network.phase = "prefill_kvcacheobs"
            data_count = 1
            for _, ds_item in enumerate(datasets.create_dict_iterator()):
                logger.info(f"Calibrating, kvcache obs phase: dataset count: {data_count}/{total_count}")
                input_ids = ds_item['input_ids'].asnumpy()
                output = network_helper.generate(network, input_ids,
                                                 max_new_tokens=self._config.kvcache_calibrate_max_new_tokens)
                data_count += 1
                tokenizer = network_helper.create_tokenizer()
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

        class Converter(Processor):
            """A network iterator for applying algorithm on network."""
            def __init__(self, backend, is_deploy):
                self._backend = backend
                self._is_deploy = is_deploy

            def process_cell(self, cell_name: str, cell: Cell) -> Tuple[Cell, bool]:
                if not isinstance(cell, PTQCell):
                    return cell, False
                logger.info(f"convert {cell_name} to real-quant cell.")
                cell.convert(self._backend, self._is_deploy)
                nonlocal changed
                changed = True
                return cell, True

        changed = False
        Converter(self._config.backend, self._is_deploy).process(net_opt)
        net_opt.update_parameters_name()
        if not changed and self._config.mode == PTQMode.QUANTIZE:
            warn_str = "No layer found in network is suitable for quantization, please check network and " \
                       "opname_blacklist, and make sure call apply before convert."
            warnings.warn(warn_str, RuntimeWarning)
        return net_opt
