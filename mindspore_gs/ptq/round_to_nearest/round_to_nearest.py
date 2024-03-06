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
from mindspore.common.dtype import QuantDtype
import mindspore.log as logger

from mindspore_gs import CompAlgo
from mindspore_gs.validator import Validator
from mindspore_gs.quantization.net_policy import NetPolicy
from mindspore_gs.ptq.quant_cells import PTQCell
from mindspore_gs.ptq.processor import Processor
from mindspore_gs.ptq.convert_utils import QuantCell
from mindspore_gs.ptq.ptq_config import PTQConfig, InnerPTQConfig
from mindspore_gs.common.gs_enum import PTQMode
from .rtn_net_policy import RTNNetPolicy


class RoundToNearest(CompAlgo):
    """MinMaxPTQ"""

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

    def set_act_per_channel(self, act_per_channel):
        """
        Set value of act_per_channel of RoundToNearestPTQ `config`

        Args:
            act_per_channel (bool): Quantization granularity based on layer or on channel. If ``True`` then base on
                per channel, otherwise base on per layer. Only support ``False`` now.

        Raises:
            TypeError: If `act_per_channel` is not bool.
            ValueError: Only supported if `act_per_channel` is ``False`` yet.
        """
        Validator.check_bool(act_per_channel, "act_per_channel", self.__class__.__name__)
        if act_per_channel:
            raise ValueError(f'Only supported if `act_per_channel` is False yet.')
        self._config.act_per_channel = act_per_channel

    def set_weight_per_channel(self, weight_per_channel):
        """
        Set value of weight_per_channel of RoundToNearestPTQ `config`

        Args:
            weight_per_channel (bool): Quantization granularity based on layer or on channel. If ``True`` then base on
                per channel, otherwise base on per layer.

        Raises:
            TypeError: If `weight_per_channel` is not bool.
        """
        Validator.check_bool(weight_per_channel, "weight_per_channel", self.__class__.__name__)
        if not weight_per_channel:
            raise ValueError("Only supported if `weight_per_channel` is `True` yet.")
        self._config.weight_per_channel = weight_per_channel

    def set_act_quant_dtype(self, act_quant_dtype):
        """
        Set value of act_quant_dtype of RoundToNearestPTQ `config`

        Args:
            act_quant_dtype (QuantDtype): Datatype used to quantize activations.

        Raises:
            TypeError: If `act_quant_dtype` is not QuantDtype.
            ValueError: Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet.
        """
        if not isinstance(act_quant_dtype, QuantDtype):
            raise TypeError(f'The parameter `act quant dtype` must be isinstance of QuantDtype, '
                            f'but got {act_quant_dtype}.')
        if act_quant_dtype != QuantDtype.INT8:
            raise ValueError("Only supported if `act_quant_dtype` is `QuantDtype.INT8` yet.")
        self._config.act_quant_dtype = act_quant_dtype

    def set_weight_quant_dtype(self, weight_quant_dtype):
        """
        Set value of weight_quant_dtype of RoundToNearestPTQ `config`.

        Args:
            weight_quant_dtype (QuantDtype): Datatype used to quantize weight.

        Raises:
            TypeError: If `weight_quant_dtype` is not QuantDtype.
            ValueError: Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet.
        """
        if not isinstance(weight_quant_dtype, QuantDtype):
            raise TypeError(f'The parameter `weight quant dtype` must be isinstance of QuantDtype, '
                            f'but got {weight_quant_dtype}.')
        if weight_quant_dtype != QuantDtype.INT8:
            raise ValueError("Only supported if `weight_quant_dtype` is `QuantDtype.INT8` yet.")
        self._config.weight_quant_dtype = weight_quant_dtype

    def set_act_symmetric(self, act_symmetric):
        """
        Set value of act_symmetric of RoundToNearestPTQ `config`.

        Args:
            act_symmetric (bool): Whether the quantization algorithm use act symmetric or not. If ``True`` then base on
                symmetric, otherwise base on asymmetric.

        Raises:
            TypeError: If `act_symmetric` is not bool.
        """
        Validator.check_bool(act_symmetric, "act_symmetric", self.__class__.__name__)
        if not act_symmetric:
            raise ValueError("Only supported if `act_symmetric` is `True` yet.")
        self._config.act_symmetric = act_symmetric

    def set_weight_symmetric(self, weight_symmetric):
        """
        Set value of weight_symmetric of RoundToNearestPTQ `config`

        Args:
            weight_symmetric (bool): Whether the quantization algorithm use weight symmetric or not. If ``True`` then
                base on symmetric, otherwise base on asymmetric.

        Raises:
            TypeError: If `weight_symmetric` is not bool.
        """
        Validator.check_bool(weight_symmetric, "weight_symmetric", self.__class__.__name__)
        if not weight_symmetric:
            raise ValueError("Only supported if `weight_symmetric` is `True` yet.")
        self._config.weight_symmetric = weight_symmetric

    def set_act_narrow_range(self, act_narrow_range):
        """
        Set value of act_narrow_range of RoundToNearestPTQ `config`

        Args:
            act_narrow_range (bool): Whether the quantization algorithm use act narrow_range or not. If ``True`` then
                base on narrow_range, otherwise base on not narrow_range.

        Raises:
            TypeError: If `act_narrow_range` is not bool.
        """
        Validator.check_bool(act_narrow_range, "act_narrow_range", self.__class__.__name__)
        if act_narrow_range:
            raise ValueError("Only supported if `act_narrow_range` is `False` yet.")
        self._config.act_narrow_range = act_narrow_range

    def set_weight_narrow_range(self, weight_narrow_range):
        """
        Set value of weight_narrow_range of RoundToNearestPTQ `config`

        Args:
            weight_narrow_range (bool): Whether the quantization algorithm use weight narrow_range or not. If
                ``True`` then base on narrow_range, otherwise base on not narrow_range.

        Raises:
            TypeError: If `weight_narrow_range` is not bool.
        """
        Validator.check_bool(weight_narrow_range, "weight_narrow_range", self.__class__.__name__)
        if weight_narrow_range:
            raise ValueError("Only supported if `weight_narrow_range` is `False` yet.")
        self._config.weight_narrow_range = weight_narrow_range

    def set_linear_w8a16(self, is_enable: bool):
        """
        Set value of weight_only of RoundToNearestPTQ `config`

        Args:
            is_enable (bool): Whether enable Linear W8A16 quant algorithm.

        Raises:
            TypeError: If `is_enable` is not bool.
        """
        Validator.check_bool(is_enable, "is_enable_linear_w8a16", self.__class__.__name__)
        self._config.enable_linear_w8a16 = is_enable

    def set_kv_int8_quant(self, is_enable: bool):
        """
        Set value of weight_only of RoundToNearestPTQ `config`

        Args:
            is_enable (bool): Whether enable KVCache Int8 quant algorithm.

        Raises:
            TypeError: If `is_enable` is not bool.
        """
        Validator.check_bool(is_enable, "is_enable_kvcache_int8", self.__class__.__name__)
        self._config.enable_kvcache_int8 = is_enable

    @staticmethod
    def _convert2list(name, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            value = [value]
        elif len(value) > 2:
            raise ValueError("input `{}` len should be less than 3".format(name))
        return value

    def _update_config_from_dict(self, config: dict):
        """Create RoundToNearestPTQ `config` from a dict"""
        quant_dtype_list = RoundToNearest. \
            _convert2list("quant dtype", config.get("quant_dtype", [QuantDtype.INT8, QuantDtype.INT8]))
        per_channel_list = RoundToNearest._convert2list("per channel", config.get("per_channel", [False, True]))
        symmetric_list = RoundToNearest._convert2list("symmetric", config.get("symmetric", [True, True]))
        narrow_range_list = RoundToNearest._convert2list("narrow range", config.get("narrow_range", [False, False]))

        self.set_act_quant_dtype(quant_dtype_list[0])
        self.set_weight_quant_dtype(quant_dtype_list[-1])
        self.set_act_per_channel(per_channel_list[0])
        self.set_weight_per_channel(per_channel_list[-1])
        self.set_act_symmetric(symmetric_list[0])
        self.set_weight_symmetric(symmetric_list[-1])
        self.set_act_narrow_range(narrow_range_list[0])
        self.set_weight_narrow_range(narrow_range_list[-1])
        if config.get("enable_kvcache_int8") is not None:
            self.set_kv_int8_quant(config["enable_kvcache_int8"])
        if config.get("enable_linear_w8a16") is not None:
            self.set_linear_w8a16(config["enable_linear_w8a16"])

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

    def set_deploy(self, is_deploy: bool = True):
        """
        Set algorithm exchange to deploy mode. In deploy mode, algorithm will not statistic quant parameter and not
        quant weight anymore, only load quant parameter and quanted weight from checkpoint instead.
        """
        self._is_deploy = is_deploy

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
