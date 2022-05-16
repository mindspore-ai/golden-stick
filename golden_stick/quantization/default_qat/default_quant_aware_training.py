# Copyright 2022 Huawei Technologies Co., Ltd
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
"""DefaultQuantAwareTraining."""

from mindspore.nn import Cell
from ..quant_aware_training import QuantAwareTraining
from .default_net_policy import DefaultNetworkPolicy
from .constant import QuantDtype
from .quant_config import QuantConfig


class DefaultQuantAwareTraining(QuantAwareTraining):
    """
    Derived class of GoldenStick. Default QAT-algorithm.
    """

    def __init__(self, config=None):
        super(DefaultQuantAwareTraining, self).__init__(config)
        if config is None:
            config = {}
        self._config: QuantConfig = QuantConfig()
        self._update_qconfig_by_dict(config)

        self._qat_policy = DefaultNetworkPolicy(self._config)
        self._custom_transforms = {}
        self._custom_layer_policy_map = {}
        if "custom_transforms" in config.keys():
            self._custom_transforms = config["custom_transforms"]
        if "custom_policies" in config.keys():
            self._custom_layer_policy_map = config["custom_policies"]

    def set_bn_fold(self, bn_fold):
        """Set value of bn_fold of `_config`"""
        self._config.bn_fold = bn_fold

    def set_act_quant_delay(self, delay):
        """Set value of act_quant_delay of `_config`"""
        self._config.act_quant_delay = delay

    def set_weight_quant_delay(self, delay):
        """Set value of weight_quant_delay of `_config`"""
        self._config.weight_quant_delay = delay

    def set_act_per_channel(self, per_channel):
        """Set value of act_per_channel of `_config`"""
        self._config.act_per_channel = per_channel

    def set_weight_per_channel(self, per_channel):
        """Set value of weight_per_channel of `_config`"""
        self._config.weight_per_channel = per_channel

    def set_act_symmetric(self, symmetric):
        """Set value of act_symmetric of `_config`"""
        self._config.act_symmetric = symmetric

    def set_weight_symmetric(self, symmetric):
        """Set value of weight_symmetric of `_config`"""
        self._config.weight_symmetric = symmetric

    def _update_qconfig_by_dict(self, config: dict):
        """Update `_config` from a dict"""
        self._config.bn_fold = config.get("bn_fold", True)
        self._config.freeze_bn = config.get("freeze_bn", 10000000)
        quant_delay = config.get("quant_delay", [0, 0])
        self._config.act_quant_delay = quant_delay[0]
        self._config.weight_quant_delay = quant_delay[1]
        quant_dtype = config.get("quant_dtype", [QuantDtype.INT8, QuantDtype.INT8])
        self._config.act_quant_dtype = quant_dtype[0]
        self._config.weight_quant_dtype = quant_dtype[1]
        per_channel = config.get("per_channel", [False, True])
        self._config.act_per_channel = per_channel[0]
        self._config.weight_per_channel = per_channel[1]
        symmetric = config.get("symmetric", [False, False])
        self._config.act_symmetric = symmetric[0]
        self._config.weight_symmetric = symmetric[1]
        narrow_range = config.get("narrow_range", [False, False])
        self._config.act_narrow_range = narrow_range[0]
        self._config.weight_narrow_range = narrow_range[1]
        self._config.one_conv_fold = config.get("one_conv_fold", True)

    def apply(self, network: Cell) -> Cell:
        self._qat_policy.build()
        return super(DefaultQuantAwareTraining, self).apply(network)
