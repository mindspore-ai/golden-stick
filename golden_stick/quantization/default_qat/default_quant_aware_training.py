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

from ..quant_aware_training import QuantAwareTraining
from .default_net_policy import DefaultNetworkPolicy


class DefaultQuantAwareTraining(QuantAwareTraining):
    """
    Derived class of GoldenStick. Default QAT-algorithm.
    """

    def __init__(self, config=None):
        super(DefaultQuantAwareTraining, self).__init__(config)
        if config is None:
            config = {}
        self._qat_policy = DefaultNetworkPolicy(config)
        self._custom_transforms = {}
        self._custom_layer_policy_map = {}
        if "custom_transforms" in config.keys():
            self._custom_transforms = config["custom_transforms"]
        if "custom_policies" in config.keys():
            self._custom_layer_policy_map = config["custom_policies"]
