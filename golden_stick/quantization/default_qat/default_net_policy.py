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
"""DefaultNetworkPolicy."""

from mindspore.nn.layer import Conv2d, Dense, BatchNorm2d, ReLU, Conv2dBnAct
from mindspore.rewrite import PatternEngine
from ..net_policy import NetPolicy
from .default_layer_policy import ConvLayerPolicy, DenseLayerPolicy, ConvBnLayerPolicy
from .default_transforms import Conv2dBnActFuse, DenseBnActFuse, DenseActFuse
from .quant_config import QuantConfig


class DefaultNetworkPolicy(NetPolicy):
    """
    Derived class of NetworkQConfig. Default network-quant-config.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` .
    """

    def __init__(self, config=QuantConfig()):
        super().__init__(config)
        self._config: QuantConfig = config
        self._build = False
        self._net_layer_policy = None
        self._pattern_engines: [PatternEngine] = []
        self._layer_policy_map: dict = {}

    def build(self):
        """Initialize `DefaultNetworkPolicy`. A `DefaultNetworkPolicy` can only be built once."""
        if self._build:
            return
        if self._config.enable_fusion:
            self._pattern_engines.append(PatternEngine([Conv2d, BatchNorm2d, ReLU], Conv2dBnActFuse()))
            self._pattern_engines.append(PatternEngine([Conv2d, BatchNorm2d], Conv2dBnActFuse()))
            self._pattern_engines.append(PatternEngine([Conv2d, ReLU], Conv2dBnActFuse()))
            self._pattern_engines.append(PatternEngine([Dense, BatchNorm2d, ReLU], DenseBnActFuse()))
            self._pattern_engines.append(PatternEngine([Dense, BatchNorm2d], DenseBnActFuse()))
            self._pattern_engines.append(PatternEngine([Dense, ReLU], DenseActFuse()))
        self._layer_policy_map[Conv2d] = ConvLayerPolicy([], [], self._config)
        self._layer_policy_map[Dense] = DenseLayerPolicy([], [], self._config)
        self._layer_policy_map[Conv2dBnAct] = ConvBnLayerPolicy([], [], self._config)
        self._build = True
