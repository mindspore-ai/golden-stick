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
"""SlbNetPolicy."""

from mindspore.nn.layer import Conv2d
from mindspore.rewrite import PatternEngine
from ..net_policy import NetPolicy
from .slb_layer_policy import ConvLayerPolicy
from .slb_quant_config import SlbQuantConfig


class SlbNetPolicy(NetPolicy):
    """
    Derived class of NetworkQConfig. slb network-quant-config.

    """

    def __init__(self, config=SlbQuantConfig()):
        super().__init__(config)
        self._config: SlbQuantConfig = config
        self._build = False
        self._net_layer_policy = None
        self._pattern_engines: [PatternEngine] = []
        self._layer_policy_map: dict = {}

    def build(self):
        """Initialize `SlbNetPolicy`. A `SlbNetPolicy` can only be built once."""
        if self._build:
            return
        self._layer_policy_map[Conv2d] = ConvLayerPolicy([], [], self._config)
        self._build = True
