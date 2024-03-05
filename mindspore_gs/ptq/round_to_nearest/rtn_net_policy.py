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
"""RTNNetPolicy."""

from mindformers.modules import Linear, KVCacheMgr
from mindspore.rewrite import PatternEngine

from mindspore_gs.quantization.net_policy import NetPolicy
from mindspore_gs.ptq.ptq_config import PTQConfig
from .rtn_layer_policy import LinearLayerPolicy, KVCacheMgrPolicy


class RTNNetPolicy(NetPolicy):
    """
    Derived class of NetworkQConfig. RoundToNearestPTQ config.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` .
    """

    def __init__(self, config=PTQConfig()):
        super().__init__(config)
        self._config: PTQConfig = config
        self._build = False
        self._net_layer_policy = None
        self._pattern_engines: [PatternEngine] = []
        self._layer_policy_map: dict = {}

    def build(self):
        """Initialize `RTNNetPolicy`. A `RTNNetPolicy` can only be built once."""
        if self._build:
            return
        if self._config.weight_only:
            self._layer_policy_map[Linear] = LinearLayerPolicy([], [], self._config)
        if self._config.enable_kvcache_int8:
            self._layer_policy_map[KVCacheMgr] = KVCacheMgrPolicy([], [], self._config)
        self._build = True
