# Copyright 2023-2024 Huawei Technologies Co., Ltd
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

from mindspore.rewrite import PatternEngine
from mindspore_gs.common import logger
from mindspore_gs.ptq.ptq_policy import PTQNetPolicy
from mindspore_gs.ptq.context import InnerPTQConfig


class RTNNetPolicy(PTQNetPolicy):
    """
    Derived class of NetworkQConfig. RoundToNearestPTQ config.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` .
    """

    def __init__(self, config=InnerPTQConfig()):
        super().__init__(config)
        self._config: InnerPTQConfig = config
        self._build = False
        self._net_layer_policy = None
        self._pattern_engines: [PatternEngine] = []
        self._layer_policy_map: dict = {}

    def build(self):
        """Initialize `RTNNetPolicy`. A `RTNNetPolicy` can only be built once."""
        if self._build:
            return
        for key, value in PTQNetPolicy.register_policy_map.items():
            if key[0] is not RTNNetPolicy:
                continue
            policy = value(self._config)
            layer_type = key[1]
            logger.info(f"Map layer_policy for {layer_type} to {type(policy)}")
            self._layer_policy_map[layer_type] = policy
        self._build = True
