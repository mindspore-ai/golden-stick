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
"""NetworkQConfig."""
from typing import Optional

from .layer_policy import LayerPolicy
from .transformer import Transformer


class NetPolicy:
    """
    Base class for network quantize configure.

    Args:
        config (Dict): User config for QAT. Config specification is default by derived class.

    Note:
        Derived class must define `_pattern_engines` and `_support_layer_map` in constructor.
    """

    def __init__(self, config=None):
        self._pattern_engines: [Transformer] = []
        self._layer_policy_map: dict = {}
        self._net_layer_policy: Optional[LayerPolicy] = None

    def get_transformers(self) -> [Transformer]:
        """
        Get transformers.

        """
        return self._pattern_engines

    def get_layer_policy_map(self) -> {str, LayerPolicy}:
        """
        Get layer policy map.

        """
        return self._layer_policy_map

    def get_layer_policy(self, layer_type) -> Optional[LayerPolicy]:
        """
        Get layer policy by type.

        """
        return self._layer_policy_map.get(layer_type)

    def get_net_layer_policy(self) -> Optional[LayerPolicy]:
        """
        Get network layer policy.

        """
        return self._net_layer_policy
