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
from .default_transforms import Conv2dBnActFuse, DenseBnActFuse


class DefaultNetworkPolicy(NetPolicy):
    """
    Derived class of NetworkQConfig. Default network-quant-config.

    Supported Config:
        ``quant_delay`` ``quant_dtype`` ``per_channel`` ``symmetric`` ``narrow_range`` ``one_conv_fold``.
    """

    def __init__(self, config=None):
        super().__init__(config)
        if config is None:
            config = {}
        self._net_layer_policy = None
        self._pattern_engines: [PatternEngine] = [
            PatternEngine([Conv2d, BatchNorm2d], Conv2dBnActFuse()),
            PatternEngine([Dense, BatchNorm2d], DenseBnActFuse()),
            PatternEngine([Conv2d, BatchNorm2d, ReLU], Conv2dBnActFuse()),
            PatternEngine([Dense, BatchNorm2d, ReLU], DenseBnActFuse()),
        ]
        self._layer_policy_map: dict = {
            Conv2d: ConvLayerPolicy([], [], config),
            Dense: DenseLayerPolicy([], [], config),
            Conv2dBnAct: ConvBnLayerPolicy([], [], config)
        }
