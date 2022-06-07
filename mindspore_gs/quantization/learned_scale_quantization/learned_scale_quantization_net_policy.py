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
"""learned scale quantization net_policy."""

from mindspore.nn.layer import Conv2d, Dense, Conv2dBnAct
from ..simulated_quantization.simulated_quantization_net_policy import SimulatedNetPolicy
from .learned_scale_quantization_layer_policy import ConvLayerPolicy, DenseLayerPolicy, ConvBnLayerPolicy
from .learned_scale_quantization_config import LearnedScaleQuantizationConfig


class LearnedScaleQuantizationNetPolicy(SimulatedNetPolicy):
    """
    Derived class of SimulatedNetPolicy. LSQ quantization config.
    """
    def __init__(self, config=LearnedScaleQuantizationConfig()):
        super().__init__(config)
        self._config: LearnedScaleQuantizationConfig = config

    def build(self):
        super().build()
        self._layer_policy_map[Conv2d] = ConvLayerPolicy([], [], self._config)
        self._layer_policy_map[Dense] = DenseLayerPolicy([], [], self._config)
        self._layer_policy_map[Conv2dBnAct] = ConvBnLayerPolicy([], [], self._config)
        self._build = True
