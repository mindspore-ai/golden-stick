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
"""learned step size quantization net_policy."""

from mindspore.nn.layer import Conv2d, Dense, Conv2dBnAct
from ..simulated_quantization.simulated_quantization_net_policy import SimulatedNetPolicy
from .learned_step_size_quantization_layer_policy import LearnedStepSizeQuantizationConvLayerPolicy, \
    LearnedStepSizeQuantizationDenseLayerPolicy, LearnedStepSizeQuantizationConvBnLayerPolicy
from .learned_step_size_quantization_config import LearnedStepSizeQuantizationConfig


class LearnedStepSizeQuantizationNetPolicy(SimulatedNetPolicy):
    """
    Derived class of SimulatedNetPolicy. LSQ quantization config.
    """
    def __init__(self, config=LearnedStepSizeQuantizationConfig()):
        super().__init__(config)
        self._config: LearnedStepSizeQuantizationConfig = config

    def build(self):
        super().build()
        self._layer_policy_map[Conv2d] = LearnedStepSizeQuantizationConvLayerPolicy([], [], self._config)
        self._layer_policy_map[Dense] = LearnedStepSizeQuantizationDenseLayerPolicy([], [], self._config)
        self._layer_policy_map[Conv2dBnAct] = LearnedStepSizeQuantizationConvBnLayerPolicy([], [], self._config)
        self._build = True
