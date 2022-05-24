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
"""
MindSpore golden stick simulated-quantization.
"""

from .simulated_quantization_layer_policy import SimulatedLayerPolicy
from .simulated_quantization_net_policy import SimulatedNetPolicy
from .simulated_quantization_aware_training import SimulatedQuantizationAwareTraining

__all__ = ["SimulatedLayerPolicy", "SimulatedNetPolicy", "SimulatedQuantizationAwareTraining"]
