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
"""learned step size quantization config"""
from ..simulated_quantization.simulated_quantization_config import SimulatedQuantizationConfig


class LearnedStepSizeQuantizationConfig(SimulatedQuantizationConfig):
    """
    Config for learned step size quantization aware training.
    See more details in learned_step_size_quantization_aware_training.py
    """
    def __init__(self):
        super(LearnedStepSizeQuantizationConfig, self).__init__()
        self.weight_symmetric = True
        self.act_symmetric = True
        self.weight_narrow_range = True
        self.act_narrow_range = True
        self.weight_neg_trunc = False
        self.act_neg_trunc = False
        self.freeze_bn = 0
