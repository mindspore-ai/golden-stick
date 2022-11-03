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
"""Config for aware-training-quantization."""

from ...comp_algo import CompAlgoConfig
from ..constant import QuantDtype


class SlbQuantConfig(CompAlgoConfig):
    """
    Config for SLB(Searching for Low-Bit Weights) QAT-algorithm.
    See more details in slb_quant_aware_training.py
    """

    def __init__(self):
        super(SlbQuantConfig, self).__init__()
        self.act_quant_dtype = QuantDtype.INT8
        self.weight_quant_dtype = QuantDtype.INT1
        self.enable_act_quant = False
        self.enable_bn_calibration = False
        self.epoch_size = -1
        self.has_trained_epoch = -1
        self.t_start_val = 1.0
        self.t_start_time = 0.2
        self.t_end_time = 0.6
        self.t_factor = 1.2
