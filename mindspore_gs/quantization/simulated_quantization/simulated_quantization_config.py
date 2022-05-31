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

from ..constant import QuantDtype


class SimulatedQuantizationConfig:
    """Config for aware-training-quantization."""
    def __init__(self):
        self.bn_fold = False
        self.freeze_bn = 10000000
        self.act_quant_delay = 0
        self.weight_quant_delay = 0
        self.act_quant_dtype = QuantDtype.INT8
        self.weight_quant_dtype = QuantDtype.INT8
        self.act_per_channel = False
        self.weight_per_channel = False
        self.act_symmetric = False
        self.weight_symmetric = False
        self.act_narrow_range = False
        self.weight_narrow_range = False
        self.one_conv_fold = True
        self.enable_fusion = False
