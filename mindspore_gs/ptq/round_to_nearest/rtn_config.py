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
"""Config for RTN post-training quantization."""

from mindspore.common.dtype import QuantDtype
from mindspore_gs.comp_algo import CompAlgoConfig


class RTNConfig(CompAlgoConfig):
    """
    Config for simulated quantization aware training.
    See more details in simulated_quantization_aware_training.py
    """

    def __init__(self):
        super(RTNConfig, self).__init__()
        self.act_quant_dtype = QuantDtype.INT8
        self.weight_quant_dtype = QuantDtype.INT8
        self.act_per_channel = False
        self.weight_per_channel = True
        self.act_symmetric = False
        self.weight_symmetric = True
        self.act_narrow_range = False
        self.weight_narrow_range = False
        self.enable_kvcache_int8 = False
        self.enable_linear_w8a16 = True
