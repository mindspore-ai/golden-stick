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
MindSpore golden stick simulated-quantization ops.
"""

from .min_max_update_perlayer import MinMaxUpdatePerLayer
from .fake_quant_perlayer import FakeQuantPerLayer
from .fake_quant_perchannel import FakeQuantPerChannel
from .min_max_update_perchannel import MinMaxUpdatePerChannel

__all__ = ["MinMaxUpdatePerLayer", "FakeQuantPerLayer", "FakeQuantPerChannel", "MinMaxUpdatePerChannel"]
