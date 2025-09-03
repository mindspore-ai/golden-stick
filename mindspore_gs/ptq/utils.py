# Copyright 2025 Huawei Technologies Co., Ltd
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
"""ptq quant type"""
from enum import Enum

class QuantType(Enum):
    """
    Quant type of each layer in network.

    - ``UNKNOWN``: A type that has not been identified yet.
    - ``FLOAT``: Unquantified layers.
    - ``W8A16``: The weight is 8bit and activation is float.
    - ``W4A16``: The weight is 4bit and activation is float.
    - ``W8A8``: The weight is 4bit and activation is also 8bit.
    - ``KV8``: The KVCache is 8bit.
    - ``W8A8_DYNAMIC``: The weight is 8bit and activation is 8bit per_token.
    - ``W4A8_DYNAMIC``: The weight is 4bit and activation is 8bit per_token.
    - ``FAQuant``: FlashAttention quant.
    """
    UNKNOWN = "UNKNOWN"
    FLOAT = "FLOAT"
    W8A16 = "W8A16"
    W4A16 = "W4A16"
    W8A8 = "W8A8"
    KV8 = "C8"
    W8A8_DYNAMIC = "W8A8_DYNAMIC"
    W4A8_DYNAMIC = "W4A8_DYNAMIC"
    FAQUANT = "FAQuant"
