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
"""Quantization utils."""

from mindspore.common.dtype import QuantDtype


def get_quant_dtype_num_bits(quant_dtype: QuantDtype):
    if 0 <= quant_dtype.value() <= 15:
        return quant_dtype.value() + 1
    if 100 <= quant_dtype.value() <= 115:
        return quant_dtype.value() - 99
    raise ValueError("Unsupported QuantDtype.")
