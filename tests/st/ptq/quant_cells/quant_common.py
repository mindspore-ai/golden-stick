# Copyright 2024 Huawei Technologies Co., Ltd
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
"""quantization related numpy quant operations"""

import numpy as np

class NumpyQuantOps:
    """
    numpy quant ops for test.
    """
    @staticmethod
    def anti_quant(data, scale, offset, sqrt_mode=False, dst_type=np.float16):
        """
        convert compressed dtype to orin dtype
        anti_quant_data = (data - offset) * scale (* scale if sqrt_mode is True)
        """
        anti_quant_data = data.astype(np.float32)
        if sqrt_mode:
            return ((anti_quant_data - offset) * scale * scale).astype(dst_type)
        return ((anti_quant_data - offset) * scale).astype(dst_type)

    @staticmethod
    def quant(data, scale, offset, sqrt_mode=False, dst_type=np.int8):
        """
        compress data to lower bit dtype
        quant_data = data / scale + offset
        """
        if sqrt_mode:
            quant_data = np.round(data / (scale * scale) + offset)
        else:
            quant_data = np.round(data / scale + offset)
        return quant_data.astype(dst_type)
