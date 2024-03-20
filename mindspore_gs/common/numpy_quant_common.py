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

    @staticmethod
    def dequant(data, scale, sqrt_mode=False, dst_type=np.float16):
        """
        dequant data from compressed dtype to origin dtype
        dequant_data = data * scale
        """
        if sqrt_mode:
            quant_data = data * scale * scale
        else:
            quant_data = data * scale
        return quant_data.astype(dst_type)

    @staticmethod
    def trans_fp32_to_u64(scale_fp32: list):
        """transport fp32 data to uint64"""
        fp32_scale_deq = np.array(scale_fp32, dtype=np.float32)
        ui32_scale_deq = np.frombuffer(fp32_scale_deq, np.uint32).reshape(fp32_scale_deq.shape)
        ui64_scale_deq = np.zeros(fp32_scale_deq.shape, np.uint64)
        ui64_scale_deq |= np.uint64(ui32_scale_deq)
        return ui64_scale_deq.tolist()


class NumpyFullQuant:
    """full quant process using numpy"""
    def __init__(self,
                 weight_scale,
                 act_scale,
                 act_offset):
        self.weight_scale = weight_scale
        self.act_scale = act_scale
        self.act_offset = act_offset

    def process(self, activation, weight, bias):
        quant_act = NumpyQuantOps.quant(activation, self.act_scale, self.act_offset)
        quant_weight = NumpyQuantOps.quant(weight, self.weight_scale, 0)
        quant_bias = (bias / (self.act_scale * self.weight_scale)).astype(np.int32)
        fused_bias = -np.sum(self.act_offset.astype(np.int32) * quant_weight.astype(np.int32), axis=0) + quant_bias
        quant_result = np.matmul(quant_act.astype(np.int32), quant_weight.astype(np.int32)) + fused_bias
        dequant_result = quant_result * self.act_scale * self.weight_scale
        return dequant_result

    def orin_process(self, activation, weight, bias):
        return np.matmul(activation, weight) +  bias
