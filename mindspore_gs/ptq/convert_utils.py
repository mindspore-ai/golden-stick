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
"""Convert network to target backend quant network from mindposre quant network."""

import numpy as np
from mindspore import Parameter, Tensor, dtype, context
from mindspore.nn import Cell
from mindspore.ops import operations as msops
from mindspore.ops.operations import FakeQuantParam
from mindspore.ops.operations._inner_ops import AntiQuant, Quant, Dequant
from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell


class AntiQuantCell(Cell):
    """AntiQuantCell, warp AntiQuant to support per-channel AntiQuant."""
    def __init__(self, scale: list, zp: list):
        super().__init__()
        if context.get_context("device_target") == "Ascend":
            outdtype = dtype.float16
        else:
            outdtype = dtype.float32
        self.scale = Parameter(Tensor(scale, dtype=outdtype))
        self.zp_neg = Parameter(Tensor(np.array(zp) * -1, dtype=dtype.int32))
        self.anti_quant = AntiQuant(1., 0.)
        self.mul = msops.Mul()
        self.add = msops.Add()

    def construct(self, x):
        x = self.anti_quant(x)
        x = self.add(x, self.zp_neg)
        x = self.mul(x, self.scale)
        return x


def convert_to_antiquant(fqcell: FakeQuantParamCell) -> AntiQuantCell:
    """Convert FakeQuantParamCell to AntiQuantCell."""
    fq: FakeQuantParam = fqcell.fq
    if not isinstance(fq, FakeQuantParam):
        raise ValueError("Only support convert FakeQuantParam to AntiQuant.")
    scale = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
    zp = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_zero_point, None)
    if scale is None:
        raise ValueError("Can not find scale in FakeQuantParamCell.")
    if zp is None:
        raise ValueError("Can not find zp in FakeQuantParamCell.")
    return AntiQuantCell(scale, zp)


class QuantCell(Cell):
    """QuantCell, warp Quant to support serialize and deserialize."""
    def __init__(self, scale: float, zp: float):
        super().__init__()
        self.scale = Parameter(Tensor([scale], dtype=dtype.float32))
        self.zp = Parameter(Tensor([zp], dtype=dtype.float32))
        self.quant = None
        self.update_ascend_quant()

    def update_ascend_quant(self):
        scale = self.scale.asnumpy().tolist()[0]
        zp = self.zp.asnumpy().tolist()[0]
        self.quant = Quant(scale, zp)

    def construct(self, x):
        return self.quant(x)


def convert_to_quant(fqcell: FakeQuantParamCell) -> QuantCell:
    """Convert FakeQuantParamCell to Quant."""
    fq: FakeQuantParam = fqcell.fq
    if not isinstance(fq, FakeQuantParam):
        raise ValueError("Only support convert FakeQuantParam to Quant.")
    scale = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
    zp = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_zero_point, None)
    if scale is None:
        raise ValueError("Can not find scale in FakeQuantParamCell.")
    if scale is None:
        raise ValueError("Can not find zp in FakeQuantParamCell.")
    return QuantCell(scale[0], zp[0])


class DequantCell(Cell):
    """DequantCell, warp Dequant to support zero-point."""
    def __init__(self, scale: list):
        super().__init__()
        scale_ui64 = DequantCell._trans_fp32_to_u64(scale)
        self.scale = Parameter(Tensor(scale_ui64, dtype=dtype.uint64))
        self.dequant = Dequant()

    @staticmethod
    def _trans_fp32_to_u64(scale_fp32: list):
        fp32_scale_deq = np.array(scale_fp32, dtype=np.float32)
        ui32_scale_deq = np.frombuffer(fp32_scale_deq, np.uint32)
        ui64_scale_deq = np.zeros(fp32_scale_deq.shape, np.uint64)
        ui64_scale_deq |= np.uint64(ui32_scale_deq)
        return ui64_scale_deq.tolist()

    def construct(self, x):
        return self.dequant(x, self.scale)


def convert_to_dequant(input_fqcell: FakeQuantParamCell, weight_fqcell: FakeQuantParamCell) -> DequantCell:
    """Convert FakeQuantParamCell to DequantCell."""
    input_fq: FakeQuantParam = input_fqcell.fq
    if not isinstance(input_fq, FakeQuantParam):
        raise ValueError("Only support convert FakeQuantParam to DeQuant.")
    weight_fq: FakeQuantParam = weight_fqcell.fq
    if not isinstance(weight_fq, FakeQuantParam):
        raise ValueError("Only support convert FakeQuantParam to DeQuant.")
    input_scale = input_fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
    weight_scale = weight_fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
    if input_scale is None:
        raise ValueError("Can not find scale in input FakeQuantParamCell.")
    if weight_scale is None:
        raise ValueError("Can not find scale in weight FakeQuantParamCell.")
    if len(input_scale) != 1:
        raise ValueError("Input only support perlayer quant.")
    return DequantCell(input_scale[0] * weight_scale)
