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
from mindspore.ops.auto_generate import WeightQuantBatchMatmul

from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell


class AntiQuantCell(Cell):
    """AntiQuantCell, warp AntiQuant to support per-channel AntiQuant."""
    def __init__(self, scale: list, zp: list, dst_dtype=dtype.float16, sqrt_mode=False):
        super().__init__()
        self.outdtype = dst_dtype
        self.scale = Parameter(Tensor(scale, dtype=self.outdtype))
        self.zp_neg = Parameter(Tensor(np.array(zp) * -1, dtype=self.outdtype))
        self.anti_quant = AntiQuant(sqrt_mode)
        self.mul = msops.Mul()
        self.add = msops.Add()
        self.cast = msops.Cast()
        self.scale1 = Parameter(Tensor(1., dtype=self.outdtype), requires_grad=False)
        self.zp0 = Parameter(Tensor(0., dtype=self.outdtype), requires_grad=False)

    def construct(self, x):
        x = self.anti_quant(x, self.scale1, self.zp0)
        x = self.cast(x, self.outdtype)
        x = self.add(x, self.zp_neg)
        x = self.mul(x, self.scale)
        x = self.cast(x, self.outdtype)
        return x

    def shard(self, strategy):
        self.anti_quant.shard(strategy)


def convert_to_antiquant(fqcell: FakeQuantParamCell, strategy=None, dst_dtype=None) -> AntiQuantCell:
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
    if not dst_dtype:
        if context.get_context("device_target") == "Ascend":
            dst_dtype = dtype.float16
        else:
            dst_dtype = dtype.float32
    anti_quant = AntiQuantCell(scale, zp, dst_dtype=dst_dtype)
    if strategy is not None:
        anti_quant.shard(strategy)
    return anti_quant


class QuantCell(Cell):
    """QuantCell, warp Quant to support serialize and deserialize."""
    def __init__(self, t_scale: Tensor, t_zp: Tensor):
        super().__init__()
        if t_scale.shape != t_zp.shape:
            raise ValueError(f"Size of scale({t_scale.shape}) should be equal to size of zp({t_zp.shape}).")
        t_scale = 1 / t_scale
        self._is_perchannel: bool = t_scale.shape != (1,)
        if self._is_perchannel:
            self.t_scale = Parameter(Tensor([1.0], dtype=dtype.float32))
            self.t_zp = Parameter(Tensor([0.0], dtype=dtype.float32))
            self.mul = msops.Mul()
            self.add = msops.Add()
            self.mul_param = Parameter(t_scale)
            self.add_param = Parameter(t_zp)
        else:
            self.t_scale = Parameter(t_scale)
            self.t_zp = Parameter(t_zp)
        self.quant = Quant(self.t_scale.asnumpy().tolist()[0], self.t_zp.asnumpy().tolist()[0])

    def update_ascend_quant(self):
        self.quant.add_prim_attr('scale', self.t_scale.asnumpy().tolist()[0])
        self.quant.add_prim_attr('offset', self.t_zp.asnumpy().tolist()[0])

    def construct(self, x):
        if self._is_perchannel:
            x = self.mul(x, self.mul_param)
            x = self.add(x, self.add_param)
        return self.quant(x)

    def shard(self, strategy):
        self.quant.shard(strategy)


def convert_to_quant(fqcell: FakeQuantParamCell, strategy=None) -> QuantCell:
    """Convert FakeQuantParamCell to Quant."""
    fq: FakeQuantParam = fqcell.fq
    if not isinstance(fq, FakeQuantParam):
        raise ValueError("Only support convert FakeQuantParam to Quant.")
    scale = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
    zp = fq.attrs.get(FakeQuantParam.attr_key_linear_quant_zero_point, None)
    if scale is None:
        raise ValueError("Can not find scale in FakeQuantParamCell.")
    if zp is None:
        raise ValueError("Can not find zp in FakeQuantParamCell.")
    quant_cell = QuantCell(Tensor(scale, dtype=dtype.float32), Tensor(zp, dtype=dtype.float32))
    if strategy is not None:
        quant_cell.shard(strategy)
    return quant_cell


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


class AntiquantBMMCell(Cell):
    """fused anti quant cell."""
    def __init__(self,
                 scale,
                 offset,
                 out_dtype=dtype.float16,
                 transpose_x: bool = False,
                 transpose_weight: bool = False):
        super().__init__()
        self.out_dtype = out_dtype
        self.scale = Parameter(Tensor(np.squeeze(scale), dtype=self.out_dtype))
        self.zp_neg = Parameter(Tensor(np.squeeze(np.array(offset)) * -1, dtype=self.out_dtype))
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight)
        self.cast = msops.Cast()

    def construct(self,
                  x,
                  weight,
                  bias=None):
        out = self.weight_qbmm(x, weight, self.scale, self.zp_neg, None, None, bias)
        return self.cast(out, self.out_dtype)

    def shard(self, strategy):
        self.weight_qbmm.shard(strategy)


def convert_to_fusion_antiquant(fqcell: FakeQuantParamCell,
                                transpose_weight=False,
                                transpose_x=False,
                                strategy=None,
                                dst_dtype=None) -> AntiquantBMMCell:
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
    if not dst_dtype:
        if context.get_context("device_target") == "Ascend":
            dst_dtype = dtype.float16
        else:
            dst_dtype = dtype.float32
    anti_quant = AntiquantBMMCell(scale,
                                  zp,
                                  out_dtype=dst_dtype,
                                  transpose_x=transpose_x,
                                  transpose_weight=transpose_weight)
    if strategy is not None:
        anti_quant.shard(strategy)
    return anti_quant
