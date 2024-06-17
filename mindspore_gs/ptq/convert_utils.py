# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
from mindspore.ops.operations._inner_ops import AntiQuant, Dequant
from mindspore.ops.auto_generate import WeightQuantBatchMatmul, QuantBatchMatmul
from mindspore_gs.quantization.fake_quantizer import FakeQuantParamCell
from mindspore_gs.common.numpy_quant_common import NumpyQuantOps


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
        """forward for antiquant"""
        x = self.anti_quant(x, self.scale1, self.zp0)
        x = self.cast(x, self.outdtype)
        x = self.add(x, self.zp_neg)
        x = self.mul(x, self.scale)
        x = self.cast(x, self.outdtype)
        return x

    def shard(self, strategy):
        """shard strategy for anti quant"""
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
        self.t_scale = Parameter(t_scale)
        self.t_zp = Parameter(t_zp)
        self.mul = msops.Mul()
        self.add = msops.Add()
        self.round = msops.Round()
        self.cast = msops.Cast()

    def construct(self, x):
        """construct network forward"""
        x = self.mul(x, self.t_scale)
        x = self.add(x, self.t_zp)
        x = self.round(x)
        x = self.cast(x, dtype.int8)
        return x

    def shard(self, strategy):
        """shard strategy for quant cell"""
        self.mul.shard((strategy[0], (1,)))
        self.add.shard((strategy[0], (1,)))
        self.round.shard(strategy)
        self.cast.shard(strategy)


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
        scale_ui64 = NumpyQuantOps.trans_fp32_to_u64(scale)
        self.scale = Parameter(Tensor(scale_ui64, dtype=dtype.uint64))
        self.dequant = Dequant()

    def construct(self, x):
        """dequant forward"""
        return self.dequant(x, self.scale)

    def shard(self, strategy):
        self.dequant.shard(strategy)


def convert_to_dequant(input_fqcell: FakeQuantParamCell, weight_fqcell: FakeQuantParamCell, is_transpose=False,
                       strategy=None) -> DequantCell:
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
    dequant_scale = np.array(input_scale[0]) * np.array(weight_scale)
    if is_transpose:
        dequant_scale = dequant_scale.transpose()
    dequant = DequantCell(dequant_scale.tolist())
    if strategy is not None:
        dequant.shard(strategy)
    return dequant


class AntiquantBMMCell(Cell):
    """fused anti quant cell."""
    def __init__(self, scale, offset, out_dtype=dtype.float16, transpose_x: bool = False,
                 transpose_weight: bool = False):
        super().__init__()
        self.out_dtype = out_dtype
        self.scale = Parameter(Tensor(np.squeeze(scale), dtype=self.out_dtype))
        self.zp_neg = Parameter(Tensor(np.squeeze(np.array(offset)) * -1, dtype=self.out_dtype))
        self.weight_qbmm = WeightQuantBatchMatmul(transpose_x, transpose_weight)
        self.cast = msops.Cast()

    def construct(self, x, weight, bias=None):
        """forward for antiquant bmm cell"""
        out = self.weight_qbmm(x, weight, self.scale, self.zp_neg, None, None, bias)
        return self.cast(out, self.out_dtype)

    def shard(self, strategy):
        """shard strategy for antiquant bmm"""
        self.weight_qbmm.shard(strategy)


def convert_to_fusion_antiquant(fqcell: FakeQuantParamCell, transpose_weight=False, transpose_x=False, strategy=None,
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
    anti_quant = AntiquantBMMCell(scale, zp, out_dtype=dst_dtype, transpose_x=transpose_x,
                                  transpose_weight=transpose_weight)
    if strategy is not None:
        anti_quant.shard(strategy)
    return anti_quant


def convert_to_fusion_antiquant_for_deploy(axis, output_channel, data_rank, is_per_channel,
                                           transpose_weight=False, transpose_x=False, strategy=None,
                                           dst_dtype=None) -> AntiquantBMMCell:
    """convert_to_fusion_antiquant_for_deploy."""
    if not dst_dtype:
        if context.get_context("device_target") == "Ascend":
            dst_dtype = dtype.float16
        else:
            dst_dtype = dtype.float32
    if axis < 0:
        axis += data_rank
    pre_dims = axis
    post_dims = data_rank - axis - 1
    param_shape = [1] * pre_dims + [-1] + [1] * post_dims
    if is_per_channel:
        scale = np.ones(output_channel).reshape(param_shape).tolist()
        zp = np.zeros(output_channel).reshape(param_shape).tolist()
    else:
        scale = np.ones(1).reshape(param_shape).tolist()
        zp = np.zeros(1).reshape(param_shape).tolist()
    anti_quant = AntiquantBMMCell(scale, zp, out_dtype=dst_dtype, transpose_x=transpose_x,
                                  transpose_weight=transpose_weight)
    if strategy is not None:
        anti_quant.shard(strategy)
    return anti_quant


class DequantBMMCell(Cell):
    """matmul and dequant fused cell"""

    def __init__(self, scale, offset=None, transpose_a=False, transpose_b=False, dst_dtype=dtype.float16):
        super().__init__()
        self._use_fusion = True
        self.dbmm = QuantBatchMatmul(transpose_x1=transpose_a, transpose_x2=transpose_b, dtype=dst_dtype)
        scale_ui64 = NumpyQuantOps.trans_fp32_to_u64(scale)
        self.scale = Parameter(Tensor(np.squeeze(scale_ui64), dtype=dtype.uint64))
        if offset is None:
            self.offset = None
        else:
            self.offset = Parameter(Tensor(offset, dtype=dtype.float32))

    def construct(self, x1, x2, bias):
        """(matmul(x1, x2) + bias) * scale + offset"""
        return self.dbmm(x1, x2, self.scale, self.offset, bias)

    def shard(self, strategy):
        """shard."""
        self.dbmm.shard(strategy)


def convert_to_dequant_bmm(input_fqcell, weight_fqcell, weight_quant, bias_quant, offset=None, dst_dtype=dtype.float16,
                           transpose_a=False, transpose_b=False, strategy=None):
    """convert_to_dequant_bmm."""

    def _fused_bias(quant_weight, quant_bias, act_offset):
        if quant_weight is None:
            return None
        new_bias = -np.sum(act_offset.astype(np.int32) * quant_weight.asnumpy().astype(np.int32),
                           axis=1 if transpose_b else 0).astype(np.int32)
        if quant_bias is not None:
            new_bias = quant_bias.astype(np.int32) + new_bias
        return new_bias

    input_fq: FakeQuantParam = input_fqcell.fq
    if not isinstance(input_fq, FakeQuantParam):
        raise ValueError("Only support convert FakeQuantParam to DeQuant.")
    weight_fq: FakeQuantParam = weight_fqcell.fq
    if not isinstance(weight_fq, FakeQuantParam):
        raise ValueError("Only support convert FakeQuantParam to DeQuant.")
    input_scale = input_fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
    input_zp = input_fq.attrs.get(FakeQuantParam.attr_key_linear_quant_zero_point, None)
    weight_scale = weight_fq.attrs.get(FakeQuantParam.attr_key_linear_quant_scale, None)
    if input_scale is None:
        raise ValueError("Can not find scale in input FakeQuantParamCell.")
    if weight_scale is None:
        raise ValueError("Can not find scale in weight FakeQuantParamCell.")
    if len(input_scale) != 1:
        raise ValueError("Input only support perlayer quant.")
    dequant_scale = np.array(input_scale[0], dtype=np.float32) * np.array(weight_scale, dtype=np.float32)
    bias_data = _fused_bias(weight_quant, bias_quant, np.array(input_zp))
    bias = Tensor(bias_data, dtype=dtype.int32) if bias_data is not None else None
    dbmm_cell = DequantBMMCell(dequant_scale, offset, transpose_a, transpose_b, dst_dtype)
    if strategy is not None:
        dbmm_cell.shard(strategy)
    return dbmm_cell, bias
