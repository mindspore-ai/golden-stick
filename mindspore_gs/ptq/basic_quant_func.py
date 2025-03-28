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
"""Quantization utils."""

import numpy as np
import mindspore as ms
from mindspore.common.dtype import QuantDtype
from mindspore import Tensor
from mindspore import dtype as msdtype


QUANT_DTYPE_NUM_BITS = {
    msdtype.int8: 8,
    msdtype.qint4x2: 4
}


def np_int4data_pack_to_int8(np_data):
    np_data = np_data.astype(np.int8)
    np_data &= 0x000F
    np_data[::, 0::2] <<= 0
    np_data[::, 1::2] <<= 4
    np_int4_data = np_data[::, 0::2] | np_data[::, 1::2]
    return np_int4_data


def np_int4data_pack_to_int8_3d(np_data):
    np_data = np_data.astype(np.int8)
    np_data &= 0x000F
    np_data[::, ::, 0::2] <<= 0
    np_data[::, ::, 1::2] <<= 4
    np_int4_data = np_data[::, ::, 0::2] | np_data[::, ::, 1::2]
    return np_int4_data


def get_quant_dtype_num_bits(quant_dtype: QuantDtype):
    if 0 <= quant_dtype.value() <= 15:
        return quant_dtype.value() + 1
    if 100 <= quant_dtype.value() <= 115:
        return quant_dtype.value() - 99
    raise ValueError("Unsupported QuantDtype.")


def cal_quantization_params(input_min, input_max, quant_min, quant_max, symmetric=False, high_precision=True):
    r"""
    Calculate quantization params for scale and zero point.

    Args:
        input_min (numpy.ndarray): The dimension of channel or 1.
        input_max (numpy.ndarray): The dimension of channel or 1.
        quant_min (int): The minimum quantization integer.
        quant_max (int): The maximum quantization integer.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.
        high_precision(bool): Whether to use high float precision while calculating. Default: True.

    Returns:
        scale (numpy.ndarray): quantization param.
        zero point (numpy.ndarray): quantization param.
    """
    if not isinstance(input_min, Tensor):
        input_min = Tensor(input_min)
    if not isinstance(input_max, Tensor):
        input_max = Tensor(input_max)

    if input_min.shape != input_max.shape:
        raise ValueError("input min shape should be equal to input max.")
    if (input_max == input_min).all():
        return ms.ops.ones(input_min.shape), ms.ops.zeros(input_min.shape)

    # calculate scale
    if symmetric:
        input_max = ms.ops.maximum(ms.ops.abs(input_min), ms.ops.abs(input_max))
        input_min = -input_max
    input_min = input_min.astype(msdtype.float64)
    input_max = input_max.astype(msdtype.float64)
    scale = (input_max - input_min) / (quant_max - quant_min)

    # calculate zero point
    zp_double = quant_min - input_min / scale
    if not high_precision:
        scale = scale.astype(msdtype.float32)
    if symmetric:
        zp = ms.ops.zeros_like(zp_double).astype(msdtype.float64 if high_precision else msdtype.float32)
    else:
        zp = ms.ops.round(zp_double).astype(msdtype.float64 if high_precision else msdtype.float32)
    return scale, zp

def get_float_max_min(rank, tensor, min_op, max_op, quant_axis):
    """get_float_max_min"""
    if rank not in (2, 3):
        raise ValueError(f"Only support rank of tensor being 2 and 3, but got {rank}")
    if rank == 2:
        minmax_axis = 1 if quant_axis == 0 else 0
        float_max = max_op(tensor, minmax_axis, keepdims=True)[0].asnumpy()
        float_min = min_op(tensor, minmax_axis, keepdims=True)[0].asnumpy()
    else:
        float_max, float_min = [], []
        minmax_axis = 1 if quant_axis == 1 else 0
        for i in range(tensor.shape[0]):
            float_max.append(max_op(tensor[i], minmax_axis, keepdims=True)[0].asnumpy())
            float_min.append(min_op(tensor[i], minmax_axis, keepdims=True)[0].asnumpy())
        float_max = np.array(float_max)
        float_min = np.array(float_min)

    return float_max, float_min

def quant_tensor(tensor: Tensor, min_op, max_op, narrow_range, symmetric, need_group, group_size,
                 quant_dtype=msdtype.int8, quant_axis=-1, if_quant_data: bool = True, if_pesudo_quant: bool = False,
                 is_transpose: bool = True, high_precision_params=True):
    """quant_tensor"""
    if quant_dtype not in QUANT_DTYPE_NUM_BITS.keys():
        raise ValueError(f"Only support quant to {QUANT_DTYPE_NUM_BITS.keys()}, but got {quant_dtype}")
    num_bits = QUANT_DTYPE_NUM_BITS[quant_dtype]

    signed = True

    org_shape = tensor.shape
    if need_group and group_size > 0:
        if len(tensor.shape) == 3:
            tensor_shape = (org_shape[0], -1, group_size) if is_transpose else (org_shape[0], group_size, -1)
        else:
            tensor_shape = (-1, group_size) if is_transpose else (group_size, -1)
        tensor = tensor.reshape(tensor_shape)

    if quant_axis == -1:
        float_max = max_op(tensor)[0].reshape(-1)
        float_min = min_op(tensor)[0].reshape(-1)
    else:
        rank = len(tensor.shape)
        float_max, float_min = get_float_max_min(rank, tensor, min_op, max_op, quant_axis)
    quant_min, quant_max = get_quant_min_max(num_bits=num_bits, signed=signed, narrow_range=narrow_range)
    scale, zp = cal_quantization_params(float_min, float_max, quant_min, quant_max, symmetric=symmetric,
                                        high_precision=high_precision_params)

    if if_quant_data:
        qtensor = quant_tensor_data(tensor, scale, zp, quant_min, quant_max, quant_axis,
                                    msdtype.int8)
        if if_pesudo_quant:
            t_scale = Tensor(scale, tensor.dtype)
            t_zp = Tensor(zp, tensor.dtype)
            qtensor = (qtensor - t_zp) * t_scale
        qtensor = qtensor.reshape(org_shape)
    else:
        qtensor = None
    if need_group and quant_axis != -1:
        if len(tensor.shape) == 3:
            scale_zp_shape = (org_shape[0], org_shape[quant_axis], -1) if is_transpose \
                else (org_shape[0], -1, org_shape[quant_axis])
            scale = scale.reshape(scale_zp_shape).transpose((0, 2, 1)) if is_transpose \
                else scale.reshape(scale_zp_shape)
            zp = zp.reshape(scale_zp_shape).transpose((0, 2, 1)) if is_transpose \
                else zp.reshape(scale_zp_shape)
        else:
            scale_zp_shape = (org_shape[quant_axis], -1) if is_transpose else (-1, org_shape[quant_axis])
            scale = scale.reshape(scale_zp_shape).transpose(1, 0) if is_transpose else scale.reshape(scale_zp_shape)
            zp = zp.reshape(scale_zp_shape).transpose(1, 0) if is_transpose else zp.reshape(scale_zp_shape)
    return scale, zp, qtensor


def convert_fp32_to_int64(scale) -> np.ndarray:
    """convert_fp32_to_int64"""
    new_scale = np.frombuffer(scale.tobytes(), dtype=np.uint32)
    return new_scale.astype(np.int64)


def get_quant_min_max(num_bits=8, signed=True, narrow_range=False):
    """Calculate quantization params for minimum/maximum quantization integer"""
    if signed:
        quant_min = 0 - 2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1
    else:
        quant_min = 0
        quant_max = 2 ** num_bits - 1
    if narrow_range:
        quant_min = quant_min + 1
    return quant_min, quant_max


def quant_tensor_data(tensor: Tensor, scale, zero_point, quant_min, quant_max, data_axis=-1, dtype=ms.dtype.int8):
    r"""
    Calculate int8/uint8 weight from fp32. the formula is defined as:

    .. math::
        int8/uint8 = round(float/scale) + offset

    Args:
        tensor (Tensor): The dimension of channel or 1. Should be NCHW.
        scale (numpy.ndarray): The dimension of channel or 1.
        zero_point (numpy.ndarray): The dimension of channel or 1.
        quant_min (int): The minimum quantization integer.
        quant_max (int): The maximum quantization integer.
        data_axis (int): Quantize axis.

    Returns:
        weight (numpy.ndarray): The dimension of channel or 1.
    """
    if scale.shape != zero_point.shape:
        raise ValueError("`scale` and `zero_point` should have the same shape.")
    if scale.shape[0] < 0:
        raise ValueError("`scale` and `zero_point` shape should be greater than zero.")
    if tensor.shape[data_axis] != scale.shape[data_axis]:
        raise ValueError(f"Dim({tensor.shape[data_axis]}) of `data`'s `data_axis`({data_axis}) should be equal to "
                         f"`scale`'s shape({scale.shape}[0]).")
    if data_axis >= len(tensor.shape):
        raise ValueError("`data_axis` out of range of `data`'s shape.")

    t_scale = Tensor(scale, dtype=tensor.dtype)
    t_zp = Tensor(zero_point, dtype=tensor.dtype)
    quanted_data = tensor / t_scale
    quanted_data = quanted_data + t_zp
    quanted_data = ms.ops.round(quanted_data)
    quanted_data = ms.ops.clamp(quanted_data, quant_min, quant_max)
    quanted_data = ms.ops.cast(quanted_data, dtype)
    return quanted_data


def quant_data(data, scale, zero_point, quant_min, quant_max, data_axis=-1):
    r"""
    Calculate int8/uint8 weight from fp32. the formula is defined as:

    .. math::
        int8/uint8 = round(float/scale) + offset

    Args:
        data (numpy.ndarray): The dimension of channel or 1. Should be NCHW.
        scale (numpy.ndarray): The dimension of channel or 1.
        zero_point (numpy.ndarray): The dimension of channel or 1.
        quant_min (int): The minimum quantization integer.
        quant_max (int): The maximum quantization integer.
        data_axis (int): Quantize axis.

    Returns:
        weight (numpy.ndarray): The dimension of channel or 1.
    """
    if scale.shape != zero_point.shape:
        raise ValueError("`scale` and `zero_point` should have the same shape.")
    if scale.shape[0] < 0:
        raise ValueError("`scale` and `zero_point` shape should be greater than zero.")
    if data.shape[data_axis] != scale.shape[0]:
        raise ValueError("Dim of `data`'s `data_axis` should be equal to `scale`'s shape.")
    if data_axis >= len(data.shape):
        raise ValueError("`data_axis` out of range of `data`'s shape.")

    if data_axis >= 0:
        # for perchannel
        if data_axis == 0:
            # `Conv2d` or `Dense` op weight
            shape_list = [-1] + [1] * len(data.shape[1:])
            scale = scale.reshape(shape_list)
            zero_point = zero_point.reshape(shape_list)
        elif data_axis == 1:
            # `DepthwiseConv2d` op weight
            shape_list = [1, -1] + [1] * len(data.shape[2:])
            scale = scale.reshape(shape_list)
            zero_point = zero_point.reshape(shape_list)
        else:
            raise ValueError("Unsupported data_axis({})".format(data_axis))

    quanted_data = np.round((data / scale) + zero_point)
    quanted_data[quanted_data > quant_max] = quant_max
    quanted_data[quanted_data < quant_min] = quant_min
    return quanted_data
