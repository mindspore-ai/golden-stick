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

import numpy as np
import mindspore as ms
from mindspore.common.dtype import QuantDtype
from mindspore import Tensor

__all__ = ["compute_kl_threshold", "fold_batchnorm", "cal_quantization_params", "get_quant_min_max",
           "get_quant_dtype_num_bits"]


def get_quant_dtype_num_bits(quant_dtype: QuantDtype):
    """Get number of bits for quantization data type.

    Args:
        quant_dtype (QuantDtype): The quantization data type.

    Returns:
        int: The number of bits for the given quantization data type.

    Raises:
        ValueError: If the quantization data type is not supported.
    """
    if 0 <= quant_dtype.value() <= 15:
        return quant_dtype.value() + 1
    if 100 <= quant_dtype.value() <= 115:
        return quant_dtype.value() - 99
    raise ValueError("Unsupported QuantDtype.")


def cal_quantization_params(input_min,
                            input_max,
                            quant_min,
                            quant_max,
                            symmetric=False):
    r"""
    Calculate quantization params for scale and zero point.

    Args:
        input_min (numpy.ndarray): The dimension of channel or 1.
        input_max (numpy.ndarray): The dimension of channel or 1.
        quant_min (int): The minimum quantization integer.
        quant_max (int): The maximum quantization integer.
        symmetric (bool): Whether the quantization algorithm is symmetric or not. Default: False.

    Returns:
        scale (numpy.ndarray): quantization param.
        zero point (numpy.ndarray): quantization param.
    """

    if input_min.shape != input_max.shape:
        raise ValueError("input min shape should be equal to input max.")
    if (input_max == input_min).all():
        return np.ones(input_min.shape), np.zeros(input_min.shape)

    # calculate scale
    if symmetric:
        input_max = np.maximum(np.abs(input_min), np.abs(input_max))
        input_min = -input_max
    input_min = input_min.astype(np.float64)
    input_max = input_max.astype(np.float64)
    scale = (input_max - input_min) / (quant_max - quant_min)

    # calculate zero point
    zp_double = quant_min - input_min / scale
    if symmetric:
        zp = np.zeros_like(zp_double).astype(np.float64)
    else:
        zp = np.round(zp_double).astype(np.float64)
    return scale, zp


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
    if tensor.shape[data_axis] != scale.shape[0]:
        raise ValueError(f"Dim({tensor.shape[data_axis]}) of `data`'s `data_axis`({data_axis}) should be equal to "
                         f"`scale`'s shape({scale.shape}[0]).")
    if data_axis >= len(tensor.shape):
        raise ValueError("`data_axis` out of range of `data`'s shape.")

    if data_axis >= 0:
        # for perchannel
        if data_axis == 0:
            # `Conv2d` or `Dense` op weight
            shape_list = [-1] + [1] * len(tensor.shape[1:])
            scale = scale.reshape(shape_list)
            zero_point = zero_point.reshape(shape_list)
        elif data_axis == 1:
            # `DepthwiseConv2d` op weight
            shape_list = [1, -1] + [1] * len(tensor.shape[2:])
            scale = scale.reshape(shape_list)
            zero_point = zero_point.reshape(shape_list)
        else:
            raise ValueError("Unsupported data_axis({})".format(data_axis))

    t_scale = Tensor(scale)
    t_scale.asnumpy()
    t_zp = Tensor(zero_point)
    t_zp.asnumpy()
    t_scale = ms.ops.cast(t_scale, tensor.dtype)
    t_scale.asnumpy()
    t_zp = ms.ops.cast(t_zp, tensor.dtype)
    t_zp.asnumpy()
    quanted_data = tensor / t_scale
    quanted_data.asnumpy()
    quanted_data = quanted_data + t_zp
    quanted_data.asnumpy()
    quanted_data = ms.ops.round(quanted_data)
    quanted_data.asnumpy()
    quanted_data = ms.ops.clamp(quanted_data, quant_min, quant_max)
    quanted_data.asnumpy()
    quanted_data = ms.ops.cast(quanted_data, dtype)
    quanted_data.asnumpy()
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


def fold_batchnorm(weight, cell_quant):
    r"""
    Fold the batchnorm in `Conv2dBnFoldQuant` to weight.

    Calculate from `FakeQuantWithMinMax`'s Parameter or Fake quant primitive.

    Args:
        weight (numpy.ndarray): Weight of `cell_quant`.
        cell_quant (Cell): Object of `mindspore.nn.layer.Conv2dBnFoldQuant`.

    Returns:
        weight (numpy.ndarray): Folded weight.
        bias (numpy.ndarray): Folded bias.
    """
    variance = cell_quant.moving_variance.data.asnumpy()
    mean = cell_quant.moving_mean.data.asnumpy()
    gamma = cell_quant.gamma.data.asnumpy()
    beta = cell_quant.beta.data.asnumpy()
    epsilon = cell_quant.eps
    sigma = np.sqrt(variance + epsilon)

    if gamma.shape[0] == weight.shape[0]:
        # `Conv2d` or `Dense` op weight
        shape_list = [-1] + [1] * len(weight.shape[1:])
        gamma_ = gamma.reshape(shape_list)
        sigma_ = sigma.reshape(shape_list)
    elif gamma.shape[0] == weight.shape[1]:
        # `DepthwiseConv2d` op weight
        shape_list = [1, -1] + [1] * len(weight.shape[2:])
        gamma_ = gamma.reshape(shape_list)
        sigma_ = sigma.reshape(shape_list)
    else:
        raise ValueError("Unsupported weight shape({})".format(weight.shape))

    weight = weight * gamma_ / sigma_
    bias = beta - gamma * mean / sigma
    return weight, bias


def without_fold_batchnorm(weight, cell_quant):
    r"""
    Fold the batchnorm in `Conv2dBnWithoutFoldQuant` to weight.

    Calculate from `FakeQuantWithMinMax`'s Parameter or Fake quant primitive.

    Args:
        weight (numpy.ndarray): Weight of `cell_quant`.
        cell_quant (Cell): Object of `mindspore.nn.layer.Conv2dBnWithoutFoldQuant`.

    Returns:
        weight (numpy.ndarray): whihout folded weight.
        bias (numpy.ndarray): without folded bias.
    """
    variance = cell_quant.batchnorm.moving_variance.data.asnumpy()
    mean = cell_quant.batchnorm.moving_mean.data.asnumpy()
    gamma = cell_quant.batchnorm.gamma.data.asnumpy()
    beta = cell_quant.batchnorm.beta.data.asnumpy()
    epsilon = cell_quant.batchnorm.eps
    sigma = np.sqrt(variance + epsilon)

    if gamma.shape[0] == weight.shape[0]:
        # `Conv2d` or `Dense` op weight
        shape_list = [-1] + [1] * len(weight.shape[1:])
        gamma_ = gamma.reshape(shape_list)
        sigma_ = sigma.reshape(shape_list)
    elif gamma.shape[0] == weight.shape[1]:
        # `DepthwiseConv2d` op weight
        shape_list = [1, -1] + [1] * len(weight.shape[2:])
        gamma_ = gamma.reshape(shape_list)
        sigma_ = sigma.reshape(shape_list)
    else:
        raise ValueError("Unsupported weight shape({})".format(weight.shape))

    weight = weight * gamma_ / sigma_
    bias = beta - gamma * mean / sigma
    return weight, bias


def compute_kl_threshold(data, bitwidth):
    r"""
    Using KL-J Distance to calculate the clip threshold.

    Args:
        - **data** (NumpyArray) - Data observed to calculate the threshold for quantization,
        - **bitwidth** (QuantDtype) - The datatype of quantization.
    Outputs:
        Tensor with Shape 1. Threshold to calculate the data.
    """
    data_max = np.abs(data).max()
    if data_max < 1e-5:
        return 1e-5
    hist, bin_edges = np.histogram(np.abs(data), bins='sqrt', range=(0, data_max), density=True)
    # For the sake of high efficiency, we limit the maximum number of bins to 1024 in `sqrt` mode, If it exceeds the
    # largest size, turn to use the default bins config.
    largest_bin_size = 1024
    if hist.shape[0] > largest_bin_size:
        hist, bin_edges = np.histogram(np.abs(data), range=(0, data_max), density=True)
    hist = hist / np.sum(hist)
    cumsum = np.cumsum(hist)
    bit_pow_range = pow(2, int(bitwidth) - 1)
    threshold = []
    scaling_factor = []
    kl = []
    if bit_pow_range + 1 > len(bin_edges) - 1:
        th_layer_out = bin_edges[-1]
        return float(th_layer_out)
    for i in range(bit_pow_range + 1, len(bin_edges), 1):
        threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
        threshold = np.concatenate((threshold, [threshold_tmp]))
        scaling_factor_tmp = threshold_tmp / (bit_pow_range - 1)
        scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
        # forward interpolation
        cumsum_tmp = np.copy(cumsum)
        cumsum_tmp[(i - 1):] = 1
        fwd_x = np.linspace(0.0, 1.0, bit_pow_range)
        fwd_xp = np.linspace(0.0, 1.0, i)
        fwd_fp = cumsum_tmp[:i]
        forward_interp = np.interp(fwd_x, fwd_xp, fwd_fp)
        # backward interpolation
        bwd_x = np.linspace(0.0, 1.0, i)
        bwd_xp = np.linspace(0.0, 1.0, bit_pow_range)
        bwd_fp = forward_interp
        backward_interp = np.interp(bwd_x, bwd_xp, bwd_fp)
        cumsum_tmp[:i] = backward_interp
        kl_tmp = np.sum((cumsum - cumsum_tmp) * np.log2(cumsum / cumsum_tmp))  # Kullback-Leibler-J
        kl = np.concatenate((kl, [kl_tmp]))
    th_layer_out = threshold[np.argmin(kl)]
    threshold = float(th_layer_out)
    if threshold < 1e-5:
        threshold = 1e-5
    return threshold


def quant_bias_data(tensor: Tensor, scale, dtype=ms.dtype.int32):
    r"""
    Calculate int32 bias from fp32. the formula is defined as:

    .. math::
        int32 = round(bias / (scale_weight * scale_act))

    Args:
        tensor (Tensor): The bias tensor to be quanted
        scale (numpy.ndarray): The dimension of channel or 1.
        dtype(ms.dtype): default is dtype.int32

    Returns:
        quanted_bias (Tensor): quanted bias tensor
    """

    quant_scale = Tensor(np.squeeze(scale))
    quanted_data = ms.ops.round(tensor / quant_scale)
    quant_min = -2 ** 31
    quant_max = 2 ** 31 - 1
    quanted_data = ms.ops.clamp(quanted_data, quant_min, quant_max)
    quanted_data = ms.ops.cast(quanted_data, dtype)
    return quanted_data
