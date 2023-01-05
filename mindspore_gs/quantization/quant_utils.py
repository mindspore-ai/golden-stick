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

from mindspore.ops.operations import _quant_ops as Q
from mindspore.nn import Cell

__all__ = ["compute_kl_threshold", "fold_batchnorm", "cal_quantization_params", "get_quant_min_max"]


class LinearFakeQuantCell(Cell):
    """
    Fake quant layer with mindspore standard operator.
    """

    def __init__(self, quant_dtype, scale, zero_point, is_per_channel):
        super(LinearFakeQuantCell, self).__init__()
        self._quant_dtype = quant_dtype
        self._scale = scale
        self._zero_point = zero_point
        self._is_per_channel = is_per_channel
        self.fake_quant = Q.FakeQuantParam.linear_quant_param(quant_dtype=quant_dtype, scale=scale, zp=zero_point,
                                                              is_per_channel=is_per_channel)

    def extend_repr(self):
        """Display instance object as string."""
        s = 'quant_dtype={}, scale={}, zero_point={}, is_per_channel={}'.format(self._quant_dtype, self._scale,
                                                                                self._zero_point, self._is_per_channel)
        return s

    def construct(self, x):
        x = self.fake_quant(x)
        return x


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
    if len(input_min.shape) > 1:
        raise ValueError("input min and max shape should be one dim.")
    if (input_min > input_max).all():
        raise ValueError("input_min min should be less than input max.")
    if (input_max == input_min).all():
        return np.ones(input_min.shape), np.zeros(input_min.shape)

    # calculate scale
    if symmetric:
        input_max = np.maximum(-input_min, input_max)
        input_min = -input_max
    scale = (input_max - input_min) / (quant_max - quant_min)

    # calculate zero point
    zp_double = quant_min - input_min / scale
    zp = (np.floor(zp_double + 0.5)).astype(np.int)

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
