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
"""Wrapper cell with mask."""

import numpy as np
from mindspore import nn, Tensor
from ..ops import MaskedConv2d, MaskedDense


def _prune_coci_weight(in_mask: np.array, out_mask: np.array, weight: np.array, cout, cin) -> np.array:
    """
    Prune weight in format [COut, CIn, ...] with in_mask and out_mask.
    """
    for co in range(cout - 1, -1, -1):
        if out_mask[co] == 1:
            continue
        weight = np.delete(weight, co, axis=0)
    for ci in range(cin - 1, -1, -1):
        if in_mask[ci] == 1:
            continue
        weight = np.delete(weight, ci, axis=1)
    return weight


def _prune_bias(out_mask: np.array, bias: np.array, cout) -> np.array:
    """
    Prune bias with out_mask.
    """
    for co in range(cout - 1, -1, -1):
        if out_mask[co] == 1:
            continue
        bias = np.delete(bias, co, axis=0)
    return bias


class UniPruningMaskedConv2d(MaskedConv2d):
    """
    Wrap Conv2d with mask.

    Raises:
        TypeError: If `handler` is not nn.Conv2d.
        TypeError: If `in_mask_shape` is not a tuple, a list nor an int.
        TypeError: If `out_mask_shape` is not a tuple, a list nor an int.
        ValueError: If `in_mask_shape` has non-positive number.
        ValueError: If `out_mask_shape` has non-positive number.
    """

    def zeroing(self):
        """ Zero cout and cin dimension of weight according to mask. """
        in_mask = self.in_mask.asnumpy()
        out_mask = self.out_mask.asnumpy()
        conv2d: nn.Conv2d = self.handler
        weight = conv2d.weight.asnumpy()
        # kernel of Conv2d in MindSpore is in [COut, CIn, KH, KW]
        for co in range(0, conv2d.out_channels):
            if out_mask[co] == 1:
                continue
            weight[co, :, :, :] *= 0
        for co in range(0, conv2d.out_channels):
            for ci in range(0, conv2d.in_channels):
                if in_mask[ci] == 1:
                    continue
                weight[co, ci, :, :] *= 0
        conv2d.weight.init_flag = False
        conv2d.weight.init = None
        conv2d.weight.assign_value(Tensor(weight))
        if not conv2d.has_bias:
            return
        bias = conv2d.bias
        for co in range(0, conv2d.out_channels):
            if out_mask[co] == 1:
                continue
            bias[co] = 0
        conv2d.bias.init_flag = False
        conv2d.bias.init = None
        conv2d.bias.assign_value(Tensor(bias))

    def prune(self) -> nn.Cell:
        """ Erase cout and cin dimension of weight according to mask. """
        in_mask = self.in_mask.asnumpy()
        out_mask = self.out_mask.asnumpy()
        conv2d: nn.Conv2d = self.handler
        weight = conv2d.weight.asnumpy()
        # kernel of Conv2d in MindSpore is in [COut, CIn, KH, KW]
        new_weight = _prune_coci_weight(in_mask, out_mask, weight, conv2d.out_channels, conv2d.in_channels)
        conv2d.weight.init_flag = False
        conv2d.weight.init = None
        conv2d.weight.assign_value(Tensor(new_weight))
        if conv2d.has_bias:
            bias = conv2d.bias.asnumpy()
            new_bias = _prune_bias(out_mask, bias, conv2d.out_channels)
            conv2d.bias.init_flag = False
            conv2d.bias.init = None
            conv2d.bias.assign_value(Tensor(new_bias))
        conv2d.out_channels = conv2d.weight.shape[0]
        conv2d.in_channels = conv2d.weight.shape[1]
        return conv2d


class UniPruningMaskedDense(MaskedDense):
    """
    Wrap Dense with mask.

    Raises:
        TypeError: If `handler` is not nn.Dense.
        TypeError: If `in_mask_shape` is not a tuple, a list nor an int.
        TypeError: If `out_mask_shape` is not a tuple, a list nor an int.
        ValueError: If `in_mask_shape` has non-positive number.
        ValueError: If `out_mask_shape` has non-positive number.
    """

    def zeroing(self):
        """ Zero cout and cin dimension of weight according to mask. """
        in_mask = self.in_mask.asnumpy()
        out_mask = self.out_mask.asnumpy()
        dense: nn.Dense = self.handler
        weight = dense.weight.asnumpy()
        # kernel of Dense in MindSpore is in [COut, CIn]
        for co in range(0, dense.out_channels):
            if out_mask[co] == 1:
                continue
            weight[co, :] *= 0
        for co in range(0, dense.out_channels):
            for ci in range(0, dense.in_channels):
                if in_mask[ci] == 1:
                    continue
                weight[co, ci] = 0
        dense.weight.init_flag = False
        dense.weight.init = None
        dense.weight.assign_value(Tensor(weight))
        if not dense.has_bias:
            return
        bias = dense.bias
        for co in range(0, dense.out_channels):
            if out_mask[co] == 1:
                continue
            bias[co] = 0
        dense.bias.init_flag = False
        dense.bias.init = None
        dense.bias.assign_value(Tensor(bias))

    def prune(self) -> nn.Cell:
        """ Erase cout and cin dimension of weight according to mask. """
        in_mask = self.in_mask.asnumpy()
        out_mask = self.out_mask.asnumpy()
        dense: nn.Dense = self.handler
        weight = dense.weight.asnumpy()
        # kernel of Dense in MindSpore is in [COut, CIn]
        new_weight = _prune_coci_weight(in_mask, out_mask, weight, dense.out_channels, dense.in_channels)
        dense.weight.init_flag = False
        dense.weight.init = None
        dense.weight.assign_value(Tensor(new_weight))
        if dense.has_bias:
            bias = dense.bias.asnumpy()
            new_bias = _prune_bias(out_mask, bias, dense.out_channels)
            dense.bias.init_flag = False
            dense.bias.init = None
            dense.bias.assign_value(Tensor(new_bias))
        dense.out_channels = dense.weight.shape[0]
        dense.in_channels = dense.weight.shape[1]
        return dense
