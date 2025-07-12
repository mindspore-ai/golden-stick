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
"""Wrapper cell with mask."""
from typing import Union

import numpy as np
import mindspore
from mindspore import nn, Parameter, Tensor
from mindspore.common.initializer import initializer


class MaskedCell(nn.Cell):
    """
    Wrap cell with mask.

    Raises:
        TypeError: If `handler` is not Cell.
        TypeError: If `in_mask_shape` is not a tuple, a list nor an int.
        TypeError: If `out_mask_shape` is not a tuple, a list nor an int.
        ValueError: If `in_mask_shape` has non-positive number.
        ValueError: If `out_mask_shape` has non-positive number.
    """
    def __init__(self, handler: nn.Cell, in_mask_shape: Union[tuple, list, int],
                 out_mask_shape: Union[tuple, list, int]):
        super(MaskedCell, self).__init__()
        if not isinstance(handler, nn.Cell):
            raise TypeError(f'The parameter `handler` must be isinstance of Cell, but got {type(handler)}.')
        if not isinstance(in_mask_shape, (tuple, list, int)):
            raise TypeError(f'The parameter `in_mask_shape` must be isinstance of (tuple, list, int), but got '
                            f'{type(in_mask_shape)}.')
        if not isinstance(out_mask_shape, (tuple, list, int)):
            raise TypeError(f'The parameter `out_mask_shape` must be isinstance of (tuple, list, int), but got '
                            f'{type(out_mask_shape)}.')
        self.handler = handler
        self.in_mask = Parameter(initializer("ones", in_mask_shape, mindspore.int8), name='input_mask',
                                 requires_grad=False)
        self.out_mask = Parameter(initializer("ones", out_mask_shape, mindspore.int8), name='output_mask',
                                  requires_grad=False)

    def zeroing(self):
        """ Zero weight according to mask. """
        raise NotImplementedError("Please implement 'zeroing' method.")

    def prune(self) -> nn.Cell:
        """ Erase weight according to mask. """
        raise NotImplementedError("Please implement 'prune' method.")

    def set_in_mask(self, in_mask: np.array):
        """set_in_mask"""
        if in_mask.dtype != np.int8:
            raise ValueError(f'Data type of `in_mask` must be numpy.int8, but got {in_mask.dtype}.')
        if in_mask.shape != self.in_mask.asnumpy().shape:
            raise ValueError(f'Shape of `in_mask` must be equal to original in_mask:{self.in_mask.asnumpy().shape}, '
                             f'but got {in_mask.shape}.')
        self.in_mask.set_data(Tensor(in_mask), True)

    def set_out_mask(self, out_mask: np.array):
        """set_out_mask"""
        if out_mask.dtype != np.int8:
            raise ValueError(f'Data type of `out_mask` must be numpy.int8, but got {out_mask.dtype}.')
        if out_mask.shape != self.out_mask.asnumpy().shape:
            raise ValueError(f'Shape of `out_mask` must be equal to original '
                             f'out_mask:{self.out_mask.asnumpy().shape}, but got {out_mask.shape}.')
        self.out_mask.set_data(Tensor(out_mask), True)

    def construct(self, *inputs):
        """construct"""
        return self.handler(*inputs)
