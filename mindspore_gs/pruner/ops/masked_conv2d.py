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
"""Wrapper conv2d with mask."""

from mindspore import nn
from .masked_cell import MaskedCell


class MaskedConv2d(MaskedCell):
    """
    Wrap Conv2d with mask.

    Raises:
        TypeError: If `handler` is not nn.Conv2d.
        TypeError: If `in_mask_shape` is not a tuple, a list nor an int.
        TypeError: If `out_mask_shape` is not a tuple, a list nor an int.
        ValueError: If `in_mask_shape` has non-positive number.
        ValueError: If `out_mask_shape` has non-positive number.
    """
    def __init__(self, handler: nn.Conv2d, in_mask_shape=None, out_mask_shape=None):
        if not isinstance(handler, nn.Conv2d):
            raise TypeError(f'The parameter `handler` must be isinstance of nn.Conv2d, but got {type(handler)}.')
        if in_mask_shape is None:
            in_mask_shape = handler.in_channels
        if out_mask_shape is None:
            out_mask_shape = handler.out_channels
        super(MaskedConv2d, self).__init__(handler, in_mask_shape, out_mask_shape)

    def zeroing(self):
        """ Zero cout and cin dimension of weight according to mask. """
        raise NotImplementedError("Please implement 'zeroing' method.")

    def prune(self) -> nn.Cell:
        """ Erase cout and cin dimension of weight according to mask. """
        raise NotImplementedError("Please implement 'prune' method.")
