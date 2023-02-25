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
"""ST-Test for MaskedCell of Pruning algorithm."""
import pytest
import numpy as np
import mindspore
from mindspore import nn, Parameter
from mindspore_gs.pruner.ops import MaskedCell


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedcell_init():
    """
    Feature: mindspore_gs.prune.ops.MaskedCell.
    Description: instantiate a MaskedCell and check its field.
    Expectation: Except success.
    """
    cell = nn.Conv2d(5, 5, (3, 3))
    mask_cell = MaskedCell(cell, (3, 3), [3, 3])
    assert isinstance(mask_cell.handler, nn.Conv2d)
    assert isinstance(mask_cell.in_mask, Parameter)
    assert isinstance(mask_cell.out_mask, Parameter)
    assert mask_cell.in_mask.dtype == mindspore.int8
    assert not mask_cell.in_mask.requires_grad
    assert mask_cell.in_mask.shape == (3, 3)
    assert (mask_cell.in_mask.value().asnumpy() == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.int8)).all()
    assert mask_cell.out_mask.dtype == mindspore.int8
    assert not mask_cell.out_mask.requires_grad
    assert mask_cell.out_mask.shape == (3, 3)
    assert (mask_cell.out_mask.value().asnumpy() == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.int8)).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedcell_init_error():
    """
    Feature: mindspore_gs.prune.ops.MaskedCell.
    Description: Feed int type `handler`, Cell type `in_mask_shape`, Cell type `out_mask_shape`, negative
                 `in_mask_shape` and negative `out_mask_shape` to instantiate a MaskedCell.
    Expectation: Except raises.
    """
    cell = nn.Conv2d(5, 5, (3, 3))
    with pytest.raises(TypeError, match="The parameter `handler` must be isinstance of Cell, but got <class 'int'>."):
        MaskedCell(4, (3, 3), [3, 3])

    with pytest.raises(TypeError, match="The parameter `in_mask_shape` must be isinstance of \\(tuple, list, int\\), "
                                        "but got <class 'mindspore.nn.layer.conv.Conv2d'>."):
        MaskedCell(cell, cell, [3, 3])

    with pytest.raises(TypeError, match="The parameter `out_mask_shape` must be isinstance of \\(tuple, list, int\\), "
                                        "but got <class 'mindspore.nn.layer.conv.Conv2d'>."):
        MaskedCell(cell, [3, 3], cell)

    with pytest.raises(ValueError, match="For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                                         "must be positive integer, but got \\(-1,\\)"):
        MaskedCell(cell, -1, [3, 3])

    with pytest.raises(ValueError, match="For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                                         "must be positive integer, but got \\(-1,\\)"):
        MaskedCell(cell, [3, 3], -1)
