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
"""ST-Test for unipruning_masked_layer of Pruning algorithm."""
import pytest
import numpy as np
import mindspore
from mindspore import nn, Parameter
from mindspore_gs.pruner.uni_pruning.unipruning_masked_layer import UniPruningMaskedDense, UniPruningMaskedConv2d


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskedconv2d_init():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedConv2d.
    Description: instantiate a UniPruningMaskedConv2d and check its field.
    Expectation: Except success.
    """
    cell = nn.Conv2d(6, 5, (3, 3))
    mask_conv2d = UniPruningMaskedConv2d(cell, (3, 3), [3, 3])
    assert isinstance(mask_conv2d.handler, nn.Conv2d)
    assert isinstance(mask_conv2d.in_mask, Parameter)
    assert isinstance(mask_conv2d.out_mask, Parameter)
    assert mask_conv2d.in_mask.dtype == mindspore.int8
    assert not mask_conv2d.in_mask.requires_grad
    assert mask_conv2d.in_mask.shape == (3, 3)
    assert (mask_conv2d.in_mask.value().asnumpy() == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.int8)).all()
    assert mask_conv2d.out_mask.dtype == mindspore.int8
    assert not mask_conv2d.out_mask.requires_grad
    assert mask_conv2d.out_mask.shape == (3, 3)
    assert (mask_conv2d.out_mask.value().asnumpy() == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.int8)).all()

    mask_conv2d = UniPruningMaskedConv2d(cell)
    assert isinstance(mask_conv2d.handler, nn.Conv2d)
    assert isinstance(mask_conv2d.in_mask, Parameter)
    assert isinstance(mask_conv2d.out_mask, Parameter)
    assert mask_conv2d.in_mask.dtype == mindspore.int8
    assert not mask_conv2d.in_mask.requires_grad
    assert mask_conv2d.in_mask.shape == (6,)
    assert (mask_conv2d.in_mask.value().asnumpy() == np.array([1, 1, 1, 1, 1, 1], np.int8)).all()
    assert mask_conv2d.out_mask.dtype == mindspore.int8
    assert not mask_conv2d.out_mask.requires_grad
    assert mask_conv2d.out_mask.shape == (5,)
    assert (mask_conv2d.out_mask.value().asnumpy() == np.array([1, 1, 1, 1, 1], np.int8)).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskedconv2d_init_error():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedConv2d.
    Description: Feed int type `handler`, Cell type `in_mask_shape`, Cell type `out_mask_shape`, negative
                 `in_mask_shape` and negative `out_mask_shape` to instantiate a MaskedConv2d.
    Expectation: Except raises.
    """
    cell = nn.Conv2d(5, 5, (3, 3))
    with pytest.raises(TypeError, match="The parameter `handler` must be isinstance of nn.Conv2d, but got "
                                        "<class 'int'>."):
        UniPruningMaskedConv2d(4, (3, 3), [3, 3])

    with pytest.raises(TypeError, match="The parameter `handler` must be isinstance of nn.Conv2d, but got "
                                        "<class 'mindspore.nn.layer.basic.Dense'>."):
        UniPruningMaskedConv2d(nn.Dense(5, 5), (3, 3), [3, 3])

    with pytest.raises(TypeError, match="The parameter `in_mask_shape` must be isinstance of \\(tuple, list, int\\), "
                                        "but got <class 'mindspore.nn.layer.conv.Conv2d'>."):
        UniPruningMaskedConv2d(cell, cell, [3, 3])

    with pytest.raises(TypeError, match="The parameter `out_mask_shape` must be isinstance of \\(tuple, list, int\\), "
                                        "but got <class 'mindspore.nn.layer.conv.Conv2d'>."):
        UniPruningMaskedConv2d(cell, [3, 3], cell)

    with pytest.raises(ValueError, match="For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                                         "must be positive integer, but got \\(-1,\\)"):
        UniPruningMaskedConv2d(cell, -1, [3, 3])

    with pytest.raises(ValueError, match="For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                                         "must be positive integer, but got \\(-1,\\)"):
        UniPruningMaskedConv2d(cell, [3, 3], -1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskedconv2d_set_in_mask():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedConv2d.
    Description: invoke set_in_mask check result.
    Expectation: Except success.
    """
    cell = nn.Conv2d(6, 5, (3, 3))
    mask_conv2d = UniPruningMaskedConv2d(cell)
    with pytest.raises(ValueError, match="Data type of `in_mask` must be numpy.int8, but got uint8."):
        mask_conv2d.set_in_mask(np.array([1, 0, 1, 0, 1, 0], np.uint8))
    with pytest.raises(ValueError, match="Shape of `in_mask` must be equal to original in_mask:\\(6,\\), but got "
                                         "\\(5,\\)."):
        mask_conv2d.set_in_mask(np.array([1, 0, 1, 0, 1], np.int8))

    mask_conv2d.set_in_mask(np.array([1, 0, 1, 0, 1, 0], np.int8))
    assert mask_conv2d.in_mask.dtype == mindspore.int8
    assert not mask_conv2d.in_mask.requires_grad
    assert mask_conv2d.in_mask.shape == (6,)
    assert (mask_conv2d.in_mask.value().asnumpy() == np.array([1, 0, 1, 0, 1, 0], np.int8)).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskedconv2d_set_out_mask():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedConv2d.
    Description: invoke set_out_mask check result.
    Expectation: Except success.
    """
    cell = nn.Conv2d(6, 5, (3, 3))
    mask_conv2d = UniPruningMaskedConv2d(cell)
    with pytest.raises(ValueError, match="Data type of `out_mask` must be numpy.int8, but got uint8."):
        mask_conv2d.set_out_mask(np.array([1, 0, 1, 0, 1], np.uint8))
    with pytest.raises(ValueError, match="Shape of `out_mask` must be equal to original out_mask:\\(5,\\), but got "
                                         "\\(6,\\)."):
        mask_conv2d.set_out_mask(np.array([1, 0, 1, 0, 1, 0], np.int8))

    mask_conv2d.set_out_mask(np.array([1, 0, 1, 0, 1], np.int8))
    assert mask_conv2d.out_mask.dtype == mindspore.int8
    assert not mask_conv2d.out_mask.requires_grad
    assert mask_conv2d.out_mask.shape == (5,)
    assert (mask_conv2d.out_mask.value().asnumpy() == np.array([1, 0, 1, 0, 1], np.int8)).all()


@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskedconv2d_zeroing():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedConv2d.
    Description: invoke zeroing check result.
    Expectation: Except success.
    """
    cell = nn.Conv2d(3, 2, (2, 2), has_bias=True)
    mask_conv2d = UniPruningMaskedConv2d(cell)
    mask_conv2d.set_in_mask(np.array([1, 0, 1], np.int8))
    mask_conv2d.set_out_mask(np.array([1, 0], np.int8))
    mask_conv2d.zeroing()
    assert mask_conv2d.handler.weight.shape == (2, 3, 2, 2)
    new_weight_data = mask_conv2d.handler.weight.value().asnumpy()
    assert (new_weight_data[1, :, :, :] == np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]])).all()
    assert (new_weight_data[:, 1, :, :] == np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])).all()

    assert mask_conv2d.handler.bias.shape == (2,)
    new_bias_data = mask_conv2d.handler.bias.value().asnumpy()
    assert new_bias_data[1] == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskedconv2d_prune():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedConv2d.
    Description: invoke prune check result.
    Expectation: Except success.
    """
    cell = nn.Conv2d(3, 2, (2, 2), has_bias=True)
    mask_conv2d = UniPruningMaskedConv2d(cell)
    mask_conv2d.set_in_mask(np.array([1, 0, 1], np.int8))
    mask_conv2d.set_out_mask(np.array([1, 0], np.int8))
    mask_conv2d.prune()
    assert mask_conv2d.handler.weight.shape == (1, 2, 2, 2)
    assert mask_conv2d.handler.bias.shape == (1,)
    assert mask_conv2d.handler.in_channels == 2
    assert mask_conv2d.handler.out_channels == 1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskeddense_init():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedDense.
    Description: instantiate a UniPruningMaskedDense and check its field.
    Expectation: Except success.
    """
    cell = nn.Dense(6, 5)
    mask_dense = UniPruningMaskedDense(cell, (3, 3), [3, 3])
    assert isinstance(mask_dense.handler, nn.Dense)
    assert isinstance(mask_dense.in_mask, Parameter)
    assert isinstance(mask_dense.out_mask, Parameter)
    assert mask_dense.in_mask.dtype == mindspore.int8
    assert not mask_dense.in_mask.requires_grad
    assert mask_dense.in_mask.shape == (3, 3)
    assert (mask_dense.in_mask.value().asnumpy() == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.int8)).all()
    assert mask_dense.out_mask.dtype == mindspore.int8
    assert not mask_dense.out_mask.requires_grad
    assert mask_dense.out_mask.shape == (3, 3)
    assert (mask_dense.out_mask.value().asnumpy() == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.int8)).all()

    mask_dense = UniPruningMaskedDense(cell)
    assert isinstance(mask_dense.handler, nn.Dense)
    assert isinstance(mask_dense.in_mask, Parameter)
    assert isinstance(mask_dense.out_mask, Parameter)
    assert mask_dense.in_mask.dtype == mindspore.int8
    assert not mask_dense.in_mask.requires_grad
    assert mask_dense.in_mask.shape == (6,)
    assert (mask_dense.in_mask.value().asnumpy() == np.array([1, 1, 1, 1, 1, 1], np.int8)).all()
    assert mask_dense.out_mask.dtype == mindspore.int8
    assert not mask_dense.out_mask.requires_grad
    assert mask_dense.out_mask.shape == (5,)
    assert (mask_dense.out_mask.value().asnumpy() == np.array([1, 1, 1, 1, 1], np.int8)).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskeddense_init_error():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedDense.
    Description: Feed int type `handler`, Cell type `in_mask_shape`, Cell type `out_mask_shape`, negative
                 `in_mask_shape` and negative `out_mask_shape` to instantiate a MaskedConv2d.
    Expectation: Except raises.
    """
    cell = nn.Dense(5, 5)
    with pytest.raises(TypeError, match="The parameter `handler` must be isinstance of nn.Dense, but got "
                                        "<class 'int'>."):
        UniPruningMaskedDense(4, (3, 3), [3, 3])

    with pytest.raises(TypeError, match="The parameter `handler` must be isinstance of nn.Dense, but got "
                                        "<class 'mindspore.nn.layer.conv.Conv2d'>."):
        UniPruningMaskedDense(nn.Conv2d(5, 5, (3, 3)), (3, 3), [3, 3])

    with pytest.raises(TypeError, match="The parameter `in_mask_shape` must be isinstance of \\(tuple, list, int\\), "
                                        "but got <class 'mindspore.nn.layer.basic.Dense'>."):
        UniPruningMaskedDense(cell, cell, [3, 3])

    with pytest.raises(TypeError, match="The parameter `out_mask_shape` must be isinstance of \\(tuple, list, int\\), "
                                        "but got <class 'mindspore.nn.layer.basic.Dense'>."):
        UniPruningMaskedDense(cell, [3, 3], cell)

    with pytest.raises(ValueError, match="For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                                         "must be positive integer, but got \\(-1,\\)"):
        UniPruningMaskedDense(cell, -1, [3, 3])

    with pytest.raises(ValueError, match="For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                                         "must be positive integer, but got \\(-1,\\)"):
        UniPruningMaskedDense(cell, [3, 3], -1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskeddense_set_in_mask():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedDense.
    Description: invoke set_in_mask check result.
    Expectation: Except success.
    """
    cell = nn.Dense(6, 5)
    mask_dense = UniPruningMaskedDense(cell)
    with pytest.raises(ValueError, match="Data type of `in_mask` must be numpy.int8, but got uint8."):
        mask_dense.set_in_mask(np.array([1, 0, 1, 0, 1, 0], np.uint8))
    with pytest.raises(ValueError, match="Shape of `in_mask` must be equal to original in_mask:\\(6,\\), but got "
                                         "\\(5,\\)."):
        mask_dense.set_in_mask(np.array([1, 0, 1, 0, 1], np.int8))

    mask_dense.set_in_mask(np.array([1, 0, 1, 0, 1, 0], np.int8))
    assert mask_dense.in_mask.dtype == mindspore.int8
    assert not mask_dense.in_mask.requires_grad
    assert mask_dense.in_mask.shape == (6,)
    assert (mask_dense.in_mask.value().asnumpy() == np.array([1, 0, 1, 0, 1, 0], np.int8)).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskeddense_set_out_mask():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedDense.
    Description: invoke set_out_mask check result.
    Expectation: Except success.
    """
    cell = nn.Dense(6, 5)
    mask_dense = UniPruningMaskedDense(cell)
    with pytest.raises(ValueError, match="Data type of `out_mask` must be numpy.int8, but got uint8."):
        mask_dense.set_out_mask(np.array([1, 0, 1, 0, 1], np.uint8))
    with pytest.raises(ValueError, match="Shape of `out_mask` must be equal to original out_mask:\\(5,\\), but got "
                                         "\\(6,\\)."):
        mask_dense.set_out_mask(np.array([1, 0, 1, 0, 1, 0], np.int8))

    mask_dense.set_out_mask(np.array([1, 0, 1, 0, 1], np.int8))
    assert mask_dense.out_mask.dtype == mindspore.int8
    assert not mask_dense.out_mask.requires_grad
    assert mask_dense.out_mask.shape == (5,)
    assert (mask_dense.out_mask.value().asnumpy() == np.array([1, 0, 1, 0, 1], np.int8)).all()


@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskeddense_zeroing():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedDense.
    Description: invoke zeroing check result.
    Expectation: Except success.
    """
    cell = nn.Dense(3, 2)
    mask_dense = UniPruningMaskedDense(cell)
    mask_dense.set_in_mask(np.array([1, 0, 1], np.int8))
    mask_dense.set_out_mask(np.array([1, 0], np.int8))
    mask_dense.zeroing()
    assert mask_dense.handler.weight.shape == (2, 3)
    new_weight_data = mask_dense.handler.weight.value().asnumpy()
    assert (new_weight_data[1, :] == np.array([[0, 0, 0]])).all()
    assert (new_weight_data[:, 1] == np.array([0, 0])).all()

    assert mask_dense.handler.bias.shape == (2,)
    new_bias_data = mask_dense.handler.bias.value().asnumpy()
    assert new_bias_data[1] == 0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_unipruningmaskeddense_prune():
    """
    Feature: mindspore_gs.prune.unipruning.unipruning_masked_layer.UniPruningMaskedDense.
    Description: invoke prune check result.
    Expectation: Except success.
    """
    cell = nn.Dense(3, 2)
    mask_dense = UniPruningMaskedDense(cell)
    mask_dense.set_in_mask(np.array([1, 0, 1], np.int8))
    mask_dense.set_out_mask(np.array([1, 0], np.int8))
    mask_dense.prune()
    assert mask_dense.handler.weight.shape == (1, 2)
    assert mask_dense.handler.bias.shape == (1,)
    assert mask_dense.handler.in_channels == 2
    assert mask_dense.handler.out_channels == 1
