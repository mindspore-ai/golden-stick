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
"""Implementation of pruning function that prunes network weights according to pruning mask"""
import numpy as np
from mindspore import Tensor
from mindspore.nn import Conv2d
from mindspore.ops import Conv2D
from .model_utils import find_ms_cell
from .mask import get_expanded_mask


def update_channel_out(cell, new_val):
    """
    Reinitialise mindspore.ops.Conv2d of nn.Conv2d after pruning
    """
    cell.out_channels = new_val
    if isinstance(cell, Conv2d):
        cell.conv2d = Conv2D(
            out_channel=new_val,
            kernel_size=cell.conv2d.kernel_size,
            mode=cell.conv2d.mode,
            pad_mode=cell.conv2d.pad_mode,
            pad=cell.conv2d.padding,
            stride=cell.conv2d.stride,
            dilation=cell.conv2d.dilation,
            group=cell.conv2d.group,
            data_format=cell.conv2d.format
        )


def prune_net(groups, mask):
    """
    Pruning function of network according to mask.
    """
    expanded_mask = get_expanded_mask(groups, mask)
    for key, layer_mask in expanded_mask.items():
        cell = find_ms_cell(groups, key)
        if layer_mask['type'] in ['conv', 'fc'] and \
        layer_mask['cin'] + layer_mask['cout']:
            weight = cell.weight.asnumpy()
            if layer_mask['cin']:
                weight = np.delete(weight, layer_mask['cin_idx'], axis=1)
            if layer_mask['cout']:
                weight = np.delete(weight, layer_mask['cout_idx'], axis=0)
                cell.out_channels = weight.shape[0]
                update_channel_out(cell, weight.shape[0])
                if cell.has_bias:
                    bias = cell.bias.asnumpy()
                    bias = np.delete(bias, layer_mask['cout_idx'])
                    cell.bias.init_flag = False
                    cell.bias.init = None
                    cell.bias.assign_value(Tensor.from_numpy(bias))

            shape = weight.shape
            cell.weight.init_flag = False
            cell.weight.init = None
            cell.weight.assign_value(Tensor.from_numpy(np.ravel(weight).reshape(shape)))

        elif layer_mask['type'] == 'bn' and layer_mask['cout']:
            gamma = cell.gamma.asnumpy()
            gamma = np.delete(gamma, layer_mask['cout_idx'])
            cell.gamma.init_flag = False
            cell.gamma.init = None
            cell.gamma.assign_value(Tensor.from_numpy(gamma))
            beta = cell.beta.asnumpy()
            beta = np.delete(beta, layer_mask['cout_idx'])
            cell.beta.init_flag = False
            cell.beta.init = None
            cell.beta.assign_value(Tensor.from_numpy(beta))
            moving_mean = cell.moving_mean.asnumpy()
            moving_mean = np.delete(moving_mean, layer_mask['cout_idx'])
            cell.moving_mean.init_flag = False
            cell.moving_mean.init = None
            cell.moving_mean.assign_value(Tensor.from_numpy(moving_mean))
            moving_variance = cell.moving_variance.asnumpy()
            moving_variance = np.delete(moving_variance, layer_mask['cout_idx'])
            cell.moving_variance.init_flag = False
            cell.moving_variance.init = None
            cell.moving_variance.assign_value(Tensor.from_numpy(moving_variance))
