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
"""Implementation of functions to compute pruning mask and zeroize weights according to mask."""
import numpy as np
from mindspore import Tensor, float32 as ms_f32
from .importance_criteria import get_medians, choose_channel_group_to_zero
from .model_utils import get_model_size, get_layer_type, find_ms_cell


def get_mask(groups, norms, step, filter_num_threshold, target_sparsity):
    """
    Iteratively get the pruning mask by:
        1. Getting medians from groupped channel importances.
        2. Measuring relative importance of the group.
        3. Choosing groups of channels with the highest relative importance measure
            throughout all layers until reaching target sparsity.
    """
    reached_target_size = False
    mask = {}
    target_size = get_model_size(groups, get_expanded_mask(groups, mask)) * target_sparsity
    while not reached_target_size:
        medians = get_medians(norms, step)
        zero_idx, block_idx = choose_channel_group_to_zero(medians, norms,
                                                           step, filter_num_threshold)
        for layer in norms[block_idx]:
            mask[layer] = np.append(mask[layer], zero_idx) if layer in mask else zero_idx
            norms[block_idx][layer][0][zero_idx] = -10.0
            norms[block_idx][layer][1] = np.argsort(norms[block_idx][layer][0])

        pruned_model_size = get_model_size(groups, get_expanded_mask(groups, mask))
        reached_target_size = pruned_model_size <= target_size

    return mask

def get_expanded_mask(groups, mask):
    """
    Expand mask, for each layer count the number of zeroed output/input channels,
        their indexes and layer type.
    """
    layer_mask = {}
    for group in groups:
        for key in group.ms_starts.keys():
            layer_mask[key] = {'cout': 0, 'cin': 0}
            layer_mask[key]['type'] = get_layer_type(group.ms_starts[key])
        for key in group.ms_middles.keys():
            layer_mask[key] = {'cout': 0}
            layer_mask[key]['type'] = get_layer_type(group.ms_middles[key])

    passed = {group: False for group in groups}
    for key in mask.keys():
        pruned_channels = len(mask[key])
        for group in groups:
            if key in group.ms_starts.keys() and not passed[group]:
                for st_layer in group.ms_starts:
                    layer_mask[st_layer]['cout'] = pruned_channels
                    layer_mask[st_layer]['cout_idx'] = mask[key]
                for m_layer in group.ms_middles:
                    layer_mask[m_layer]['cout'] = pruned_channels
                    layer_mask[m_layer]['cout_idx'] = mask[key]
                for end_layer in group.ms_ends:
                    layer_mask[end_layer]['cin'] = pruned_channels
                    layer_mask[end_layer]['cin_idx'] = mask[key]

                passed[group] = True
                break
    return layer_mask

def do_mask(groups, mask):
    """
    Zero layer channels according to pruning mask.
    """
    expanded_mask = get_expanded_mask(groups, mask)
    for key, layer_mask in expanded_mask.items():
        cell = find_ms_cell(groups, key)
        if layer_mask['type'] in ['conv', 'fc']:
            weight = cell.weight.asnumpy()
            if layer_mask['cin']:
                weight[:, layer_mask['cin_idx']] *= 0
            if layer_mask['cout']:
                weight[layer_mask['cout_idx']] *= 0
                if cell.has_bias:
                    bias = cell.bias.asnumpy()
                    bias[layer_mask['cout_idx']] *= 0
                    bias = Tensor(bias, ms_f32)
                    cell.bias.init_flag = False
                    cell.bias.init = None
                    cell.bias.assign_value(bias)
            shape = weight.shape
            weight = Tensor(np.ravel(weight).reshape(shape), ms_f32)
            cell.weight.init_flag = False
            cell.weight.init = None
            cell.weight.assign_value(weight)

        elif layer_mask['type'] == 'bn' and layer_mask['cout']:
            gamma = cell.gamma.asnumpy()
            gamma[layer_mask['cout_idx']] *= 0
            cell.gamma.init_flag = False
            cell.gamma.init = None
            cell.gamma.assign_value(Tensor(gamma))

            beta = cell.beta.asnumpy()
            beta[layer_mask['cout_idx']] *= 0
            cell.beta.init_flag = False
            cell.beta.init = None
            cell.beta.assign_value(Tensor(beta))

            moving_mean = cell.moving_mean.asnumpy()
            moving_mean[layer_mask['cout_idx']] *= 0
            cell.moving_mean.init_flag = False
            cell.moving_mean.init = None
            cell.moving_mean.assign_value(Tensor(moving_mean))

            moving_variance = cell.moving_variance.asnumpy()
            moving_variance[layer_mask['cout_idx']] *= 0
            cell.moving_variance.init_flag = False
            cell.moving_variance.init = None
            cell.moving_variance.assign_value(Tensor(moving_variance))
