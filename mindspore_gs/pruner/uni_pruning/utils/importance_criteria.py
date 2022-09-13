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
"""Functions related to UniPruning criterion"""
from collections import OrderedDict
import numpy as np


def get_medians(norms, step):
    """
    Count medians of each channel group.
    """
    medians = []
    for idx, norm_group in enumerate(norms):
        medians.append(OrderedDict.fromkeys(norm_group.keys()))
        for key in norm_group.keys():
            filter_norms, norms_idx = norm_group[key][0], norm_group[key][1]
            for i in range(0, len(norms_idx) - 1, step):
                median = np.median(filter_norms[norms_idx[i:i+step]])
                if i == 0:
                    medians[idx][key] = [median]
                else:
                    medians[idx][key].append(median)
    return medians

def get_channel_importances(groups, filter_num_thr):
    """
    For each layer count 2 arrays: one with channel importances, second with sorted channel indexes.
    """
    norms = []
    for group in groups:
        start = group.ms_starts
        norms.append({})
        for layer in start.keys():
            filters = start[layer].weight.asnumpy()
            if filters.shape[0] <= filter_num_thr:
                break
            filter_norms = np.linalg.norm(filters.reshape((filters.shape[0], -1)), 2, 1)
            norms_idx = np.argsort(filter_norms)
            norms[-1][layer] = [filter_norms, norms_idx]

    return norms

def choose_channel_group_to_zero(medians, norms, step, filter_num_threshold):
    """
    Choose the most unimportant group in all layers by relative importance criteria,
    for each layer's channel group:
        - Divide the biggest median in layer by the channel group median
        - The group with the highest value is the most unimportant
    """
    max_ratio = -1
    for idx, median_group in enumerate(medians):
        for key in median_group.keys():
            highest_median = median_group[key][-1]
            length = len(median_group[key])
            cnt = 0
            for i in range(length - 1):
                current_median = median_group[key][i]
                if current_median == 0:
                    chosen_key, block_idx, chosen_idx = key, idx, i
                    zero_idx = np.array(norms[block_idx][chosen_key][1]\
                        [chosen_idx * step : (chosen_idx + 1) * step])
                    return zero_idx, block_idx
                ratio = highest_median / (current_median + 1e-5)
                if ratio > max_ratio:
                    max_ratio = ratio
                    chosen_key, block_idx, chosen_idx = key, idx, i

                cnt += step
                if length * step - cnt <= filter_num_threshold:
                    break
    zero_idx = np.array(norms[block_idx][chosen_key][1]\
                        [chosen_idx * step : (chosen_idx + 1) * step])
    return zero_idx, block_idx
