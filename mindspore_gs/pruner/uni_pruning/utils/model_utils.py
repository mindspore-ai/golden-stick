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
"""Various functions that are used in UniPruning algorithm."""
import os
import json
import numpy as np
from mindspore import nn, float32 as ms_f32, Tensor, load, export, save_checkpoint
from mindspore_gs.common import logger


def get_model_size(groups, layer_mask):
    """
    Count the number of params in the model with respect to pruning mask.
    """
    size = 0
    for group in groups:
        for layer in group.ms_starts:
            mod = group.ms_starts[layer]
            if isinstance(mod, nn.Conv2d):
                shape = mod.weight.shape
                size += (shape[0] - layer_mask[layer]['cout']) * \
                    (shape[1] - layer_mask[layer]['cin']) * shape[2] * shape[3]
                if mod.bias is not None:
                    size += (shape[0] - layer_mask[layer]['cout'])
            if isinstance(mod, nn.Dense):
                shape = mod.weight.shape
                size += (shape[0] - layer_mask[layer]['cout']) * \
                    (shape[1] - layer_mask[layer]['cin'])
                if mod.bias is not None:
                    size += (shape[0] - layer_mask[layer]['cout'])
        for layer in group.ms_middles:
            mod = group.ms_middles[layer]
            if isinstance(mod, nn.BatchNorm2d):
                shape = mod.gamma.shape
                size += (shape[0] - layer_mask[layer]['cout']) * 4
    return size


def get_layer_type(layer):
    """
    Get layer type as a string.
    """
    if isinstance(layer, nn.Conv2d):
        return 'conv'
    if isinstance(layer, nn.Dense):
        return 'fc'
    if isinstance(layer, nn.BatchNorm2d):
        return 'bn'

    return 'unused_type'


def save_model_and_mask(net, output_path, exp_name, cur_step_num,
                        input_size, device_target, save_model=True, mask=None, export_air=False):
    """
    Save model as .MINDIR and .AIR, weights as .ckpt and mask as .json.
    """
    fake_input = np.random.uniform(0.0, 1.0, size=input_size).astype(np.float32)
    fake_input = Tensor(fake_input, ms_f32)
    save_checkpoint(net, f'{os.path.join(output_path, exp_name)}_epoch{cur_step_num}.ckpt')
    if save_model:
        logger.info(f'Exporting model {exp_name} into MINDIR')
        export(net, fake_input,
               file_name=f'{os.path.join(output_path, exp_name)}_epoch{cur_step_num}.mindir',
               file_format='MINDIR')
        if device_target == 'Ascend' and export_air:
            logger.info(f'Exporting model {exp_name} into AIR')
            export(net, fake_input,
                   file_name=f'{os.path.join(output_path, exp_name)}_epoch{cur_step_num}.air',
                   file_format='AIR')

    if mask is not None:
        logger.info('saving mask into JSON')
        save_mask = {key: val.tolist() for key, val in mask.items()}
        mask_save_path = f"{os.path.join(output_path, exp_name)}_epoch{cur_step_num}_mask.json"
        with open(mask_save_path, 'w+', encoding='utf8') as file_path:
            json.dump(save_mask, file_path, indent=3)


def load_model(output_path, exp_name, cur_step_num, input_size, dtype):
    """
    Load mindir model.
    """
    file_name = f'{os.path.join(output_path, exp_name)}_epoch{cur_step_num}.mindir'
    graph = load(file_name)
    net = nn.GraphCell(graph)
    fake_input = Tensor(np.ones(input_size).astype(np.float32), dtype)
    logger.info(f"Pruned MINDIR output shape {net(fake_input).shape}")


def find_ms_cell(groups, key):
    """
    Get layer as mindspore cell by name.
    """
    for group in groups:
        for start_key in group.ms_starts.keys():
            if start_key == key:
                return group.ms_starts[start_key]
        for middle_key in group.ms_middles.keys():
            if middle_key == key:
                return group.ms_middles[middle_key]
    return None
