# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
Compare parameter of network after quantization.
Quickly find different in wrong ckpt and correct ckpt.
"""

import argparse
import numpy as np
import mindspore as ms

# layers include the quantized layer name
layers = ['w_qkv', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w_gate_hidden']


def update_param_name(param_dict):
    ckpt_list = {}
    for name in sorted(param_dict.keys()):
        if all([_ not in name for _ in layers]):
            continue
        name_list = name.split('.')
        new_name = '.'.join(name_list[:5]) + '.' + name_list[-1]
        ckpt_list.update({new_name: param_dict[name]})
    return ckpt_list


def statistic_error(corr_array, wrong_array):
    mean_error = np.mean(np.abs(corr_array - wrong_array))
    return mean_error


def compare_ckpt(correct_ckpt_file, wrong_ckpt_file):
    """compare_ckpt"""
    print(f"Trying to compare ckpt with '{correct_ckpt_file}' and '{wrong_ckpt_file}'.", flush=True)

    corr_param = ms.load_checkpoint(correct_ckpt_file)
    corr_param = update_param_name(corr_param)
    wrong_param = ms.load_checkpoint(wrong_ckpt_file)
    wrong_param = update_param_name(wrong_param)

    for name in sorted(corr_param.keys()):
        corr_val = corr_param[name].asnumpy()
        wrong_val = wrong_param[name].asnumpy()
        mean_error = statistic_error(corr_val, wrong_val)
        print(f'{name} mean error: {mean_error}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--correct_ckpt_file', default='correct.ckpt')
    parser.add_argument('--wrong_ckpt_file', default='wrong.ckpt')
    args = parser.parse_args()
    compare_ckpt(correct_ckpt_file=args.correct_ckpt_file, wrong_ckpt_file=args.wrong_ckpt_file)
