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
Quickly find different in new ckpt and correct ckpt.
"""

import argparse
import numpy as np
import mindspore as ms
from mindspore import dtype

# layers include the quantized layer name
layers = ['w_qkv', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w_gate_hidden']


def statistic_error(corr_array, new_array):
    """statistic_error"""
    mean_error = np.mean(np.abs(corr_array - new_array))
    sum_error = np.sum(np.abs(corr_array - new_array))
    return mean_error, sum_error


def compare_ckpt(correct_ckpt_file, new_ckpt_file):
    """compare_ckpt"""
    print(f"Trying to compare ckpt with '{correct_ckpt_file}' and '{new_ckpt_file}'.", flush=True)

    corr_param = ms.load_checkpoint(correct_ckpt_file)
    new_param = ms.load_checkpoint(new_ckpt_file)

    for name in sorted(corr_param.keys()):
        print(f"start compare {name}-----")
        if name not in new_param:
            print(f'{name} in corr_param but not in new_param', flush=True)
            print(f"compare {name} error!-----")
            continue
        corr_dtype = corr_param[name].dtype
        new_dtype = new_param[name].dtype
        corr_shape = corr_param[name].dtype
        new_shape = new_param[name].dtype
        if corr_dtype != new_dtype or corr_shape != new_shape:
            print(f'compared {name} corr_dtype:{corr_dtype}, new_dtype:{new_dtype},' \
                  f'corr_shape:{corr_shape}, new_shape:{new_shape} not equal.', flush=True)
            print(f"compare {name} error!-----")
            continue
        if corr_dtype == dtype.bfloat16:
            corr_val = corr_param[name].astype(np.float32).asnumpy()
            new_val = new_param[name].astype(np.float32).asnumpy()
        else:
            corr_val = corr_param[name].asnumpy()
            new_val = new_param[name].asnumpy()
        mean_error, sum_err = statistic_error(corr_val, new_val)
        if mean_error != 0 or sum_err != 0:
            print(f'{name} mean error: {mean_error}, sum error: {sum_err} ', flush=True)
            print(f"compare {name} error!-----")
        else:
            print(f"all equal, compare {name} success!-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--correct_ckpt_file', default='correct.ckpt')
    parser.add_argument('--new_ckpt_file', default='new.ckpt')
    args = parser.parse_args()
    compare_ckpt(correct_ckpt_file=args.correct_ckpt_file, new_ckpt_file=args.new_ckpt_file)
