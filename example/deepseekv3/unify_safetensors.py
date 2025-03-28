# Copyright 2025 Huawei Technologies Co., Ltd
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
"""gene_strategy"""

import argparse
import os
import mindspore as ms

def trans_int8_to_int4(ckpt_path):
    '''trans_int8_to_int4'''
    for folder_name in os.listdir(ckpt_path):
        if folder_name.endswith(".safetensors"):
            folder_path = os.path.join(ckpt_path, folder_name)
            ori_param = ms.load_checkpoint(folder_path, format="safetensors")
            for name in sorted(ori_param.keys()):
                value = ori_param[name]
                if value.dtype == ms.dtype.int8:
                    print(f"convert {name} to int4.")
                    ori_param[name] = ms.Parameter(ms.Tensor(value.asnumpy(), ms.qint4x2), name=name,
                                                   requires_grad=False)
            ms.save_checkpoint(ori_param, folder_path, format="safetensors")
            print(f"save {folder_path} int4 safetensors success.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, type=str)
    parser.add_argument('--src_strategy_file', required=True, type=str)
    parser.add_argument('--dst_dir', required=True, type=str)
    parser.add_argument('--int4_trans', required=False, type=bool)
    args = parser.parse_args()
    print(f"args:---{args}")
    ms.unified_safetensors(args.src_dir, args.src_strategy_file, args.dst_dir)
    print("unified_safetensors success.")
    if args.int4_trans:
        print("start trans_int8_to_int4")
        trans_int8_to_int4(args.dst_dir)
        print("trans_int8_to_int4 success.")
