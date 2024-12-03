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
Convert llama weight.
Support huggingface format and Meta format.
"""

import os
import argparse
import time
import numpy as np
import mindspore as ms
from mindspore import load_checkpoint


def name_replace(name: str):
    """replace hf param name to ms."""
    name = name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
    name = name.replace('.self_attn.q_proj.', '.attention.wq._layer.')
    name = name.replace('.self_attn.k_proj.', '.attention.wk._layer.')
    name = name.replace('.self_attn.v_proj.', '.attention.wv._layer.')
    name = name.replace('.self_attn.o_proj.', '.attention.wo._layer.')
    name = name.replace('.mlp.gate_proj.', '.feed_forward.w1._layer.')
    name = name.replace('.mlp.down_proj.', '.feed_forward.w2._layer.')
    name = name.replace('.mlp.up_proj.', '.feed_forward.w3._layer.')
    name = name.replace('.input_layernorm.', '.attention_norm.')
    name = name.replace('.post_attention_layernorm.', '.ffn_norm.')
    name = name.replace('.norm.', '.norm_out.')
    name = name.replace('.scales', '.matmul.weight_scale')
    name = name.replace('.qzeros', '.matmul.weight_zp')
    name = name.replace('.qweight', '.weight')
    return name

def trans_int32_to_int4(np_data):
    """split int32 matrix to int4, i.e. use int8 matrix to save int4 data."""
    n, m = np_data.shape
    np_int4_data = np.zeros((n, m*8), dtype=np.int8)

    np_data = np_data.reshape(-1).astype(np.int32)
    split_data = ((np_data[:, None] >> np.arange(0, 29, 4, dtype=np.uint8)) & 0xF).astype(np.int8).reshape(n, -1)

    reordering_indices = np.array([0, 4, 1, 5, 2, 6, 3, 7])
    new_order = np.concatenate(np.array([reordering_indices + 8 * i for i in range(m)]))
    np_int4_data[:, :] = split_data[:, new_order]
    return np_int4_data

def trans_int4_to_qint4x2(np_data):
    """pack int4 data to int8"""
    np_data = np_data.astype(np.int8)
    np_data &= 0x000F
    np_data[::, 0::2] <<= 0
    np_data[::, 1::2] <<= 4
    np_int4_pack_data = np_data[::, 1::2] | np_data[::, 0::2]
    return np_int4_pack_data


def convert_hf_ckpt(torch_ckpt_dir, ms_ckpt_file, dtype=ms.float16):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{torch_ckpt_dir}'.", flush=True)

    try:
        param_dict = {}
        for file_name in os.listdir(torch_ckpt_dir):
            if file_name.endswith('.safetensors'):
                param_dict.update(
                    load_checkpoint(os.path.join(torch_ckpt_dir, file_name), format='safetensors'))
    # pylint: disable=W0703
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{torch_ckpt_dir}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    time_start = time.time()
    for name, value in param_dict.items():
        name = name_replace(name)
        value = value.asnumpy()
        print(f'\rprocessing parameter: {name} {value.shape}', end='', flush=True)
        if value.dtype == np.int32 and "._layer.weight" in name:
            value = trans_int32_to_int4(value)
            value = value - np.ones(value.shape, dtype=np.int8) * 8
            value = trans_int4_to_qint4x2(value)
            dtype = ms.qint4x2
        elif value.dtype == np.int32 and ".matmul.weight_zp" in name:
            value = trans_int32_to_int4(value)
            value = -1 * value + np.ones(value.shape, dtype=np.int8) * 8
            dtype = ms.float16
        elif value.dtype == np.float16:
            dtype = ms.float16
        ckpt_list.append({'name': name, 'data': ms.Tensor(value, dtype=dtype)})
    time_end = time.time()
    print(f'Trans takes {time_end - time_start} s')
    time_start = time.time()
    ms.save_checkpoint(ckpt_list, os.path.join(ms_ckpt_file), format=ms_ckpt_file.split('.')[-1])
    time_end = time.time()
    print(f'Save takes {time_end - time_start} s')
    print(f"\rConvert huggingface checkpoint finished, "
          f"the mindspore checkpoint is saved in '{ms_ckpt_file}'.", flush=True)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_ckpt_dir', default='./llama_model/llama-13b-hf/')
    parser.add_argument('--mindspore_ckpt_file', default='transform.ckpt')
    args = parser.parse_args()
    convert_hf_ckpt(torch_ckpt_dir=args.torch_ckpt_dir, ms_ckpt_file=args.mindspore_ckpt_file)
