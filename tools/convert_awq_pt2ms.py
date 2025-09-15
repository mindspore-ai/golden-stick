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


def _validate_paths(torch_ckpt_dir, ms_ckpt_file):
    """Validate input and output paths."""
    # Normalize and validate torch_ckpt_dir
    torch_ckpt_dir = os.path.normpath(torch_ckpt_dir)
    if not os.path.isabs(torch_ckpt_dir):
        raise ValueError("torch_ckpt_dir must be an absolute path")

    # Check for path traversal attempts
    if ".." in torch_ckpt_dir:
        raise ValueError("Path traversal detected in torch_ckpt_dir")

    # Validate torch_ckpt_dir exists and is a directory
    if not os.path.exists(torch_ckpt_dir):
        raise ValueError(f"Directory does not exist: {torch_ckpt_dir}")

    if not os.path.isdir(torch_ckpt_dir):
        raise ValueError(f"Path is not a directory: {torch_ckpt_dir}")

    # Normalize and validate ms_ckpt_file
    ms_ckpt_file = os.path.normpath(ms_ckpt_file)
    if not os.path.isabs(ms_ckpt_file):
        raise ValueError("ms_ckpt_file must be an absolute path")

    # Check for path traversal attempts
    if ".." in ms_ckpt_file:
        raise ValueError("Path traversal detected in ms_ckpt_file")

    # Validate output directory exists
    output_dir = os.path.dirname(ms_ckpt_file)
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory path is not a directory: {output_dir}")

    # Validate file extension
    allowed_extensions = ['.ckpt', '.safetensors']
    if not any(ms_ckpt_file.endswith(ext) for ext in allowed_extensions):
        raise ValueError(f"Invalid file extension. Allowed: {allowed_extensions}")

    return torch_ckpt_dir, ms_ckpt_file, output_dir


def _load_safetensors_files(torch_ckpt_dir):
    """Load all safetensors files from directory."""
    param_dict = {}
    for file_name in os.listdir(torch_ckpt_dir):
        if not file_name.endswith('.safetensors'):
            continue

        # Sanitize file name to prevent path traversal
        sanitized_file_name = file_name.replace("..", "").replace("/", "_").replace("\\", "_")

        # Construct and validate file path
        file_path = os.path.join(torch_ckpt_dir, sanitized_file_name)
        file_path = os.path.normpath(file_path)

        # Ensure the file path is within the torch_ckpt_dir
        if not file_path.startswith(torch_ckpt_dir):
            print(f"Skipping file with invalid path: {file_name}", flush=True)
            continue

        try:
            param_dict.update(
                load_checkpoint(
                    file_path,
                    format='safetensors')
            )
        # pylint: disable=W0703
        except Exception as e:
            print(
                f"Do not find huggingface checkpoint in '{torch_ckpt_dir}', "
                f"Error {e}.",
                flush=True
            )
            return None
    return param_dict


def _process_parameter(name, value):
    """Process a single parameter based on its type and name."""
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
    else:
        dtype = ms.float16

    return {'name': name, 'data': ms.Tensor(value, dtype=dtype)}


def convert_hf_ckpt(torch_ckpt_dir, ms_ckpt_file):
    """convert hf weight to ms."""
    torch_ckpt_dir, ms_ckpt_file, output_dir = _validate_paths(torch_ckpt_dir, ms_ckpt_file)

    print(f"Trying to convert huggingface checkpoint in '{torch_ckpt_dir}'.", flush=True)

    param_dict = _load_safetensors_files(torch_ckpt_dir)
    if param_dict is None:
        return False

    ckpt_list = []
    time_start = time.time()
    for name, value in param_dict.items():
        ckpt_list.append(_process_parameter(name, value))
    time_end = time.time()
    print(f'Trans takes {time_end - time_start} s')
    time_start = time.time()

    # Additional validation before saving
    if not ms_ckpt_file.startswith(output_dir):
        raise ValueError("Invalid output file path construction")

    # Extract and validate file format
    file_format = ms_ckpt_file.split('.')[-1]
    if file_format not in ['ckpt', 'safetensors']:
        raise ValueError(f"Invalid file format: {file_format}")

    ms.save_checkpoint(ckpt_list, ms_ckpt_file, format=file_format)
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
    _ = convert_hf_ckpt(torch_ckpt_dir=args.torch_ckpt_dir, ms_ckpt_file=args.mindspore_ckpt_file)
