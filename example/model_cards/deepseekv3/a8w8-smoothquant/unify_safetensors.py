"""Unify safetensors."""

import argparse
import json
import os
import re

from safetensors.numpy import load_file, save_file
from tqdm import tqdm

import numpy as np
import mindspore # pylint: disable=W0611


def get_args():
    """Get args."""
    parser = argparse.ArgumentParser(description="Unify safetensors for DeepSeekV3 quantization.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the quantized distributed model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the unified quantized model.",
    )
    parser.add_argument(
        "--output_file_prefix",
        type=str,
        required=True,
        help="Filename prefix to save the unified quantized model.",
    )
    parser.add_argument(
        "--rank_num",
        type=int,
        required=True,
        help="Number of ranks (partitions) in the distributed model.",
    )
    return parser.parse_args()


def sort_param_names(param_names):
    """Sort parameter names."""
    return sorted(param_names, key=lambda x: [int(i) if i.isdigit() else i for i in re.split(r"(\d+)", x)])


def open_files(distributed_safetensors, rank_num):
    """Open files."""
    params = []
    files = [f"{distributed_safetensors}/rank_{i}/quant.safetensors" for i in range(rank_num)]
    for file in tqdm(files, desc="Loading files"):
        params.append(load_file(file))
    param_names = sort_param_names(params[0].keys())
    return params, param_names


def get_parallel_type(param_name):
    """Get parallel type."""
    type_map = {
        ".down_proj.": "row",
        ".embed_tokens.": "column",
        ".gate.": "no_parallel",
        ".gate_proj.": "column",
        ".input_layernorm.": "no_parallel",
        ".kv_a_layernorm.": "no_parallel",
        ".kv_a_proj_with_mqa.": "no_parallel",
        ".kv_b_proj.": "column",
        "lm_head.": "column",
        ".norm.": "no_parallel",
        ".o_proj.": "row",
        ".post_attention_layernorm.": "no_parallel",
        ".q_a_layernorm.": "no_parallel",
        ".q_a_proj.": "no_parallel",
        ".q_b_proj.": "column",
        ".up_proj.": "column",
    }
    for k, v in type_map.items():
        if k in param_name:
            return v
    raise ValueError(f"Unsupported param name: {param_name}")


def get_param_axis(param_name, axis_name):
    """Get parameter axis."""
    axis_map = {
        "deq_scale": ("oc",),
        "input_offset": ("ic",),
        "input_scale": ("ic",),
        "quant_bias": ("oc",),
        "smooth_scale": ("ic",),
        "weight": ("oc", "ic"),
        "weight_offset": ("oc",),
        "weight_scale": ("oc",),
    }
    axis = axis_map[param_name.split(".")[-1]]
    if axis_name not in axis:
        return -1
    return axis.index(axis_name)


def load_parallel_param(shard_slices, shard_axis):
    """Load parallel parameter from shard slices.
    Step 1. Get full param shape by multiplying shard shape by rank_num along shard_axis.
    Step 2. Create an empty array of full param shape.
    Step 3. For each rank, load the shard and place it in the correct position in the full param array.
    """
    rank_num = len(shard_slices)
    shard_shape = shard_slices[0].shape
    full_shape = list(shard_shape)
    full_shape[shard_axis] *= rank_num
    dtype = shard_slices[0].dtype
    full_param = np.empty(full_shape, dtype=dtype)
    for i, shard in enumerate(shard_slices):
        if shard.shape != shard_shape:
            raise ValueError(
                f"Shard shape mismatch on rank {i} (shard axis is {shard_axis}): "
                f"expected {shard_shape}, got {shard.shape}"
            )
        shard_size = shard.shape[shard_axis]
        if shard_axis == 0:
            full_param[i * shard_size : (i + 1) * shard_size] = shard
        elif shard_axis == 1:
            full_param[:, i * shard_size : (i + 1) * shard_size] = shard
        else:
            raise ValueError(f"Unsupported shard axis: {shard_axis}")
    return full_param


def load_one_param(params, param_name):
    """Load one parameter."""
    shard_axis = -1

    parallel_type = get_parallel_type(param_name)
    if parallel_type == "row":
        shard_axis = get_param_axis(param_name, "ic")
    elif parallel_type == "column":
        shard_axis = get_param_axis(param_name, "oc")

    shards = [i.pop(param_name) for i in params]
    if shard_axis != -1:
        return load_parallel_param(shards, shard_axis)

    return shards[0]


def unify_and_save_safetensors(params, param_names, output_dir, output_file_prefix):
    """Unify safetensors."""
    param_dict = {}

    unified_state_dict = {}
    max_file_size = 4 * 1024**3  # 4GB
    current_size = 0
    file_idx = 1

    def save_to_file():
        filename = f"{output_file_prefix}_{file_idx:03d}.safetensors"
        filepath = f"{output_dir}/{filename}"
        save_file(unified_state_dict, filepath)
        param_dict.update((k, filename) for k in unified_state_dict)

    for param_name in tqdm(param_names, desc="Processing parameters"):
        param = load_one_param(params, param_name)
        param_bytes = param.nbytes
        if current_size + param_bytes > max_file_size and unified_state_dict:
            save_to_file()
            unified_state_dict = {}
            current_size = 0
            file_idx += 1
        unified_state_dict[param_name] = param
        current_size += param_bytes
    if unified_state_dict:
        save_to_file()

    return param_dict


def save_param_dict(param_dict, output_dir, output_file_prefix):
    """Save parameter dictionary."""
    filename = f"{output_file_prefix}.safetensors.index.json"
    filepath = f"{output_dir}/{filename}"
    save_obj = {
        "metadata": {},
        "weight_map": param_dict,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(save_obj, f, indent=2)


def main():
    """Main function."""
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    params, param_names = open_files(args.input_dir, args.rank_num)
    param_dict = unify_and_save_safetensors(params, param_names, args.output_dir, args.output_file_prefix)
    save_param_dict(param_dict, args.output_dir, args.output_file_prefix)


if __name__ == "__main__":
    main()
