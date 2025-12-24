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
"""Quant network."""

from collections import OrderedDict
import os
import argparse
from functools import partial

import numpy as np
import mindspore as ms
from mindspore import dtype as msdtype

from datasets import load_dataset
from mindone.transformers import AutoProcessor

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.ptq_config import (PTQConfig,
                                         OutliersSuppressionType)
from mindspore_gs.ptq.models import AutoQuantForCausalLM


def get_args():
    """get args from command line."""
    parser = argparse.ArgumentParser(description="Quantization of Qwen3Vl series models.")

    parser.add_argument('--model_name',
                        '-m',
                        type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Path to the pretrained model path.")
    parser.add_argument('--quant_type',
                        '-q',
                        type=str,
                        default="a8w8_osl",
                        help="Quantization type, available: a8w8_smooth_quant/"
                             "a8w8_osl")
    parser.add_argument('--calib_dataset_path',
                        '-d',
                        type=str,
                        help="calibration dataset path")
    parser.add_argument('--output_path',
                        '-o',
                        type=str,
                        default="./quant_model",
                        help="Path to save the quantized model.")
    parser.add_argument('--backend',
                        '-b',
                        type=str,
                        default="ascend",
                        choices=["none", "ascend"],
                        help="Backend target, available: none/ascend. Default: ascend")
    args = parser.parse_args()
    return args


def create_ptq_config(quant_type):
    """Create PTQ config by quant_type."""
    if quant_type.lower() == "a8w8_smooth_quant":
        cfg = PTQConfig(weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression= OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['lm_head', "down_proj", "linear_fc2", "merger"])
        layer_policies = OrderedDict()
    elif quant_type.lower() == "a8w8_osl":
        cfg = PTQConfig(weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression= OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                        opname_blacklist=['lm_head', "down_proj", "linear_fc2", "merger"])
        layer_policies = OrderedDict()
    else:
        raise ValueError(f"Unsupported quant_type: {quant_type}.")
    return cfg, layer_policies


def preprocess_and_tokenizer(processor, example):
    """Preprocess dataset example for calibration"""
    image = example["image"]
    question = example['question']
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
    )
    return inputs

def convert_to_tensor(examples):
    """examples: dict[str, np.ndarray] -> dict[str, ms.Tensor]"""
    result = {}
    for k, v in examples.items():
        if isinstance(v, (np.ndarray, list)):
            tensor = ms.tensor(v)
            if tensor.dtype == ms.int64:
                tensor = tensor.astype(ms.int32)
            result[k] = tensor
        else:
            result[k] = v
    return result


def create_calib_datasets(dataset_path, processor, num_samples=200):
    """Create calibration datasets"""
    print(f"Load datasets {dataset_path}")
    ds = load_dataset(dataset_path, split="train")
    ds = ds.shuffle(seed=2025)
    ds = ds.select(range(min(num_samples, len(ds))))
    print("before Preprocess datasets...", ds, flush=True)
    ds = ds.map(partial(preprocess_and_tokenizer, processor), remove_columns=ds.column_names)
    ds.set_transform(convert_to_tensor)
    print("after Preprocess datasets...", ds, flush=True)
    return ds


# Convert backend string to BackendTarget enum
backend_map = {
    "none": BackendTarget.NONE,
    "ascend": BackendTarget.ASCEND
}


def quant_net(args):
    """Quant network with algorithm."""
    os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
    print("Create calibration datasets...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    ds = create_calib_datasets(args.calib_dataset_path, processor)

    print("Create PTQ config...", flush=True)
    cfg, layers_policies = create_ptq_config(args.quant_type)

    print("Create model...", flush=True)
    model = AutoQuantForCausalLM.from_pretrained(args.model_name)

    print("Start calibration...", flush=True)
    model.calibrate(cfg, layers_policies, datasets=ds)

    backend = backend_map.get(args.backend.lower(), BackendTarget.ASCEND)
    print(f"Save quantized model to {args.output_path} ...", flush=True)
    model.save_quantized(args.output_path, backend=backend)


if __name__ == "__main__":
    uargs = get_args()
    quant_net(uargs)
