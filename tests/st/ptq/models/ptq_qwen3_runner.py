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
"""test interfaces of post training quant."""
from collections import OrderedDict

import argparse
import os
import sys
import shutil
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))

import mindspore as ms
from mindspore import dataset
from mindspore import dtype as msdtype
from mindformers import MindFormerConfig
from mindspore_gs.datasets import get_datasets
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.models import AutoModel
from transformers import AutoTokenizer


def create_ds(ds_path, ds_type, tokenizer_, mode, n_samples=-1):
    """Create datasets."""
    dataset.config.set_numa_enable(False)
    if not ds_path:
        raise ValueError(f"Please provide dataset_path.")
    if not ds_type:
        raise ValueError(f"Please provide dataset_type.")
    seq_ = 200
    max_decode_length = 100
    ignore_token_id = tokenizer_.pad_token_id
    ds = get_datasets(ds_type, ds_path, mode, 1, seq_, max_decode_length, tokenizer_, ignore_token_id,
                      1, False, n_samples=n_samples)
    return ds


def create_ptq_config(quant_type: str):
    """create_ptq"""
    if quant_type.lower() == 'a8w8':
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['output_layer', 'linear_fc2'])
        layer_policies = OrderedDict()
    else:
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
    return cfg, layer_policies


def evaluate(model, ds_path, tokenizer):
    """evaluate 'network' with dataset from 'dataset_path'."""
    ds = create_ds(ds_path, 'ceval', tokenizer, 'eval')
    pad_token_id = tokenizer.pad_token_id
    correct = 0
    data_count = 0
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        input_ids = ds_item['input_ids'].asnumpy()
        labels = ds_item['labels'].asnumpy()
        batch_valid_length = []
        for j in range(input_ids.shape[0]):
            batch_valid_length.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
        batch_valid_length = np.array(batch_valid_length)
        outputs = model.forward(input_ids, max_new_tokens=5)
        output_ids = []
        for j in range(input_ids.shape[0]):
            data_count += 1
            output_ids = outputs[j][int(batch_valid_length[j]):]
            pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
            labels_str = tokenizer.decode(labels[j], skip_special_tokens=True)
            question = tokenizer.decode(input_ids[j], skip_special_tokens=True)
            if labels_str.lower() in pres_str.lower():
                correct += 1
                print(f"question {data_count}: {question}\n predict: {pres_str} answer: {labels_str}. correct!",
                      flush=True)
            else:
                print(f"question {data_count}: {question}\n predict: {pres_str} answer: {labels_str}. not correct!",
                      flush=True)
    ms.ms_memory_recycle()
    return correct / data_count


def quant_qwen3(config_path_, output_dir_, quant_algo_, ds_path):
    """PTQ quant to quant qwen3"""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['ENFORCE_EAGER'] = "true"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    cur_dir_ = os.path.dirname(os.path.abspath(__file__))
    config_path_ = os.path.join(cur_dir_, config_path_)

    mfconfig = MindFormerConfig(config_path_)
    tokenizer = AutoTokenizer.from_pretrained(mfconfig.load_checkpoint)

    datasets = create_ds(ds_path, 'ceval', tokenizer, 'train', 50)
    model = AutoModel.from_pretrained(config_path_)
    cfg, layers_policy = create_ptq_config(quant_algo_)
    model.calibrate(cfg, layers_policy, datasets)
    ckpt_path = model.save_pretrained(output_dir_, quant_algo_, quant_algo_)

    os.environ.pop('ENFORCE_EAGER', None)
    return ckpt_path


def eval_qwen3(config_path_, ckpt_path_, ds_path, quant_algo_):
    """eval qwen3 by float ckpt and int ckpt"""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
    os.environ.pop('ENFORCE_EAGER', None)
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    mfconfig = MindFormerConfig(config_path_)
    tokenizer = AutoTokenizer.from_pretrained(mfconfig.load_checkpoint)
    model = AutoModel.from_pretrained(config_path_)
    cfg, layers_policy = create_ptq_config(quant_algo_)
    model.fake_quant(cfg, layers_policy, ckpt_path_)
    os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = "PagedAttention"
    res = evaluate(model, ds_path, tokenizer)
    return res


def datasets_accuracy(calibrate_config_path_, infer_config_path_, quant_ckpt_path_, quant_algo_, ds_path):
    """ptq_qwen3_predict_2stage"""
    score_mapping = {
        "A8W8": 0.41,
    }

    real_quant_ckpt_path = quant_qwen3(calibrate_config_path_, quant_ckpt_path_, quant_algo_, ds_path)

    score = eval_qwen3(infer_config_path_, real_quant_ckpt_path, ds_path, quant_algo_)
    print("="*50, flush=True)
    print(f"{quant_algo_} score {score}", flush=True)
    try:
        print(f"to rm dir: {quant_ckpt_path_}", flush=True)
        shutil.rmtree(quant_ckpt_path_)
    except (OSError, FileNotFoundError):
        pass
    error_str = f"Score {quant_algo_} is {score:.4f}, which is lower than standard f{score_mapping[quant_algo_]}"
    assert score >= score_mapping[quant_algo_], error_str
    print(f"Score of {quant_algo_} is {score}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_algo', '-a', type=str, required=True)
    uargs = parser.parse_args()
    quant_algo = uargs.quant_algo

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_config_path = os.path.join(cur_dir, "calibrate_qwen3.yaml")
    infer_config_path = os.path.join(cur_dir, "predict_qwen3.yaml")
    quant_ckpt_path = os.path.join(cur_dir, f"qwen3-quant-2p-{quant_algo}")
    dataset_path = os.path.join(cur_dir, '/nfs/dataset/workspace/mindspore_dataset/ceval/dev')
    datasets_accuracy(calibrate_config_path, infer_config_path, quant_ckpt_path, quant_algo, dataset_path)
