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
import argparse
import os
import numpy as np

import mindspore as ms
from mindspore.communication import get_rank
from mindspore import save_checkpoint
from mindspore import dtype
from mindformers.trainer.utils import transform_and_load_checkpoint

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.ptq_config import (PTQConfig, PTQMode,
                                         OutliersSuppressionType,
                                         QuantGranularity)
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFParallelLlama2Helper
from mindspore_gs.common.utils import offload_network
from mindspore_gs.datasets import get_datasets
from mindspore_gs.datasets import create_boolq_dataset

def create_ds(network_helper, ds_path, ds_type, tokenizer_):
    """Create datasets."""
    bs_ = network_helper.get_spec('batch_size')
    seq_ = network_helper.get_spec('seq_length')
    max_decode_length = network_helper.get_spec('max_decode_length')
    ignore_token_id = network_helper.get_spec('ignore_token_id')
    ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer_, ignore_token_id,
                      1, False, n_samples=30)
    return ds


def create_cfg(quant_algo_, mode):
    """create_cfg"""
    if quant_algo_ == 'A8W8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
    elif quant_algo_ == 'A16W8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"])
    elif quant_algo_ == 'C8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=None,
                        kvcache_quant_dtype=dtype.int8)
    elif quant_algo_ == 'A8W8C8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8,
                        outliers_suppression=OutliersSuppressionType.SMOOTH,
                        kvcache_quant_dtype=dtype.int8)
    elif quant_algo_ == 'A16W8C8':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"],
                        kvcache_quant_dtype=dtype.int8)
    elif quant_algo_ == 'A8W8_Dynamic':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"],
                        act_quant_dtype=dtype.int8,
                        weight_quant_dtype=dtype.int8,
                        act_quant_granularity=QuantGranularity.PER_TOKEN,
                        outliers_suppression=OutliersSuppressionType.SMOOTH)
    elif quant_algo_ == 'C8_Dynamic':
        cfg = PTQConfig(mode=mode,
                        backend=BackendTarget.ASCEND,
                        opname_blacklist=["lm_head"],
                        weight_quant_dtype=None,
                        kvcache_quant_granularity=QuantGranularity.PER_TOKEN,
                        kvcache_quant_dtype=dtype.int8)
    else:
        raise ValueError(f"Unsupported quant_algo : {quant_algo_}")
    return cfg


def evaluate(network, ds_path, tokenizer, helper):
    """evaluate 'network' with dataset from 'dataset_path'."""
    top_k = helper.get_spec("top_k")
    top_p = helper.get_spec("top_p")
    do_sample = helper.get_spec("do_sample")
    batch_size = helper.get_spec("batch_size")
    seq_length = helper.get_spec("seq_length")
    ignore_token_id = helper.get_spec("ignore_token_id")
    pad_token_id = helper.get_spec("pad_token_id")
    ds = create_boolq_dataset(ds_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id,
                              n_samples=30)

    correct = 0
    data_count = 0
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        data_count += 1
        input_ids = ds_item['input_ids'].asnumpy()
        labels = ds_item['labels'].asnumpy()

        batch_valid_length = []
        for j in range(input_ids.shape[0]):
            batch_valid_length.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
        batch_valid_length = np.array(batch_valid_length)
        outputs = network.generate(input_ids, do_sample=do_sample, max_length=seq_length,
                                   top_p=top_p, top_k=top_k, max_new_tokens=5)
        output_ids = []
        for j in range(input_ids.shape[0]):
            output_ids.append(outputs[j][int(batch_valid_length[j]):])

        pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
        labels_str = tokenizer.decode(labels, skip_special_tokens=True)

        if labels_str[0].lower() in pres_str[0].lower():
            correct += 1
    ms.ms_memory_recycle()
    return correct / data_count


def quant_llama2(config_path_, ckpt_path, output_dir_, quant_algo_, ds_path):
    """PTQ quant to quant llama2"""
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['FORCE_EAGER'] = "true"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    cur_dir_ = os.path.dirname(os.path.abspath(__file__))
    config_path_ = os.path.join(cur_dir_, config_path_)
    vocab_file = os.path.join(cur_dir_, "../../../data/llama2-tokenizer.model")

    helper = MFParallelLlama2Helper(config_path_)
    helper.mf_config.load_checkpoint = os.path.join(cur_dir_, ckpt_path)
    helper.mf_config.output_dir = os.path.join(cur_dir_, output_dir_)
    helper.mf_config.processor.tokenizer.vocab_file = vocab_file
    device_id = int(os.environ.get('DEVICE_ID', '0'))
    helper.mf_config.context.device_id = device_id
    config = helper.mf_config

    network = helper.create_network()
    tokenizer = helper.create_tokenizer()

    cfg = create_cfg(quant_algo_, PTQMode.QUANTIZE)
    ptq = PTQ(config=cfg)
    # pylint: disable=W0212
    ptq._config.enable_deploy_fusion = False
    ds = create_ds(helper, ds_path, 'boolq', tokenizer)
    network = ptq.apply(network, helper, datasets=ds)
    network = ptq.convert(network)
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    save_path = os.path.join(config.output_dir, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    save_checkpoint(network.parameters_dict(), os.path.join(save_path, "quant.ckpt"),
                    choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    offload_network(network)


def eval_llama2(is_quant, config_path_, ckpt_path_, quant_algo_, ds_path):
    """eval llama2 by float ckpt and int ckpt"""
    ms.set_context(mode=0)
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
    os.environ.pop('FORCE_EAGER', None)
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    cur_dir_ = os.path.dirname(os.path.abspath(__file__))
    config_path_ = os.path.join(cur_dir_, config_path_)
    vocab_file = os.path.join(cur_dir_, "../../../data/llama2-tokenizer.model")

    helper = MFParallelLlama2Helper(config_path_)
    helper.mf_config.load_checkpoint = ""
    helper.mf_config.processor.tokenizer.vocab_file = vocab_file

    device_id = int(os.environ.get('DEVICE_ID', '0'))
    helper.mf_config.context.device_id = device_id
    config = helper.mf_config
    network = helper.create_network()

    if is_quant:
        cfg = create_cfg(quant_algo_, PTQMode.DEPLOY)
        ptq = PTQ(config=cfg)
        # pylint: disable=W0212
        ptq._config.enable_deploy_fusion = False
        network = ptq.apply(network)
        network = ptq.convert(network)

    config.load_checkpoint = os.path.join(cur_dir_, ckpt_path_)
    transform_and_load_checkpoint(config, None, network, None)
    tokenizer = helper.create_tokenizer()

    return evaluate(network, ds_path, tokenizer, helper)

def ptq_llama2_predict_2stage(config_path_, fp16_ckpt_path_, quant_ckpt_path_, quant_algo_, ds_path):
    """ptq_llama2_predict_2stage"""
    score_mapping = {
        "A8W8C8": 0.83,
        "A16W8C8": 0.86,
        "C8": 0.86,
        "A8W8": 0.86,
        "A16W8": 0.85,
        "A8W8_Dynamic": 0.86,
        "C8_Dynamic": 0.86,
    }

    quant_llama2(config_path_, fp16_ckpt_path_, quant_ckpt_path_, quant_algo_, ds_path)

    score = eval_llama2(True, config_path_, quant_ckpt_path_, quant_algo_, ds_path)

    assert score >= score_mapping[quant_algo_], f"Score {quant_algo_} is {score:.4f}, \
                                  which is lower than standard f{score_mapping[quant_algo_]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_parallel', '-m', type=int, default=1)
    parser.add_argument('--quant_algo', '-a', type=str, required=True)
    uargs = parser.parse_args()
    model_parallel = uargs.model_parallel
    quant_algo = uargs.quant_algo

    if model_parallel == 2:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(cur_dir, "../../../data/test_llama2/predict_parallelLlama2_13b.yaml")
        fp16_ckpt_2p_path = os.path.join(cur_dir, "/home/workspace/mindspore_ckpt/ckpt/llama2/llama2-13b-fp16-2p")
        quant_ckpt_path = os.path.join(cur_dir, f"output/parallelLlama2-quant-2p-{quant_algo}")
        dataset_path = os.path.join(cur_dir, f'../../../data/boolq-dataset/dev.jsonl')

        ptq_llama2_predict_2stage(config_path, fp16_ckpt_2p_path, quant_ckpt_path, quant_algo, dataset_path)
    else:
        raise ValueError(f"Unsupported model_parallel: {model_parallel}")
