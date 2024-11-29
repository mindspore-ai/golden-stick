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
"""Quant llama2 to w8a16."""

import argparse
import time
import numpy as np
from mindformers.core.metric import EmF1Metric
from mindformers import MindFormerConfig
from mindspore_gs.common import logger
from mindspore_gs.datasets import create_squad_dataset
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper


def evaluate(net, dataset_path, network_helper, n_samples):
    """evaluate `net` with dataset from `dataset_path`."""
    top_k = network_helper.get_spec("top_k")
    top_p = network_helper.get_spec("top_p")
    do_sample = network_helper.get_spec("do_sample")
    batch_size = network_helper.get_spec("batch_size")
    seq_length = network_helper.get_spec("seq_length")
    ignore_token_id = network_helper.get_spec("ignore_token_id")
    pad_token_id = network_helper.get_spec("pad_token_id")

    tokenizer = network_helper.create_tokenizer()
    ds = create_squad_dataset(dataset_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id,
                              n_samples=n_samples)
    metric = EmF1Metric()
    metric.clear()

    data_count = 0
    total_count = ds.get_dataset_size()
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        data_count += 1
        print(f"Dataset count: {data_count}/{total_count}", flush=True)
        input_ids = ds_item['input_ids'].asnumpy()
        labels = ds_item['labels'].asnumpy()

        batch_valid_length = []
        for j in range(input_ids.shape[0]):
            # As the nonzero returns the index and we need length
            batch_valid_length.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
        batch_valid_length = np.array(batch_valid_length)
        outputs = net.generate(input_ids, do_sample=do_sample, max_length=seq_length, top_p=top_p, top_k=top_k)
        output_ids = []
        for j in range(input_ids.shape[0]):
            output_ids.append(outputs[j][int(batch_valid_length[j]):])

        pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
        labels_str = tokenizer.decode(labels, skip_special_tokens=True)
        metric.update(pres_str, labels_str)
    print(f"EMF1: {metric.eval()}", flush=True)
    print('Evaluate Over!', flush=True)


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--n_samples', '-n', type=int, default=-1)
    args = parser.parse_args()
    logger.info(f"evaluate args: {args}")
    return args


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    logger.info('Creating network...')
    config = MindFormerConfig(uargs.config_path)
    if config.model.arch.type == "LlamaForCausalLM":
        helper = MFLlama2Helper(config)
    elif config.model.arch.type == "ParallelLlamaForCausalLM":
        helper = MFParallelLlama2Helper(config)
    else:
        err_msg = f"Unsupported network arch: {config.model.arch}, please check model.arch in yaml config, " \
                  f"only support LlamaForCausalLM and ParallelLlamaForCausalLM now"
        raise ValueError(err_msg)
    network = helper.create_network()
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    evaluate(network, uargs.dataset_path, helper, uargs.n_samples)
