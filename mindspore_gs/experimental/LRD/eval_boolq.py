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
"""Eval with boolq datasets."""

import argparse
import time
import numpy as np
from mindformers import MindFormerConfig
from mindspore_gs.common import logger
from mindspore_gs.datasets import create_boolq_dataset
from lrd_mf_net_helpers import LRDDeployMFParallelLlama2Helper


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
    ds = create_boolq_dataset(dataset_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id,
                              n_samples=n_samples)

    correct = 0
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
        outputs = net.generate(input_ids, do_sample=do_sample, max_length=seq_length, top_k=top_k, top_p=top_p,
                               max_new_tokens=5)
        output_ids = []
        for j in range(input_ids.shape[0]):
            output_ids.append(outputs[j][int(batch_valid_length[j]):])

        question = tokenizer.decode(input_ids, skip_special_tokens=True)
        pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
        labels_str = tokenizer.decode(labels, skip_special_tokens=True)

        if labels_str[0].lower() in pres_str[0].lower():
            correct += 1
            print(f"question: {question}\n predict: {pres_str} answer: {labels_str}. correct!", flush=True)
        else:
            print(f"question: {question}\n predict: {pres_str} answer: {labels_str}. not correct!", flush=True)
        if data_count % 100 == 0:
            print(f"acc: {correct / data_count}", flush=True)
    print(f"total acc: {correct / data_count}", flush=True)
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
    helper = LRDDeployMFParallelLlama2Helper(config)
    network = helper.create_network()
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    evaluate(network, uargs.dataset_path, helper, uargs.n_samples)
