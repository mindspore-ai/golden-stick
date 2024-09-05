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
"""Eval w8a8 llama2 with squad1.1 datasets."""

import argparse
import time
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore_gs.common import logger
from mindspore_gs.datasets import create_boolq_dataset
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper
from mindformers.generation.text_generator import GenerationMixin
from mindformers import MindFormerConfig


def evaluate(net, dataset_path, network_helper):
    """evaluate `net` with dataset from `dataset_path`."""
    batch_size = network_helper.get_spec("batch_size")
    seq_length = network_helper.get_spec("seq_length")
    ignore_token_id = network_helper.get_spec("ignore_token_id")
    pad_token_id = network_helper.get_spec("pad_token_id")
    tokenizer = network_helper.create_tokenizer()
    ds = create_boolq_dataset(dataset_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id)
    choice_tokens = [tokenizer.encode("yes", add_special_tokens=False)[0],
                     tokenizer.encode("no", add_special_tokens=False)[0]]

    correct = 0
    data_count = 0
    total_count = ds.get_dataset_size()
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        data_count += 1
        logger.info(f"Dataset count: {data_count}/{total_count}")
        input_ids = ds_item['input_ids'].asnumpy()
        labels = ds_item['labels'].asnumpy()
        valid_length_each_example = []
        for j in range(input_ids.shape[0]):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)

        logits, current_index = net.forward(
            input_ids=input_ids,
            valid_length_each_example=valid_length_each_example,
            use_past=network_helper.get_spec("use_past"))
        logit_logsoftmax = GenerationMixin.process_logits(
            GenerationMixin, logits[0], Tensor(current_index, dtype=mstype.int32))

        logit_logsoftmax = logit_logsoftmax[:, choice_tokens]
        if logit_logsoftmax[0, 0] > logit_logsoftmax[0, 1]:
            choice = choice_tokens[0]
        else:
            choice = choice_tokens[1]
        acc = choice == labels[0][0]
        if acc:
            correct += 1
        if data_count % 10 == 0:
            logger.info(f"acc: {correct / data_count}")
    logger.info(f"total acc: {correct / data_count}")
    logger.info('Evaluate Over!')


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--network', '-n', type=str, required=True)
    args = parser.parse_args()
    logger.info(f"evaluate args: {args}")
    return args


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    logger.info('Creating network...')
    config = MindFormerConfig(uargs.config_path)
    config.model.model_config.use_past = False
    if uargs.network == "LlamaForCausalLM":
        helper = MFLlama2Helper(config)
    elif uargs.network == "ParallelLlamaForCausalLM":
        helper = MFParallelLlama2Helper(config)
    network = helper.create_network()
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    evaluate(network, uargs.dataset_path, helper)
