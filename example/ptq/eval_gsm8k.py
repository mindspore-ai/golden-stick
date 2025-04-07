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
"""Eval with gsm8k datasets."""

import argparse
import time
import re
import math
from decimal import Decimal, InvalidOperation
import numpy as np
from mindspore_gs.common import logger
from mindspore_gs.datasets import create_gsm8k_dataset
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper
from mindformers import MindFormerConfig

def cal_acc_gsm8k(answer, res_data):
    '''cal_acc_gsm8k'''
    def extract_last_digit(res):
        '''extract_last_digit'''
        res_str = r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
        pat_last_digit = re.compile(res_str)
        match = list(pat_last_digit.finditer(res))
        if match:
            last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
        else:
            last_digit = None
        return last_digit

    res_last_digit = extract_last_digit(res_data[0])
    labels_res = extract_last_digit(answer[0])
    print(f"labels res: {labels_res}, extract predict res: {res_last_digit}", flush=True)
    if res_last_digit is None or labels_res is None:
        return False
    try:
        res_last_digit = Decimal(res_last_digit)
        labels_res = Decimal(labels_res)
        return math.isclose(res_last_digit, labels_res, rel_tol=0, abs_tol=Decimal('1e-4'))
    except (InvalidOperation, ValueError, TypeError, SyntaxError) as e:
        print(f"Error evaluating expression: {e}. Please check the predict result or the answer.", flush=True)
        return False
    except OverflowError as e:
        print(f"OverflowError: {e}.", flush=True)
        return False


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
    ds = create_gsm8k_dataset(dataset_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id,
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
                               max_new_tokens=2048)
        output_ids = []
        for j in range(input_ids.shape[0]):
            output_ids.append(outputs[j][int(batch_valid_length[j]):])

        question = tokenizer.decode(input_ids, skip_special_tokens=True)
        pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
        labels_str = tokenizer.decode(labels, skip_special_tokens=True)

        if cal_acc_gsm8k(labels_str, pres_str):
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
