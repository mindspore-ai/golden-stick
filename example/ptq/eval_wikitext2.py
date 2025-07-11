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
"""Quant llama2 7b to w8a16, please set use_past=False."""

import argparse
import time
from mindformers.core.metric import PerplexityMetric
from mindformers import MindFormerConfig
from mindspore_gs.common import logger
from mindspore_gs.datasets import create_wikitext_dataset
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper


def evaluate(net, dataset_path, net_helper, n_samples):
    """evaluate."""
    bs_ = net_helper.get_spec("batch_size")
    seq_ = net_helper.get_spec("seq_length")
    tokenizer_ = net_helper.create_tokenizer()
    ds = create_wikitext_dataset(dataset_path, bs_, seq_, 1, tokenizer_, n_samples=n_samples)
    metric = PerplexityMetric()
    metric.clear()
    data_count = 0
    total_count = ds.get_dataset_size()
    net.is_first_iteration = False
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        data_count += 1
        print(f"Dataset count: {data_count}/{total_count}", flush=True)
        input_ids = ds_item['input_ids'].asnumpy()
        net_inputs = net_helper.assemble_inputs(input_ids)
        outputs = net(*net_inputs)
        metric.update(*outputs)
    print(f"PPL: {metric.eval()}", flush=True)
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
    config.run_mode = 'eval'
    config.model.model_config.use_past = False
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
