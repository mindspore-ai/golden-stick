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
"""Quant llama2 7b to w8a16."""

import os
import argparse
import time
import numpy as np
import mindspore as ms
from mindspore import log as logger
from mindspore import Model
from mindspore import ops
from mindspore.train.metrics import Perplexity
from mindspore.communication import get_rank
from mindspore_gs.datasets import create_wikitext_dataset
from networks import NetworkRegister, BaseNetwork


def evaluate(net, dataset_path, config_, net_helper: BaseNetwork):
    """evaluate."""
    bs_ = config_.model.model_config.batch_size
    seq_ = config_.model.model_config.seq_length
    tokenizer = net_helper.create_tokenizer(config_.processor.tokenizer.vocab_file)
    ds = create_wikitext_dataset(dataset_path, bs_, seq_, 1, tokenizer)
    metric = Perplexity()
    data_count = 0
    total_count = ds.get_dataset_size()
    for _, ds_item in enumerate(ds.create_dict_iterator()):
        data_count += 1
        logger.info(f"Dataset count: {data_count}/{total_count}")
        input_ids = ds_item['input_ids'].asnumpy()
        net_inputs = net_helper.assemble_inputs(input_ids, config_)
        output = net(*net_inputs)
        output = ops.squeeze(output)[:-1, :]
        label = input_ids[:, 1:]
        metric.update(output, label)
    print('...........Evaluate Over!...............', flush=True)
    print(f"PPL: {metric.eval()}", flush=True)


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    args = parser.parse_args()
    logger.info(f"-------------------------------------------------evaluate args: {args}")
    return args


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    print('------------------------- Creating network...', flush=True)
    net_mgr: BaseNetwork = NetworkRegister.instance().from_config(uargs.config_path)
    config = net_mgr.create_mfconfig(uargs.config_path)
    network = net_mgr.create_network(config)
    network.set_train(False)
    network.phase = 'predict'
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    start = time.time()
    rank_id = get_rank() or 0
    ckpt_path = config.load_checkpoint
    bs = config.model.model_config.batch_size
    seq = config.model.model_config.seq_length
    block_size = config.model.model_config.block_size
    if os.path.isdir(ckpt_path):
        for file in os.listdir(os.path.join(ckpt_path, f"rank_{rank_id}")):
            if not file.endswith(".ckpt"):
                continue
            ckpt_path = os.path.join(ckpt_path, f"rank_{rank_id}", file)
            model = Model(network)
            inputs = network.prepare_inputs_for_predict_layout(input_ids=np.ones([bs, seq], dtype=np.int32))
            model.infer_predict_layout(*inputs)
            break
    logger.info(f'Loading ckpt :{ckpt_path}.')
    ms.load_checkpoint(ckpt_path, network)
    logger.info(f'Load ckpt cost time is {time.time() - start} s.')
    evaluate(network, uargs.dataset_path, config, net_mgr)
