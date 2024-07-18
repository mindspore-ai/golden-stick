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

import os
import argparse
import time
import numpy as np
from mindformers.core.metric import EmF1Metric
import mindspore as ms
from mindspore import log as logger
from mindspore import Model
from mindspore.communication import get_rank
from mindspore_gs.datasets import create_squad_dataset
from llama2 import Llama2Network


def evaluate(net, dataset_path, cfg, net_helper: Llama2Network):
    """evaluate `net` with dataset from `dataset_path`."""
    top_k = cfg.model.model_config.top_k
    top_p = cfg.model.model_config.top_p
    do_sample = cfg.model.model_config.do_sample
    batch_size = cfg.model.model_config.batch_size
    seq_length = cfg.model.model_config.seq_length
    ignore_token_id = cfg.model.model_config.ignore_token_id
    pad_token_id = cfg.model.model_config.pad_token_id

    tokenizer = net_helper.create_tokenizer(cfg.processor.tokenizer.vocab_file)
    ds = create_squad_dataset(dataset_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id)
    metric = EmF1Metric()
    metric.clear()

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

        outputs = net.generate(input_ids, do_sample=do_sample, max_length=seq_length, top_p=top_p, top_k=top_k)
        output_ids = []
        for j in range(input_ids.shape[0]):
            output_ids.append(outputs[j][int(valid_length_each_example[j]):])

        pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
        labels_str = tokenizer.decode(labels, skip_special_tokens=True)
        metric.update(pres_str, labels_str)
    metric.eval()
    print('...........Evaluate Over!...............', flush=True)


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
    net_mgr: Llama2Network = Llama2Network()
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
