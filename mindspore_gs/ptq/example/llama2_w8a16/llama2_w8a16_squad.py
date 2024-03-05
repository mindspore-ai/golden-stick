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

import numpy as np
import mindspore as ms
from mindspore import context
from mindformers import LlamaForCausalLM, LlamaTokenizer
from mindformers.core.metric import EmF1Metric
from mindspore_gs.datasets import create_squad_dataset
from mindspore_gs import PTQMode, BackendTarget
from common import create_mfconfig, quant_llama2


def evaluate(net: LlamaForCausalLM, dataset_path, vocab_file, cfg):
    """evaluate `net` with dataset from `dataset_path`."""
    top_k = cfg.model.model_config.top_k
    top_p = cfg.model.model_config.top_p
    do_sample = cfg.model.model_config.do_sample
    batch_size = cfg.model.model_config.batch_size
    seq_length = cfg.model.model_config.seq_length
    ignore_token_id = cfg.model.model_config.ignore_token_id

    tokenizer = LlamaTokenizer(vocab_file=vocab_file)
    ds = create_squad_dataset(dataset_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id)
    metric = EmF1Metric()
    metric.clear()

    pad_token_id = tokenizer.pad_token_id
    data_count = 0
    total_count = ds.get_dataset_size()
    for _, inputs in enumerate(ds.create_dict_iterator()):
        data_count += 1
        print(f"Dataset count: {data_count}/{total_count}", flush=True)
        input_ids = inputs['input_ids'].asnumpy()
        labels = inputs['labels'].asnumpy()

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
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--quant', '-q', type=int, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    parser.add_argument('--parallel', '-p', type=int, default=1)
    args = parser.parse_args()
    print(f"-------------------------------------------------evaluate args: {args}", flush=True)
    return args


if __name__ == "__main__":
    uargs = get_args()
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    config = create_mfconfig(uargs.config_path, uargs.device_id, 1, 2048, uargs.tokenizer_path,
                             model_parallel=uargs.parallel)

    network = LlamaForCausalLM(config.model.model_config)
    network.set_train(False)
    network.phase = 'predict'

    if uargs.quant:
        network = quant_llama2(network, mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
        if not uargs.ckpt_path:
            uargs.ckpt_path = "llama2-w8a16.ckpt"
        print('------------ eval W8A16 quant llama2 ------------', flush=True)
    else:
        print('------------ eval llama2 ------------', flush=True)
    ms.load_checkpoint(uargs.ckpt_path, network)
    evaluate(network, uargs.dataset_path, uargs.tokenizer_path, config)
