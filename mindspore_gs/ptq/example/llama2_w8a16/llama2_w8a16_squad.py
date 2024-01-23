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

import mindspore as ms
from mindspore import context
from mindspore_gs import Backend
from mindspore_gs.datasets import create_squad_dataset
from mindspore_gs.ptq.evaluate import LLMGenerateNetwork
from mindformers import LlamaForCausalLM, LlamaTokenizer
from mindformers.core.metric import EmF1Metric
from common import create_mfconfig, quant_llama2


def evaluate(net, dataset_path, vocab_file, cfg):
    """evaluate `net` with dataset from `dataset_path`."""
    top_k = cfg.model.model_config.top_k
    top_p = cfg.model.model_config.top_p
    do_sample = cfg.model.model_config.do_sample
    batch_size = cfg.model.model_config.batch_size
    seq_length = cfg.model.model_config.seq_length
    pad_token_id = cfg.model.model_config.pad_token_id
    ignore_token_id = cfg.model.model_config.ignore_token_id

    tokenizer = LlamaTokenizer(vocab_file=vocab_file)
    eval_net = LLMGenerateNetwork(net, do_sample, seq_length, top_p, top_k, pad_token_id, tokenizer)
    ds = create_squad_dataset(dataset_path, "eval", batch_size, seq_length, tokenizer, ignore_token_id)
    metrics = {"EmF1Metric": EmF1Metric()}
    model = ms.Model(eval_net, metrics=metrics, eval_network=eval_net)
    output = model.eval(ds, dataset_sink_mode=cfg.runner_config.sink_mode)
    print(f"EMF1: {output}")


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--quant', '-q', type=int, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True, help="Preprocessed dataset, "
                                                                              "must be in mindrecord format.")
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------evaluate args: {args}", flush=True)
    return args


if __name__ == "__main__":
    uargs = get_args()
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    config = create_mfconfig(uargs.config_path, uargs.device_id, 1, 512, uargs.tokenizer_path)
    if not uargs.quant:
        config.load_checkpoint = uargs.ckpt_path
        config.model.model_config.checkpoint_name_or_path = uargs.ckpt_path

    network = LlamaForCausalLM(config.model.model_config)
    network.set_train(False)
    network.phase = 'predict'

    if uargs.quant:
        network = quant_llama2(network, Backend.GE_ASCEND)
        if not uargs.ckpt_path:
            uargs.ckpt_path = "llama2-w8a16.ckpt"
        print('------------ eval W8A16 quant llama2 ------------', flush=True)
    else:
        print('------------ eval llama2 ------------', flush=True)
    ms.load_checkpoint(uargs.ckpt_path, network)
    evaluate(network, uargs.dataset_path, uargs.tokenizer_path, config)
