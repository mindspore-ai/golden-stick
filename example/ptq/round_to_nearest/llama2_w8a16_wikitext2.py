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
import argparse

from mindformers import LlamaForCausalLM, LlamaTokenizer
from mindformers.core.metric import PerplexityMetric
import mindspore as ms
from mindspore import context
from mindspore import log as logger
from mindspore_gs.datasets import create_wikitext_dataset
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget
from common import create_mfconfig, quant_llama2


def evaluate(net, dataset_path, bs, seq_len, vocab_file):
    """evaluate."""
    tokenizer = LlamaTokenizer(vocab_file=vocab_file)
    ds = create_wikitext_dataset(dataset_path, bs, seq_len, tokenizer)
    metrics = {"PerplexityMetric": PerplexityMetric()}
    model = ms.Model(net, metrics=metrics, eval_network=net)
    output = model.eval(ds, dataset_sink_mode=config.runner_config.sink_mode)
    print(f"PPL: {output}", flush=True)


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
    logger.info(f"-------------------------------------------------evaluate args: {args}")
    return args


if __name__ == "__main__":
    uargs = get_args()
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    batch_size = 1
    seq_length = 2048
    config = create_mfconfig(uargs.config_path, uargs.device_id, batch_size, seq_length, uargs.tokenizer_path,
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
    evaluate(network, uargs.dataset_path, batch_size, seq_length, uargs.tokenizer_path)
