# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.datasets import create_squad_dataset
from mindspore_gs.ptq.evaluate import LLMGenerateNetwork
from mindformers import MindFormerConfig, LlamaConfig, LlamaForCausalLM, init_context, TransformerOpParallelConfig, \
    LlamaTokenizer
from mindformers.core.metric import EmF1Metric


def _set_config(config_path, device_id):
    """setup MindFormerConfig"""
    mfconfig = MindFormerConfig(config_path)
    if device_id != -1:
        mfconfig.context.device_id = device_id
    mfconfig.model.model_config = LlamaConfig(**mfconfig.model.model_config)

    init_context(use_parallel=mfconfig.use_parallel, context_config=mfconfig.context, parallel_config=mfconfig.parallel)

    parallel_config = TransformerOpParallelConfig(**mfconfig.parallel_config)
    mfconfig.model.model_config.parallel_config = parallel_config
    mfconfig.model.model_config.checkpoint_name_or_path = mfconfig.load_checkpoint
    print(mfconfig)
    return mfconfig


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
    config = _set_config(uargs.config_path, uargs.device_id)
    config.processor.tokenizer.vocab_file = uargs.tokenizer_path
    config.model.model_config.compute_dtype = ms.bfloat16
    config.model.model_config.layernorm_compute_type = ms.float32
    config.model.model_config.softmax_compute_type = ms.float16
    config.model.model_config.rotary_dtype = ms.float32
    config.model.model_config.param_init_type = ms.float32
    if not uargs.quant:
        config.load_checkpoint = uargs.ckpt_path
        config.model.model_config.checkpoint_name_or_path = uargs.ckpt_path

    network = LlamaForCausalLM(config.model.model_config)
    network.set_train(False)
    network.phase = 'predict'

    print('------------ eval llama2 ------------', flush=True)
    evaluate(network, uargs.dataset_path, uargs.tokenizer_path, config)

    ptq = RTN()
    ptq.set_weight_only_quant(True)
    quant_network = ptq.apply(network)
    ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
    ms.load_checkpoint("llama2-w8a16.ckpt", ascend_network)
    print('------------ eval W8A16 quant llama2 ------------', flush=True)
    evaluate(ascend_network, uargs.dataset_path, uargs.tokenizer_path, config)
