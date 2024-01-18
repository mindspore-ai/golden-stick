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
"""Quant llama2 7b to w8a16."""
import argparse

import mindspore as ms
from mindspore import context
from mindspore_gs import Backend
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.datasets import create_wikitext_dataset
from mindformers import MindFormerConfig, LlamaConfig, LlamaForCausalLM, init_context, TransformerOpParallelConfig, \
    LlamaTokenizer
from mindformers.core.metric import PerplexityMetric


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


def evaluate(net, dataset_path, bs, seq_len, vocab_file):
    tokenizer = LlamaTokenizer(vocab_file=vocab_file)
    ds = create_wikitext_dataset(dataset_path, bs, seq_len, tokenizer)
    metrics = {"PerplexityMetric": PerplexityMetric()}
    model = ms.Model(net, metrics=metrics, eval_network=net)
    output = model.eval(ds, dataset_sink_mode=config.runner_config.sink_mode)
    print(f"PPL: {output}")


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
    config.model.model_config.seq_length = 512
    config.model.model_config.compute_dtype = ms.bfloat16
    config.model.model_config.layernorm_compute_type = ms.float32
    config.model.model_config.softmax_compute_type = ms.float16
    config.model.model_config.rotary_dtype = ms.float32
    config.model.model_config.param_init_type = ms.float32
    if not uargs.quant:
        config.load_checkpoint = uargs.ckpt_path
        config.model.model_config.checkpoint_name_or_path = uargs.ckpt_path
    network = LlamaForCausalLM(config.model.model_config)
    batch_size = config.model.model_config.batch_size
    seq_length = config.model.model_config.seq_length
    network.set_train(False)
    network.phase = 'predict'
    if uargs.quant:
        ptq = RTN()
        ptq.set_weight_only_quant(True)
        quant_network = ptq.apply(network)
        ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
        if not uargs.ckpt_path:
            uargs.ckpt_path = "llama2-w8a16.ckpt"
        ms.load_checkpoint(uargs.ckpt_path, ascend_network)
        print('------------ eval W8A16 quant llama2 ------------', flush=True)
        evaluate(ascend_network, uargs.dataset_path, batch_size, seq_length, uargs.tokenizer_path)
    else:
        print('------------ eval llama2 ------------', flush=True)
        evaluate(network, uargs.dataset_path, batch_size, seq_length, uargs.tokenizer_path)
