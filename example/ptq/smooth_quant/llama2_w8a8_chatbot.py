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
"""Quant llama2 7b and provide a simple chat api."""
import argparse

import mindspore as ms
from mindspore import context
from mindspore_gs import BackendTarget
from mindformers import LlamaForCausalLM, LlamaTokenizer, BaseModel
from common import create_mfconfig, quant_llama2


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--quant', '-q', type=int, required=True)
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------evaluate args: {args}", flush=True)
    return args


def chat(net: BaseModel, tokenizer_: LlamaTokenizer, max_length):
    while True:
        question = input("Please input question:")
        if question == "exit":
            break
        input_ids = tokenizer_(question)['input_ids']
        outputs = net.generate(input_ids, do_sample=False, max_length=max_length, top_p=1, top_k=3)
        answer = tokenizer_.decode(outputs, skip_special_tokens=True)
        print(f"Answer: {answer}\r\n", flush=True)


if __name__ == "__main__":
    uargs = get_args()
    seq_length = 2048
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    config = create_mfconfig(uargs.config_path, uargs.device_id, 1, seq_length, uargs.tokenizer_path)
    network = LlamaForCausalLM(config.model.model_config)
    network.set_train(False)
    network.phase = 'predict'
    tokenizer = LlamaTokenizer(vocab_file=uargs.tokenizer_path)
    if uargs.quant:
        network = quant_llama2(network, is_deploy=True, backend=BackendTarget.ASCEND, max_length=2048,
                               tokernizer=tokenizer)
    ms.load_checkpoint(uargs.ckpt_path, network)
    chat(network, tokenizer, seq_length)
