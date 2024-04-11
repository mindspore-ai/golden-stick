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
from mindformers import Tokenizer, BaseModel
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget
from networks import NetworkRegister, BaseNetwork


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--quant', '-q', type=int, required=True)
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    parser.add_argument('--parallel', '-p', type=int, default=1)
    parser.add_argument('--network', '-n', type=str, default="llama2_7b",
                        help="optional: llama2_7b, llama2_13b, llama2_70b, baichuan2_13b, glm3_6b, qwen_14b.")
    args = parser.parse_args()
    print(f"-------------------------------------------------evaluate args: {args}", flush=True)
    return args


def chat(net: BaseModel, tokenizer_: Tokenizer, max_length, use_parallel: bool):
    """chat."""
    if use_parallel:
        input_ids = tokenizer_("Hello.")['input_ids']
        outputs = net.generate(input_ids, do_sample=False, max_length=max_length, top_p=1, top_k=3)
        answer = tokenizer_.decode(outputs, skip_special_tokens=True)
        print(f"Answer: {answer}\r\n", flush=True)
    else:
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
    net_mgr: BaseNetwork = NetworkRegister.instance().get(uargs.network)
    if net_mgr is None:
        raise RuntimeError(f"Unsupported network: {uargs.network}, available: llama2_7b, llama2_13b, llama2_70b, "
                           "baichuan2_13b, glm3_6b, qwen_14b.")
    seq_length = 256
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    config = net_mgr.create_mfconfig(uargs.config_path, "Ascend", uargs.device_id, 1, seq_length, uargs.tokenizer_path,
                                     model_parallel=uargs.parallel)
    network = net_mgr.create_network(config.model.model_config)
    if uargs.quant:
        network = net_mgr.quant_network(network, mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
    ms.load_checkpoint(uargs.ckpt_path, network)
    tokenizer = net_mgr.create_tokenizer(uargs.tokenizer_path)
    chat(network, tokenizer, seq_length, uargs.parallel > 1)
