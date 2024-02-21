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
"""Quant llama2 7b to kvint8."""
import argparse

import mindspore as ms
from mindspore import context
from mindspore_gs import Backend
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.datasets import create_wikitext_dataset
from mindformers import LlamaForCausalLM, LlamaTokenizer
from common import create_mfconfig


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--fp_ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


if __name__ == "__main__":
    uargs = get_args()
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    batch_size = 1
    seq_length = 1024
    config = create_mfconfig(uargs.config_path, uargs.device_id, batch_size, seq_length, ckpt_path=uargs.fp_ckpt_path)
    network = LlamaForCausalLM(config.model.model_config)
    network.set_train(False)
    network.phase = 'predict'

    print('------------ quant llama2 to KVCacheInt8 ------------', flush=True)
    ptq = RTN()
    ptq.set_linear_w8a16(False)
    ptq.set_kv_int8_quant(True)
    ptq.set_deploy(False)
    qnet = ptq.apply(network.model)
    network.model = qnet

    tokenizer = LlamaTokenizer(vocab_file=uargs.tokenizer_path)
    ds = create_wikitext_dataset(uargs.dataset_path, batch_size, seq_length, tokenizer)
    data_count = 0
    total_count = ds.get_dataset_size()
    for _, inputs in enumerate(ds.create_dict_iterator()):
        data_count += 1
        print(f"Dataset count: {data_count}/{total_count}", flush=True)
        input_ids = inputs['input_ids'].asnumpy()
        network.generate(input_ids, do_sample=False, max_length=seq_length, top_p=1, top_k=3)
        ptq.calibrate(network)

    network.model = ptq.convert(network.model, backend=Backend.GE_ASCEND)
    ms.save_checkpoint(network, "llama2-kvint8.ckpt")
