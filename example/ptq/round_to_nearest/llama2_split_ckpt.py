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

import mindspore as ms
from mindspore import log as logger
from networks import NetworkRegister, BaseNetwork


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--parallel', '-p', type=int, default=1)
    parser.add_argument('--tokenizer', '-t', type=str, required=True)
    parser.add_argument('--network', '-n', type=str, default="llama2_7b",
                        help="optional: llama2_7b, llama2_13b, llama2_70b, baichuan2_13b, qwen_14b.")
    args = parser.parse_args()
    logger.info(f"-------------------------------------------------evaluate args: {args}")
    return args


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    net_mgr: BaseNetwork = NetworkRegister.instance().get(uargs.network)
    if not uargs.ckpt_path:
        logger.warning(f'Float checkpoint path is empty, will quantize random init network.')
    config = net_mgr.create_mfconfig(uargs.config_path, "CPU", -1, 1, -1, ckpt_path=uargs.ckpt_path)
    network = net_mgr.create_network(config)
    network.set_train(False)
    network.phase = 'predict'
    logger.info(f'Create Network cost time is {time.time() - start} s.')

    print('------------ split llama2 ------------', flush=True)
    start = time.time()
    logger.info(f'Load ckpt cost time is {time.time() - start} s.')
    start = time.time()
    tokenizer = net_mgr.create_tokenizer(uargs.tokenizer)
    input_ids = tokenizer('Hello')['input_ids']
    outputs = network.generate(input_ids, do_sample=False,
                               max_length=config.model.model_config.seq_length, top_p=1, top_k=3)
    logger.info(f'Infer predict layout cost time is {time.time() - start} s.')
    start = time.time()
    rank_id = os.getenv('RANK_ID')
    ms.save_checkpoint(network, f"./quant_ckpt/rank_{rank_id}/{uargs.network}_w8a16.ckpt")
    logger.info(f'Save ckpt cost time is {time.time() - start} s.')
