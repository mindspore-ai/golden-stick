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
from mindformers import MindFormerConfig
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget
from networks import NetworkRegister, BaseNetwork


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--fp_ckpt_path', '-k', type=str, default="")
    parser.add_argument('--fp_strategy_file', '-f', type=str, default="")
    parser.add_argument('--quant_strategy_file', '-q', type=str, default="")
    parser.add_argument('--network', '-n', type=str, default="llama2_7b",
                        help="optional: llama2_7b, llama2_13b, llama2_70b, baichuan2_13b, qwen_14b.")
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


def get_ckpt_list(ckpt_dir):
    """get_ckpt_list."""
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"ckpt_dir should be a directory.")
    ckpt_dict = {}
    for subdir in os.listdir(ckpt_dir):
        if not subdir.startswith("rank_"):
            continue
        rank_id_str = subdir[5:]
        if not rank_id_str.isnumeric():
            continue
        full_dir = os.path.join(ckpt_dir, subdir)
        find_ckpt = False
        for file in os.listdir(full_dir):
            if not file.endswith(".ckpt"):
                continue
            find_ckpt = True
            ckpt_dict[int(rank_id_str)] = os.path.join(ckpt_dir, subdir, file)
            break
        if not find_ckpt:
            raise ValueError(f"No ckpt file found in {full_dir}.")
    index = 0
    result = []
    while True:
        ckpt_file = ckpt_dict.get(index)
        if not ckpt_file:
            break
        result.append(ckpt_file)
        index += 1
    return result


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    print('------------------------- Creating network...', flush=True)
    net_mgr: BaseNetwork = NetworkRegister.instance().get(uargs.network)
    if not uargs.fp_ckpt_path:
        logger.warning(f'Float checkpoint path is empty, will quantize random init network.')
    config = net_mgr.create_mfconfig(uargs.config_path, "CPU", -1, 1, -1)
    network = net_mgr.create_network(config)
    network.set_train(False)
    network.phase = 'predict'
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    if uargs.fp_strategy_file:
        print('------------------------- Loading distributed checkpoint...', flush=True)
        start = time.time()
        ckpt_list = get_ckpt_list(uargs.fp_ckpt_path)
        ms.load_distributed_checkpoint(network, ckpt_list, train_strategy_filename=uargs.fp_strategy_file)
        logger.info(f'Loading distributed checkpoint cost time is {time.time() - start} s.')
    else:
        print('------------------------- Loading checkpoint...', flush=True)
        start = time.time()
        ms.load_checkpoint(uargs.fp_ckpt_path, network)
        logger.info(f'Loading checkpoint cost time is {time.time() - start} s.')

    print('------------------------- Quantize-ing network...', flush=True)
    start = time.time()
    device_config = MindFormerConfig(uargs.config_path)
    ms.set_context(device_target=device_config.context.device_target)
    network = net_mgr.quant_network(network, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    logger.info(f'Quant Network cost time is {time.time() - start} s.')
    print('------------------------- Saving checkpoint...', flush=True)
    start = time.time()
    os.makedirs("./llama2_w8a16/rank_0", exist_ok=True)
    ms.save_checkpoint(network, "./llama2_w8a16/rank_0/llama2_w8a16.ckpt",
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)
    if uargs.quant_strategy_file:
        print('------------------------- Saving distributed checkpoint...', flush=True)
        ms.transform_checkpoints("./llama2_w8a16/", "./llama2_w8a16_2p", "llama2_w8a16_", None,
                                 uargs.quant_strategy_file)
        print('------------------------- Checkpoint saved in ./llama2_w8a16_2p', flush=True)
    else:
        print('------------------------- Checkpoint saved in ./llama2_w8a16/rank_0/llama2_w8a16.ckpt', flush=True)
    logger.info(f'Saving checkpoint cost time is {time.time() - start} s.')
