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
from mindspore import Model
from mindspore.communication import get_rank
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget
from networks import NetworkRegister, BaseNetwork


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--fp_ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--save_ckpt_path', '-s', type=str, required=True)
    parser.add_argument('--network', '-n', type=str, default="llama2_7b",
                        help="optional: llama2_7b, llama2_13b, llama2_57b, llama2_70b, baichuan2_13b, qwen_14b.")
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    print('------------------------- Creating network...', flush=True)
    net_mgr: BaseNetwork = NetworkRegister.instance().get(uargs.network)
    if not uargs.fp_ckpt_path:
        logger.warning(f'Float checkpoint path is empty, will quantize random init network.')
    config = net_mgr.create_mfconfig(uargs.config_path)
    network = net_mgr.create_network(config)
    network.set_train(False)
    network.phase = 'predict'
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    start = time.time()
    rank_id = get_rank()
    model = Model(network)
    model.infer_predict_layout(*(net_mgr.gen_fake_inputs(1, 4096, 128)))
    if os.path.isdir(uargs.fp_ckpt_path):
        for file in os.listdir(os.path.join(uargs.fp_ckpt_path, f"rank_{rank_id}")):
            if not file.endswith(".ckpt"):
                continue
            uargs.fp_ckpt_path = os.path.join(uargs.fp_ckpt_path, f"rank_{rank_id}", file)
    logger.info(f'Load ckpt :{uargs.fp_ckpt_path}.')
    ms.load_checkpoint(uargs.fp_ckpt_path, network)
    ms.ms_memory_recycle()
    logger.info(f'Load ckpt cost time is {time.time() - start} s.')
    print('------------------------- Quantize-ing network...', flush=True)
    start = time.time()
    network = net_mgr.quant_network(network, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    logger.info(f'Quant Network cost time is {time.time() - start} s.')
    print('------------------------- Saving checkpoint...', flush=True)
    start = time.time()
    save_path = os.path.join(uargs.save_ckpt_path, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, "w8a16.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
