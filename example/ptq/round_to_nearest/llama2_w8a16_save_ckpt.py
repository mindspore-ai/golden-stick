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
    parser.add_argument('--network', '-n', type=str, default="llama2_7b",
                        help="optional: llama2_7b, llama2_13b, llama2_70b, baichuan2_13b, qwen_14b.")
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    net_mgr: BaseNetwork = NetworkRegister.instance().get(uargs.network)
    if not uargs.fp_ckpt_path:
        logger.warning(f'Float checkpoint path is empty, will quantize random init network.')
    config = net_mgr.create_mfconfig(uargs.config_path, "CPU", -1, 1, 2048, ckpt_path=uargs.fp_ckpt_path)
    network = net_mgr.create_network(config)
    network.set_train(False)
    network.phase = 'predict'
    logger.info(f'Create Network cost time is {time.time() - start} s.')

    print('------------ quant llama2 to W8A16 ------------', flush=True)
    start = time.time()
    device_config = MindFormerConfig(uargs.config_path)
    ms.set_context(device_target=device_config.context.device_target)
    network = net_mgr.quant_network(network, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    logger.info(f'Quant Network cost time is {time.time() - start} s.')
    start = time.time()
    ms.save_checkpoint(network, "llama2-w8a16.ckpt")
    logger.info(f'Save quant ckpt cost time is {time.time() - start} s.')
