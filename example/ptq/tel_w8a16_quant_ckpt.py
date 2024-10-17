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
from mindspore.communication import get_rank

from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.ptq.network_helpers.tel_net_helpers import TELHelper
from mindspore_gs.ptq.network_helpers.tel_net_helpers import TELHelper2

def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--network', '-n', type=str, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


def quant_network(net, net_helper, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, **kwargs):
    """Quant llama2 model to w8a16 with RTN algorithm."""
    start_time = time.time()
    if mode == PTQMode.QUANTIZE:
        logger.info("Use RTN algo to quant network and weight.")
    else:
        logger.info("Use RTN algo to quant network.")
    cfg = PTQConfig(mode=mode, backend=backend, opname_blacklist=["lm_head"])
    ptq = RTN(config=cfg)
    logger.info(f'Create PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    mfconfig = kwargs.get("mfconfig", None)
    if not mfconfig:
        raise ValueError("Please provide mfconfig for calibrating.")
    net = ptq.apply(net, net_helper)
    logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    net.phase = "quant_convert"
    net = ptq.convert(net)
    logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
    return net


if __name__ == "__main__":
    uargs = get_args()
    print('------------------------- Creating network...', flush=True)
    start = time.time()
    if uargs.network == "telechat":
        network_helper = TELHelper(uargs.config_path)
    elif uargs.network == "telechat2":
        network_helper = TELHelper2(uargs.config_path)
    network = network_helper.create_network()
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    print('------------------------- Quantize-ing network...', flush=True)
    network = quant_network(network, network_helper, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    logger.info(f'Quant Network cost time is {time.time() - start} s.')
    print('------------------------- Saving checkpoint...', flush=True)
    start = time.time()
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    save_ckpt_path = os.path.join(network_helper.mf_config.output_dir, "w8a16_ckpt")
    save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, f"{uargs.approach}.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    print(f'------------------------- Checkpoint saved to {save_path}...', flush=True)
