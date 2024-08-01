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

import numpy as np
import mindspore as ms
from mindspore import dtype as msdtype
from mindspore import Model
from mindspore.communication import get_rank
from mindformers import LlamaForCausalLM
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper
from llama2 import Llama2Network


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--approach', '-a', type=str, required=True, help="Available: w8a16, c8")
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


def quant_network(net: LlamaForCausalLM, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, kvcache_int8=False,
                  **kwargs):
    """Quant llama2 model to w8a16 with RTN algorithm."""
    start_time = time.time()
    if mode == PTQMode.QUANTIZE:
        logger.info("Use RTN algo to quant network and weight.")
    else:
        logger.info("Use RTN algo to quant network.")
    cfg = PTQConfig(mode=mode, backend=backend, opname_blacklist=["lm_head"])
    logger.info(f'Create PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    mfconfig = kwargs.get("mfconfig", None)
    if not mfconfig:
        raise ValueError("Please provide mfconfig for calibrating.")
    network_helper = MFLlama2Helper(mfconfig)
    ds = None
    if kvcache_int8:
        logger.info('create dataset for kvcache_int8 quant.')
        cfg = PTQConfig(mode=mode, backend=backend, weight_dtype=msdtype.float_, kvcache_dtype=msdtype.int8)
        ds_path = kwargs.get("ds_path", "")
        if not ds_path:
            raise ValueError("Please provide datasets for calibrating.")
        ds_type = kwargs.get("ds_type", "")
        if not ds_type:
            raise ValueError("Please provide datasets type for calibrating.")
        bs_ = network_helper.get_spec('batch_size')
        seq_ = network_helper.get_spec('seq_length')
        max_decode_length = network_helper.get_spec('max_decode_length')
        ignore_token_id = network_helper.get_spec('ignore_token_id')
        tokenizer = network_helper.create_tokenizer()
        ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1,
                          False, n_samples=200)
    ptq = RTN(config=cfg)
    net = ptq.apply(net, network_helper, ds)
    logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    net.phase = "quant_convert"
    net = ptq.convert(net)
    logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
    return net


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    if uargs.approach == "c8" and (uargs.dataset_path is None or uargs.dataset_type is None):
        raise ValueError("Please provide dataset_path and dataset_type in args when uargs.approach is c8.")
    print('------------------------- Creating network...', flush=True)
    net_mgr: Llama2Network = Llama2Network()
    config = net_mgr.create_mfconfig(uargs.config_path)
    network = net_mgr.create_network(config)
    network.set_train(False)
    network.phase = 'predict'
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    start = time.time()
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    ckpt_path = config.load_checkpoint
    if os.path.isdir(ckpt_path):
        for file in os.listdir(os.path.join(ckpt_path, f"rank_{rank_id}")):
            if not file.endswith(".ckpt"):
                continue
            ckpt_path = os.path.join(ckpt_path, f"rank_{rank_id}", file)
            model = Model(network)
            bs = config.model.model_config.batch_size
            seq = config.model.model_config.seq_length
            inputs = network.prepare_inputs_for_predict_layout(input_ids=np.ones([bs, seq], dtype=np.int32))
            model.infer_predict_layout(*inputs)
            break
    logger.info(f'Loading ckpt :{ckpt_path}.')
    ms.load_checkpoint(ckpt_path, network)
    ms.ms_memory_recycle()
    logger.info(f'Load ckpt cost time is {time.time() - start} s.')
    print('------------------------- Quantize-ing network...', flush=True)
    start = time.time()
    if uargs.approach == "c8":
        print('------------------------- c8 quant...', flush=True)
        save_ckpt_path = os.path.join(config.output_dir, "c8_ckpt")
        network = quant_network(network, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND,
                                kvcache_int8=True, ds_path=uargs.dataset_path, ds_type=uargs.dataset_type,
                                mfconfig=config)
    elif uargs.approach == "w8a16":
        print('------------------------- w8a16 quant...', flush=True)
        save_ckpt_path = os.path.join(config.output_dir, "w8a16_ckpt")
        network = quant_network(network, mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, mfconfig=config)
    else:
        raise ValueError("uargs.approach = {} is unexpected, Available: w8a16, c8.".format(uargs.approach))

    logger.info(f'Quant Network cost time is {time.time() - start} s.')
    print('------------------------- Saving checkpoint...', flush=True)
    start = time.time()
    save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, "w8a16.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    print(f'------------------------- Checkpoint saved to {save_path}...', flush=True)
