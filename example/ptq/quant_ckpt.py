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
"""Quant network."""
import os
import argparse
import time

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication import get_rank
from mindformers import LlamaForCausalLM
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq import RoundToNearest as RTN
from mindspore_gs.ptq.smooth_quant import SmoothQuant as SQ
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.ptq.omni_quant import OmniQuant as OQ
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--approach', '-a', type=str, required=True, help="Available: w8a16, w8a8, c8, ptq, omni_quant")
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)
    parser.add_argument('--network', '-n', type=str, required=True)
    args = parser.parse_args()
    logger.info(f"quant args: {args}")
    return args


def create_ptq(approach, backend=BackendTarget.ASCEND):
    """Create ptq algorithm."""
    start_time = time.time()
    if approach == 'c8':
        logger.info("Use RoundToNearest(KVCacheInt8) algo to quant network and weight.")
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, weight_quant_dtype=None,
                        kvcache_quant_dtype=msdtype.int8)
        ptq = RTN(config=cfg)
    elif approach == 'w8a8':
        logger.info("Use SmoothQuant(W8A8) algo to quant network and weight.")
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, opname_blacklist=["w2", "lm_head"],
                        act_quant_dtype=msdtype.int8)
        ptq = SQ(config=cfg)
    elif approach == 'w8a16':
        logger.info("Use RoundToNearest(W8A16) algo to quant network and weight.")
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, opname_blacklist=["lm_head"])
        ptq = RTN(config=cfg)
    elif approach == 'ptq':
        logger.info("Use ptq algo to quant network and weight.")
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, opname_blacklist=["w2", "lm_head"],
                        algo_args={'enable_smooth': True})
        ptq = PTQ(config=cfg)
    elif approach == 'omni_quant':
        logger.info("Use omni quant algo to quant network and weight.")
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, opname_blacklist=["w2", "lm_head"],
                        weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8)
        ptq = OQ(config=cfg)
    else:
        raise ValueError(f"uargs.approach = {uargs.approach} is unexpected, "
                         "Available: w8a16, w8a8, c8, ptq, omni_quant.")
    logger.info(f'Create quantizer cost time is {time.time() - start_time} s.')
    return ptq


def create_ds(network_helper, ds_path, ds_type, approach):
    """Create datasets."""
    if approach in ['w8a8', 'c8', 'ptq', 'omni_quant']:
        start_time = time.time()
        if not ds_path:
            raise ValueError(f"Please provide dataset_path when approach is {approach}.")
        if not ds_type:
            raise ValueError(f"Please provide dataset_type when approach is {approach}.")
        bs_ = network_helper.get_spec('batch_size')
        seq_ = network_helper.get_spec('seq_length')
        max_decode_length = network_helper.get_spec('max_decode_length')
        ignore_token_id = network_helper.get_spec('ignore_token_id')
        tokenizer = network_helper.create_tokenizer()
        ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1,
                          False, n_samples=200)
        logger.info(f'Create datasets cost time is {time.time() - start_time} s.')
        return ds
    return None


def quant_net(net: LlamaForCausalLM, network_helper, ptq, ds):
    """Quant network with algorithm."""
    quant_start = time.time()
    logger.info('Quantize-ing network...')
    start_time = time.time()
    ptq.apply(net, network_helper, ds)
    logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    net.phase = "quant_convert"
    ptq.convert(net)
    logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
    logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')
    return net


if __name__ == "__main__":
    uargs = get_args()
    algo = create_ptq(approach=uargs.approach)
    if uargs.network == "LlamaForCasualLM":
        helper = MFLlama2Helper(uargs.config_path)
    elif uargs.network == "ParallelLlamaForCasualLM":
        helper = MFParallelLlama2Helper(uargs.config_path)
    datasets = create_ds(helper, uargs.dataset_path, uargs.dataset_type, approach=uargs.approach)
    start = time.time()
    logger.info('Creating network...')
    network = helper.create_network()
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    network = quant_net(network, helper, algo, datasets)
    logger.info('Saving checkpoint...')
    start = time.time()
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    save_ckpt_path = os.path.join(helper.mf_config.output_dir, f"{uargs.approach}_ckpt")
    save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, f"{uargs.approach}.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    logger.info(f'Checkpoint saved to {save_path}...')
