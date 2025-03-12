# Copyright 2025 Huawei Technologies Co., Ltd
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
"""calibrate network."""

import os
import time
import argparse

import mindspore as ms
from mindspore.communication import get_rank

from mindformers import MindFormerConfig
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFDSV3Helper
from mindspore_gs.common import logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq import PTQMode
from utils import create_ptq, create_network


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--approach', '-q', type=str, required=True,
                        help="Available: awq-a16w8, awq-a16w4, smoothquant, dsquant, a16w8, a8dynw8")
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)

    args = parser.parse_args()
    logger.info(f"quant args: {args}")
    return args


def create_ds(network_helper, ds_path, ds_type, approach):
    """Create datasets."""
    if approach in ['awq-a16w8', 'awq-a16w4', 'smoothquant', 'dsquant', 'a16w8', 'a8dynw8']:
        start_time = time.time()
        if not ds_path:
            raise ValueError(f"Please provide dataset_path when approach is {approach}.")
        if not ds_type:
            raise ValueError(f"Please provide dataset_type when approach is {approach}.")
        bs_ = network_helper.get_spec('batch_size')
        seq_ = network_helper.get_spec('seq_length')
        max_decode_length = network_helper.get_spec('max_decode_length')
        ignore_token_id = network_helper.get_spec('ignore_token_id')
        tokenizer_ = network_helper.create_tokenizer()
        ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer_, ignore_token_id,
                          1, False, n_samples=200)
        logger.info(f'Create datasets cost time is {time.time() - start_time} s.')
        return ds
    return None


def quant_net(net, network_helper, ptq, ds):
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
    mfconfig = MindFormerConfig(uargs.config)
    model_name = mfconfig.trainer.model_name
    helper = MFDSV3Helper(uargs.config)
    start = time.time()
    print('Creating network...', flush=True)
    _, network = create_network(uargs.config, auto_online_trans=True)
    algo = create_ptq(uargs.approach, PTQMode.QUANTIZE)
    datasets = create_ds(helper, uargs.dataset_path, uargs.dataset_type, approach=uargs.approach)
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    print('Quanting network...', flush=True)
    network = quant_net(network, helper, algo, datasets)
    print('Saving checkpoint...', flush=True)
    start = time.time()
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0

    save_ckpt_path = os.path.join(helper.mf_config.output_dir, f"{model_name}_{uargs.approach}_safetensors")
    save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, f"{uargs.approach}"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x,
                       format="safetensors")
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    print(f'Checkpoint saved to {save_ckpt_path}', flush=True)
