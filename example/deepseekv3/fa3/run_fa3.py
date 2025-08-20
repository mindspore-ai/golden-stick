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

import time
import argparse

from mindspore import dataset

from mindformers import MindFormerConfig
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFDSV3Helper
from mindspore_gs.common import logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.fa3.fa3 import FA3Config, FA3

from example.deepseekv3.ds_utils import create_network

def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--fa3_params_path', '-p', type=str, required=False)
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)

    args = parser.parse_args()
    logger.info(f"quant args: {args}")
    return args


def create_ds(network_helper, ds_path, ds_type, tokenizer_):
    """Create datasets."""
    start_time = time.time()
    bs_ = network_helper.get_spec('batch_size')
    seq_ = network_helper.get_spec('seq_length')
    max_decode_length = network_helper.get_spec('max_decode_length')
    ignore_token_id = network_helper.get_spec('ignore_token_id')
    ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer_, ignore_token_id,
                      1, False, n_samples=200)
    logger.info(f'Create datasets cost time is {time.time() - start_time} s.')
    return ds

if __name__ == "__main__":
    uargs = get_args()
    mfconfig = MindFormerConfig(uargs.config)
    helper = MFDSV3Helper(uargs.config)
    start = time.time()
    print('Creating network...', flush=True)
    tokenizer, network = create_network(uargs.config)
    #create fa3
    export_params_path = uargs.fa3_params_path if uargs.fa3_params_path is not None else './fa3_params'
    cfg = FA3Config(export_params_path=export_params_path, dsk_config=mfconfig.model.model_config)
    fa3 = FA3(config=cfg)
    # datasets
    dataset.config.set_numa_enable(False)
    datasets = create_ds(helper, uargs.dataset_path, uargs.dataset_type, tokenizer_=tokenizer)
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    print('Runing FA3...', flush=True)
    fa3.observe(network, helper, datasets)
    print('Runing FA3 quantizer end.', flush=True)
