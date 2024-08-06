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
"""Quant llama2 7b to w8a8."""
import os
import argparse
import time

import mindspore as ms
from mindspore import dtype as msdtype
from mindformers import LlamaForCausalLM
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.omni_quant import OmniQuant as OQ
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--dataset_type', '-t', type=str, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    args = parser.parse_args()
    args.dataset_type = args.dataset_type.lower()
    logger.info(f"Quant args: {args}")
    return args


def quant_network(net: LlamaForCausalLM, net_helper, backend=BackendTarget.ASCEND, **kwargs):
    """Quant llama2 model to w8a8 with omni quant algorithm."""
    start_time = time.time()
    logger.info("Use omni quant algo to quant network and weight.")
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=backend, opname_blacklist=["w2", "lm_head"], act_dtype=msdtype.int8)
    ptq = OQ(config=cfg)
    logger.info(f'Create PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    ds_path = kwargs.get("ds_path", "")
    if not ds_path:
        raise ValueError("Please provide datasets for calibrating.")
    ds_type = kwargs.get("ds_type", "")
    if not ds_type:
        raise ValueError("Please provide datasets type for calibrating.")
    bs = net_helper.get_spec('batch_size')
    seq = net_helper.get_spec('seq_length')
    max_decode_length = net_helper.get_spec('max_decode_length')
    ignore_token_id = net_helper.get_spec('ignore_token_id')
    tokenizer = net_helper.create_tokenizer()
    ds = get_datasets(ds_type, ds_path, "train", bs, seq,
                      max_decode_length, tokenizer, ignore_token_id, 1, False, n_samples=200)
    net = ptq.apply(net, net_helper, ds=ds)
    logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    net.phase = "quant_convert"
    net = ptq.convert(net)
    logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
    return net


if __name__ == "__main__":
    start = time.time()
    uargs = get_args()
    logger.info('Creating network...')
    helper = MFLlama2Helper(uargs.config_path)
    network = helper.create_network()
    config = helper.mf_config
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    logger.info('Quantize-ing network...')
    start = time.time()
    network = quant_network(network, helper, backend=BackendTarget.ASCEND, ds_path=uargs.dataset_path,
                            ds_type=uargs.dataset_type)
    logger.info(f'Quant Network cost time is {time.time() - start} s.')
    logger.info('Saving checkpoint...')
    start = time.time()
    os.makedirs(config.output_dir, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(config.output_dir, "llama2_7b_w8a8_refactor.ckpt"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    logger.info(f'Checkpoint saved to {config.output_dir}...')
