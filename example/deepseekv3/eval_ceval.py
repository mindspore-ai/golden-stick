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
from collections import OrderedDict

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication import get_rank
from mindformers import MindFormerConfig
from mindspore_gs.ptq.ptq_config import (PTQMode, PTQConfig, OutliersSuppressionType,
                                         PrecisionRecovery, QuantGranularity)
from mindspore_gs.common import BackendTarget, logger
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFDSV3Helper


def dtype_formatter(name: str):
    """dtype_formatter"""
    if name == 'int8':
        return msdtype.int8
    if name == 'int4':
        return msdtype.qint4x2
    return None


def granularity_formatter(name: str):
    """granularity_formatter"""
    if name == 'per_tensor':
        return QuantGranularity.PER_TENSOR
    if name == 'per_channel':
        return QuantGranularity.PER_CHANNEL
    if name == 'per_token':
        return QuantGranularity.PER_TOKEN
    if name == 'per_group':
        return QuantGranularity.PER_GROUP
    return None


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)

    args = parser.parse_args()
    logger.info(f"quant args: {args}")
    return args


def create_ptq():
    """Create ptq algorithm."""
    start_time = time.time()
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                    act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                    opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                    act_quant_granularity=QuantGranularity.PER_TENSOR,
                    weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    wo_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                          outliers_suppression=OutliersSuppressionType.NONE,
                          precision_recovery=PrecisionRecovery.NONE,
                          act_quant_granularity=QuantGranularity.PER_TENSOR,
                          weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    ffn_config = PTQConfig(weight_quant_dtype=msdtype.int8, act_quant_dtype=msdtype.int8,
                           outliers_suppression=OutliersSuppressionType.NONE,
                           precision_recovery=PrecisionRecovery.NONE,
                           act_quant_granularity=QuantGranularity.PER_TENSOR,
                           weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    logger.info("Use ptq algo to quant network and weight.")
    ptq = PTQ(config=cfg, layer_configs=OrderedDict({r'.*attention\.wo.*': wo_config,
                                                     r'.*feed_forward.*': ffn_config}))
    from openmind_modules.deepseek3.deepseek3_model import DeepseekV3DecodeLayer
    ptq.decoder_layers.append(DeepseekV3DecodeLayer)
    logger.info(f'Create quantizer cost time is {time.time() - start_time} s.')
    return ptq


def create_ds(network_helper, ds_path, approach):
    """Create datasets."""
    start_time = time.time()
    if not ds_path:
        raise ValueError(f"Please provide dataset_path when approach is {approach}.")
    bs_ = network_helper.get_spec('batch_size')
    seq_ = network_helper.get_spec('seq_length')
    max_decode_length = network_helper.get_spec('max_decode_length')
    ignore_token_id = network_helper.get_spec('ignore_token_id')
    tokenizer = network_helper.create_tokenizer()
    ds = get_datasets('ceval', ds_path, "train", bs_, seq_, max_decode_length, tokenizer, ignore_token_id, 1,
                      False, n_samples=200)
    logger.info(f'Create datasets cost time is {time.time() - start_time} s.')
    return ds


def quant_net(net, ptq):
    """Quant network with algorithm."""
    quant_start = time.time()
    logger.info('Quantize-ing network...')
    start_time = time.time()
    ptq.apply(net)
    logger.info(f'Apply PTQ cost time is {time.time() - start_time} s.')
    start_time = time.time()
    net.phase = "quant_convert"
    ptq.convert(net)
    logger.info(f'Convert to real quantize cost time is {time.time() - start_time} s.')
    logger.info(f'Quant Network cost total time is {time.time() - quant_start} s.')
    return net


def ckpt_name(model_name_):
    """ckpt_name"""
    name = f"{model_name_}_ptq_hyper_a8w8"
    return name


if __name__ == "__main__":
    uargs = get_args()
    mfconfig = MindFormerConfig(uargs.config_path)
    model_name = mfconfig.trainer.model_name
    helper = MFDSV3Helper(uargs.config_path)
    start = time.time()
    print('Creating network...', flush=True)
    network = helper.create_network()
    algo = create_ptq()
    datasets = create_ds(helper, uargs.dataset_path, approach='ptq')
    logger.info(f'Create Network cost time is {time.time() - start} s.')
    print('Quanting network...', flush=True)
    network = quant_net(network, algo)
    print('Saving checkpoint...', flush=True)
    start = time.time()
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0

    if mfconfig.load_ckpt_format == "safetensors":
        save_ckpt_path = os.path.join(helper.mf_config.output_dir, f"{ckpt_name(model_name)}_safetensors")
        save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, "ptq"),
                           choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x,
                           format="safetensors")
    else:
        save_ckpt_path = os.path.join(helper.mf_config.output_dir, f"{ckpt_name(model_name)}_ckpt")
        save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, "ptq.ckpt"),
                           choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    print(f'Checkpoint saved to {save_ckpt_path}', flush=True)
