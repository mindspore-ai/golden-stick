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
import logging
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
    parser.add_argument('--approach', '-q', type=str, required=True,
                        help="Available: rtn-a16w8, rtn-c8, smooth_quant, ptq")
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)

    parser.add_argument('--weight_quant_dtype', '-w', type=str, default='none',
                        help="Available: 'int8', 'int4', 'none'")
    parser.add_argument('--act_quant_dtype', '-a', type=str, default='none', help="Available: 'int8', 'none'")
    parser.add_argument('--kvcache_quant_dtype', '-k', type=str, default='none', help="Available: 'int8', 'none'")

    parser.add_argument('--outliers_suppression', '-o', type=str, default='none', help="Available: 'smooth', 'none'")
    parser.add_argument('--precision_recovery', '-p', type=str, default='none', help="Available: gptq")

    parser.add_argument('--act_quant_granularity', '-ag', type=str, default='per_tensor',
                        help="Available: 'per_token', 'per_tensor'")
    parser.add_argument('--weight_quant_granularity', '-wg', type=str, default="per_channel",
                        help="Available: per_channel/per_group")
    parser.add_argument('--kvcache_quant_granularity', '-kvg', type=str, default="per_channel",
                        help="Available: per_channel/per_token")
    parser.add_argument('--group_size', '-g', type=int, default=0, help="Available: 64/128")

    parser.add_argument('--opname_blacklist', '-b', type=str, nargs='*',
                        help="A list of model layers not to convert, set blacklist when use PTQ algo. "
                             "For example: -b w2 lm_head.")
    parser.add_argument('--debug_mode', '-e', type=bool, default=False, help="Enable debug info, default: False, "
                                                                             "Available: True, False")
    parser.add_argument('--dump_path', '-d', type=str, default="", help="Save the quantized data to the provided file "
                                                                        "path.")

    args = parser.parse_args()
    args.act_quant_granularity = QuantGranularity.from_str(args.act_quant_granularity)
    args.weight_quant_granularity = QuantGranularity.from_str(args.weight_quant_granularity)
    args.kvcache_quant_granularity = QuantGranularity.from_str(args.kvcache_quant_granularity)
    args.outliers_suppression = OutliersSuppressionType.from_str(args.outliers_suppression)
    args.precision_recovery = PrecisionRecovery.from_str(args.precision_recovery)

    args.weight_quant_dtype = dtype_formatter(args.weight_quant_dtype)
    args.act_quant_dtype = dtype_formatter(args.act_quant_dtype)
    args.kvcache_quant_dtype = dtype_formatter(args.kvcache_quant_dtype)
    args.opname_blacklist = args.opname_blacklist if args.opname_blacklist else []

    logger.info(f"quant args: {args}")
    return args


def create_ptq(uargs_):
    """Create ptq algorithm."""
    start_time = time.time()
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
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
    # pylint: disable=protected-access
    ptq._config.set_dump_path(uargs_.dump_path)
    if uargs_.outliers_suppression == OutliersSuppressionType.AWQ:
        # pylint: disable=protected-access
        ptq._config.weight_symmetric = False
    logger.info(f'Create quantizer cost time is {time.time() - start_time} s.')
    return ptq


def create_ds(network_helper, ds_path, ds_type, approach):
    """Create datasets."""
    if approach in ['rtn-c8', 'smooth_quant', 'ptq', 'omni_quant']:
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


def ckpt_name(model_name_, uargs_):
    """ckpt_name"""
    if uargs_.approach != 'ptq':
        return f"{model_name_}_{uargs_.approach}"
    name = f"{model_name_}_ptq"
    if uargs_.outliers_suppression == OutliersSuppressionType.SMOOTH:
        name += "_smooth"
    elif uargs_.outliers_suppression == OutliersSuppressionType.AWQ:
        name += "_awq"
    else:
        name += "_no_smooth"
    if uargs_.precision_recovery == PrecisionRecovery.GPTQ:
        name += "_gptq"
    if uargs_.act_quant_dtype == msdtype.int8:
        if uargs_.act_quant_granularity is QuantGranularity.PER_TOKEN:
            name += "_a8dyn"
        else:
            name += "_a8"
    else:
        name += "_a16"
    if uargs_.weight_quant_dtype == msdtype.int8:
        name += "w8"
    elif uargs_.weight_quant_dtype == msdtype.qint4x2:
        name += "w4"
    else:
        name += "w16"
    if uargs_.kvcache_quant_dtype == msdtype.int8:
        name += "c8"
    return name


if __name__ == "__main__":
    uargs = get_args()
    if uargs.debug_mode:
        logger.set_level(logging.DEBUG)
    mfconfig = MindFormerConfig(uargs.config_path)
    model_name = mfconfig.trainer.model_name
    helper = MFDSV3Helper(uargs.config_path)
    start = time.time()
    print('Creating network...', flush=True)
    network = helper.create_network()
    algo = create_ptq(uargs)
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

    if mfconfig.load_ckpt_format == "safetensors":
        save_ckpt_path = os.path.join(helper.mf_config.output_dir, f"{ckpt_name(model_name, uargs)}_safetensors")
        save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, f"{uargs.approach}"),
                           choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x,
                           format="safetensors")
    else:
        save_ckpt_path = os.path.join(helper.mf_config.output_dir, f"{ckpt_name(model_name, uargs)}_ckpt")
        save_path = os.path.join(save_ckpt_path, f"rank_{rank_id}")
        os.makedirs(save_path, exist_ok=True)
        ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, f"{uargs.approach}.ckpt"),
                           choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x)
    logger.info(f'Save checkpoint cost time is {time.time() - start} s.')
    print(f'Checkpoint saved to {save_ckpt_path}', flush=True)
