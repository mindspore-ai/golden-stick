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
from collections import OrderedDict

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore.communication import get_rank
from mindspore import Model, Tensor
from mindspore.common import initializer
from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast

from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFDSV3Helper
from mindspore_gs.common import logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, QuantGranularity, PrecisionRecovery

from research.deepseek3.deepseek3 import DeepseekV3ForCausalLM
from research.deepseek3.deepseek3_config import DeepseekV3Config


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--approach', '-q', type=str, required=True,
                        help="Available: awq, smoothquant, dsquant")
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)

    args = parser.parse_args()
    logger.info(f"quant args: {args}")
    return args


def create_ptq(quant_type: str):
    """create_ptq"""
    if quant_type.lower() == 'dsquant':
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS,
                        opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                        act_quant_granularity=QuantGranularity.PER_TENSOR,
                        weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        ffn_config = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                               act_quant_dtype=msdtype.int8,
                               outliers_suppression=OutliersSuppressionType.NONE,
                               precision_recovery=PrecisionRecovery.NONE,
                               act_quant_granularity=QuantGranularity.PER_TOKEN,
                               weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        layer_policies = OrderedDict({r'.*\.feed_forward\..*': ffn_config})
    elif quant_type.lower() == 'awq':
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                        act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.AWQ,
                        opname_blacklist=['lm_head', 'lkv2kv'], weight_quant_granularity=QuantGranularity.PER_GROUP,
                        group_size=128)
        layer_policies = OrderedDict()
    elif quant_type.lower() == 'smoothquant':
        cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['lm_head', 'lkv2kv', 'w2'])
        layer_policies = OrderedDict()
    else:
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
    ptq = PTQ(config=cfg, layer_policies=layer_policies)
    if quant_type.lower() == 'awq':
        # pylint: disable=protected-access
        ptq._config.weight_symmetric = False
    from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
    ptq.decoder_layer_types.append(DeepseekV3DecodeLayer)
    return ptq

def create_ds(network_helper, ds_path, ds_type, approach):
    """Create datasets."""
    if approach in ['awq', 'smoothquant', 'dsquant']:
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
    ptq.summary(net)
    return net


if __name__ == "__main__":
    uargs = get_args()
    mfconfig = MindFormerConfig(uargs.config_path)
    build_context(mfconfig)
    build_parallel_config(mfconfig)
    model_config = mfconfig.model.model_config
    model_config.parallel_config = mfconfig.parallel_config
    model_config.moe_config = mfconfig.moe_config
    model_config = DeepseekV3Config(**model_config)

    tokenizer = LlamaTokenizerFast(mfconfig.processor.tokenizer.vocab_file,
                                   mfconfig.processor.tokenizer.tokenizer_file,
                                   unk_token=mfconfig.processor.tokenizer.unk_token,
                                   bos_token=mfconfig.processor.tokenizer.bos_token,
                                   eos_token=mfconfig.processor.tokenizer.eos_token,
                                   fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    network = DeepseekV3ForCausalLM(model_config)
    ms_model = Model(network)
    if mfconfig.load_checkpoint:
        seq_length = model_config.seq_length
        input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(mfconfig, ms_model, network, infer_data, do_predict=True)

    model_name = mfconfig.trainer.model_name
    helper = MFDSV3Helper(uargs.config_path)
    start = time.time()
    print('Creating network...', flush=True)
    network = helper.create_network()
    algo = create_ptq(uargs.approach)
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
