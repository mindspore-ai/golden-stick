"""Calibrate."""
import argparse
import os
from collections import OrderedDict

import mindspore as ms
from mindspore import dataset
from mindspore import dtype as msdtype
from mindformers import MindFormerConfig
from transformers import AutoTokenizer

from mindspore_gs.common import BackendTarget
from mindspore_gs.common import logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq import (OutliersSuppressionType, PrecisionRecovery,
                              PTQConfig, PTQMode, QuantGranularity,
                              GPTQQuantConfig)
from mindspore_gs.ptq.models import AutoQuantForCausalLM


def get_args():
    """Get args."""
    parser = argparse.ArgumentParser(description="OSL PTQ calibration for DeepSeekV3.")
    parser.add_argument("--config_path",
                        type=str,
                        required=True,
                        help="Path to the calibrate yaml config file.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Directory to save the quantized model.")
    parser.add_argument("--quant_type",
                        type=str,
                        required=True,
                        help="The quantization algorithm.")
    parser.add_argument("--ds_type",
                        type=str,
                        required=True,
                        help="Type of the dataset.")
    parser.add_argument("--ds_path",
                        type=str,
                        required=True,
                        help="Path to the dataset.")
    return parser.parse_args()


def create_ptq_config(quant_type: str):
    """Create PTQ configuration"""
    if quant_type.lower() == "a8w4":
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                        opname_blacklist=['output_layer', 'kv_up_proj'], weight_clip=True)
        mlp_config = PTQConfig(backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                               act_quant_dtype=msdtype.int8,
                               outliers_suppression=OutliersSuppressionType.NONE,
                               precision_recovery=PrecisionRecovery.NONE,
                               act_quant_granularity=QuantGranularity.PER_TOKEN,
                               weight_quant_granularity=QuantGranularity.PER_CHANNEL,
                               weight_clip=True)
        gptq_config = GPTQQuantConfig(static_groups=True, desc_act=True)
        moe_cfg = PTQConfig(backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                            act_quant_dtype=msdtype.int8, act_quant_granularity=QuantGranularity.PER_TOKEN,
                            weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=256,
                            algo_args=gptq_config, precision_recovery=PrecisionRecovery.GPTQ, weight_clip=True)
        layer_policies = OrderedDict({r'.*\.mlp\.linear_fc1.*': mlp_config,
                                      r'.*\.mlp\.linear_fc2.*': mlp_config,
                                      r'.*\.mlp\.shared_experts\.linear_fc1.*': mlp_config,
                                      r'.*\.mlp\.shared_experts\.linear_fc2.*': mlp_config,
                                      r'.*\.mlp\.experts\.linear_fc1.*': moe_cfg,
                                      r'.*\.mlp\.experts\.linear_fc2.*': moe_cfg})
    else:
        raise ValueError(f"Not support {quant_type} right now.")
    return cfg, layer_policies


def create_ds(ds_path, tokenizer, ds_type='ceval', n_samples=200):
    """Create datasets."""
    dataset.config.set_numa_enable(False)
    seq_ = 2048
    max_decode_length = 1024
    ignore_token_id = tokenizer.pad_token_id
    ds = get_datasets(ds_type, ds_path, 'train', 1, seq_, max_decode_length, tokenizer, ignore_token_id,
                      1, False, n_samples=n_samples)
    return ds


def quant_dsv3(config_path, output_dir, quant_type,
               ds_type, ds_path):
    """PTQ quant to quant qwen3"""
    mfconfig = MindFormerConfig(config_path)
    tokenizer = AutoTokenizer.from_pretrained(mfconfig.pretrained_model_dir)
    datasets = create_ds(ds_path, tokenizer, ds_type=ds_type)
    model = AutoQuantForCausalLM.from_pretrained(config_path)
    cfg, layers_policy = create_ptq_config(quant_type)
    calibrate_options = {
        'algorithm_cache_path': {'osl': 'osl_cache'},
        'always_use_fp_input_in_processer': True,
        'skip_offload_in_processing': True,
    }
    ms.mint.distributed.barrier()
    model.calibrate(cfg, layers_policy, datasets, **calibrate_options)
    ckpt_path = model.save_quantized(output_dir)
    logger.info(f'Save quantized model to {ckpt_path}')


def main():
    """Main function."""
    args = get_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    quant_dsv3(args.config_path, args.output_dir, args.quant_type,
               args.ds_type, args.ds_path)

if __name__ == "__main__":
    main()
