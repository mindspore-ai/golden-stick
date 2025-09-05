"""Calibrate."""
import argparse
import os
from collections import OrderedDict

from mindspore_gs.common import BackendTarget
from mindspore_gs.common import logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq import (OutliersSuppressionType, PrecisionRecovery, PTQConfig, PTQMode, QuantGranularity)
from mindspore_gs.ptq.models import AutoQuantForCausalLM
from transformers import AutoTokenizer
from mindformers import MindFormerConfig
import mindspore as ms
from mindspore import dataset
from mindspore import dtype as msdtype


def get_args():
    """Get args."""
    parser = argparse.ArgumentParser(description="OSL PTQ calibration for DeepSeekV3.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the calibrate yaml config file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the quantized model.")
    parser.add_argument("--ds_path", type=str, required=True, help="Path to the dataset.")
    return parser.parse_args()


def create_ptq_config():
    """Create PTQ configuration"""
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                    act_quant_dtype=msdtype.int8,
                    outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                    opname_blacklist=['.output_layer', '.linear_kv_up_proj'])
    mlp_config = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                           act_quant_dtype=msdtype.int8,
                           outliers_suppression=OutliersSuppressionType.NONE,
                           precision_recovery=PrecisionRecovery.NONE,
                           act_quant_granularity=QuantGranularity.PER_TOKEN,
                           weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    layer_policies = OrderedDict({r'.*\.mlp\..*': mlp_config})
    return cfg, layer_policies


def create_ds(ds_path, tokenizer, ds_type='ceval', n_samples=200):
    """Create datasets."""
    dataset.config.set_numa_enable(False)
    seq_ = 200 # 2048
    max_decode_length = 100 # 1024
    ignore_token_id = tokenizer.pad_token_id
    ds = get_datasets(ds_type, ds_path, 'train', 1, seq_, max_decode_length, tokenizer, ignore_token_id,
                      1, False, n_samples=n_samples)
    return ds


def quant_dsv3(config_path, output_dir, ds_path):
    """PTQ quant to quant qwen3"""
    mfconfig = MindFormerConfig(config_path)
    tokenizer = AutoTokenizer.from_pretrained(mfconfig.pretrained_model_dir)
    datasets = create_ds(ds_path, tokenizer)
    model = AutoQuantForCausalLM.from_pretrained(config_path)
    cfg, layers_policy = create_ptq_config()
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
    os.makedirs(args.output_dir, exist_ok=True)
    quant_dsv3(args.config_path, args.output_dir, args.ds_path)

if __name__ == "__main__":
    main()
