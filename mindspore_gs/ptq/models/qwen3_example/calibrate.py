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

from collections import OrderedDict
import time
import argparse

from mindspore import dataset
from mindspore import dtype as msdtype
from mindformers import MindFormerConfig
from mindspore_gs.ptq.network_helpers.mf_net_helpers import Qwen3Helper
from mindspore_gs.common import logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType
from mindspore_gs.ptq.models import AutoModel
from transformers import AutoTokenizer


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--approach', '-q', type=str, required=True,
                        help="Available: smoothquant")
    parser.add_argument('--dataset_type', '-t', type=str, required=False)
    parser.add_argument('--dataset_path', '-s', type=str, required=False)

    args = parser.parse_args()
    logger.info(f"quant args: {args}")
    return args


def create_ds(network_helper, ds_path, ds_type, approach, tokenizer_):
    """Create datasets."""
    if approach in ['smoothquant']:
        start_time = time.time()
        if not ds_path:
            raise ValueError(f"Please provide dataset_path when approach is {approach}.")
        if not ds_type:
            raise ValueError(f"Please provide dataset_type when approach is {approach}.")
        bs_ = network_helper.get_spec('batch_size')
        seq_ = network_helper.get_spec('seq_length')
        max_decode_length = network_helper.get_spec('max_decode_length')
        ignore_token_id = network_helper.get_spec('pad_token_id')
        ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer_, ignore_token_id,
                          1, False, n_samples=200)
        logger.info(f'Create datasets cost time is {time.time() - start_time} s.')
        return ds
    return None


def create_ptq_config(quant_type: str):
    """create_ptq"""
    if quant_type.lower() == 'smoothquant':
        cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['output_layer', 'linear_fc2'])
        layer_policies = OrderedDict()
    else:
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
    return cfg, layer_policies


if __name__ == "__main__":
    uargs = get_args()
    mfconfig = MindFormerConfig(uargs.config)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(mfconfig.load_checkpoint)
    dataset.config.set_numa_enable(False)
    helper = Qwen3Helper(uargs.config)
    datasets = create_ds(helper, uargs.dataset_path, uargs.dataset_type, approach=uargs.approach, tokenizer_=tokenizer)
    model = AutoModel.from_pretrained(uargs.config)
    cfg_, layers_policy = create_ptq_config('smoothquant')
    model.calibrate(cfg_, layers_policy, datasets)
    model.save_pretrained(mfconfig.output_dir)
