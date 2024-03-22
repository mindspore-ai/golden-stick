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
import argparse

import mindspore as ms
from mindspore import context
from mindspore import log as logger
from mindformers import LlamaForCausalLM, LlamaTokenizer
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode, PTQConfig
from mindspore_gs.ptq.smooth_quant.smooth_quant import SmoothQuant
from mindspore_gs.datasets import create_wikitext_dataset
from common import create_mfconfig


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--fp_ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    parser.add_argument('--calib_ds_path', '-s', type=str, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


if __name__ == "__main__":
    uargs = get_args()
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    bs = 1
    seq_len = 256
    config = create_mfconfig(uargs.config_path, uargs.device_id, bs, seq_len, ckpt_path=uargs.fp_ckpt_path)
    network = LlamaForCausalLM(config.model.model_config)
    tokenizer = LlamaTokenizer(vocab_file=uargs.tokenizer_path)
    network.set_train(False)
    network.phase = 'predict'

    print('------------ quant llama2 to W8A16 ------------', flush=True)
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    ptq = SmoothQuant(cfg)
    network.model = ptq.apply(network.model)
    ds = create_wikitext_dataset(uargs.calib_ds_path, bs, seq_len, tokenizer)
    data_count = 0
    total_count = ds.get_dataset_size()
    for _, inputs in enumerate(ds.create_dict_iterator()):
        data_count += 1
        logger.info(f"Dataset count: {data_count}/{total_count}")
        input_ids = inputs['input_ids'].asnumpy()
        _ = network.generate(input_ids, do_sample=False, max_length=seq_len, top_p=1, top_k=3)
    network.model = ptq.convert(network.model)
    ms.save_checkpoint(network, f"llama2-w8a8-dev{uargs.device_id}.ckpt")
