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
"""Quant llama2 7b to w8a16."""
import argparse

import mindspore as ms
from mindspore import context
from mindspore_gs import Backend
from mindformers import LlamaForCausalLM
from common import quant_llama2, create_mfconfig


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--fp_ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------quant args: {args}", flush=True)
    return args


if __name__ == "__main__":
    uargs = get_args()
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    config = create_mfconfig(uargs.config_path, uargs.device_id, 1, 512, ckpt_path=uargs.fp_ckpt_path)
    network = LlamaForCausalLM(config.model.model_config)
    network.set_train(False)
    network.phase = 'predict'

    print('------------ quant llama2 to W8A16 ------------', flush=True)
    network = quant_llama2(network, Backend.GE_ASCEND, True)
    ms.save_checkpoint(network, "llama2-w8a16.ckpt")
