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
import os
import argparse

from mindformers.core.metric import PerplexityMetric
import mindspore as ms
from mindspore import context
from mindspore import log as logger
from mindspore.communication import get_rank
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore_gs.datasets import create_wikitext_dataset
from mindspore_gs.ptq import PTQMode
from mindspore_gs.common import BackendTarget
from networks import NetworkRegister, BaseNetwork


def evaluate(net, dataset_path, bs, seq_len, max_new_tokens_, tokenizer):
    """evaluate."""
    ds = create_wikitext_dataset(dataset_path, bs, seq_len, max_new_tokens_, tokenizer)
    metrics = {"PerplexityMetric": PerplexityMetric()}
    model = ms.Model(net, metrics=metrics, eval_network=net)
    step_size = ds.get_dataset_size()
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    output = model.eval(ds, dataset_sink_mode=config.runner_config.sink_mode, callbacks=cb)
    print(f"PPL: {output}", flush=True)


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--ckpt_path', '-k', type=str, required=True)
    parser.add_argument('--quant', '-q', type=int, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True)
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    parser.add_argument('--parallel', '-p', type=int, default=1)
    parser.add_argument('--network', '-n', type=str, default="llama2_7b",
                        help="optional: llama2_7b, llama2_13b, llama2_70b, baichuan2_13b, qwen_14b.")
    args = parser.parse_args()
    logger.info(f"-------------------------------------------------evaluate args: {args}")
    return args


if __name__ == "__main__":
    uargs = get_args()
    net_mgr: BaseNetwork = NetworkRegister.instance().get(uargs.network)
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    batch_size = 1
    seq_length = 2048
    device_id = int(os.getenv('DEVICE_ID', '0'))
    config = net_mgr.create_mfconfig(uargs.config_path, "Ascend", device_id, batch_size, seq_length,
                                     uargs.tokenizer_path, model_parallel=uargs.parallel)
    rank_id = 0 if uargs.parallel == 1 else get_rank()
    print(f"start wikitext2 evaluate: rank {rank_id}, bs {batch_size}, seq_len {seq_length}, config {uargs}.",
          flush=True)
    network = net_mgr.create_network(config)
    if uargs.quant:
        network = net_mgr.quant_network(network, mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
        if not uargs.ckpt_path:
            if uargs.parallel == 1:
                uargs.ckpt_path = "llama2-w8a16.ckpt"
            else:
                uargs.ckpt_path = f"llama2-w8a16-r{rank_id}.ckpt"
        print('------------ eval W8A16 quant llama2 ------------', flush=True)
    else:
        print('------------ eval llama2 ------------', flush=True)
    ms.load_checkpoint(uargs.ckpt_path, network)
    max_new_tokens = uargs.max_new_tokens or seq_length // 2
    evaluate(network, uargs.dataset_path, batch_size, seq_length, max_new_tokens,
             net_mgr.create_tokenizer(uargs.tokenizer_path))
