# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Quant llama2 to w8a16."""
import argparse

import mindspore as ms
from mindspore import dataset
import mindspore.dataset.transforms as C
from mindspore import context, dtype
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore_gs import Backend
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindformers import MindFormerConfig, LlamaConfig, LlamaForCausalLM, init_context, TransformerOpParallelConfig
from mindformers.core.metric import PerplexityMetric


def _set_config(config_path, device_id):
    """setup MindFormerConfig"""
    mfconfig = MindFormerConfig(config_path)
    if device_id != -1:
        mfconfig.context.device_id = device_id
    mfconfig.model.model_config = LlamaConfig(**mfconfig.model.model_config)

    init_context(use_parallel=mfconfig.use_parallel, context_config=mfconfig.context, parallel_config=mfconfig.parallel)

    parallel_config = TransformerOpParallelConfig(**mfconfig.parallel_config)
    mfconfig.model.model_config.parallel_config = parallel_config
    mfconfig.model.model_config.checkpoint_name_or_path = mfconfig.load_checkpoint
    print(mfconfig)
    return mfconfig


def _get_rank_info(distribute):
    """
    get rank size and rank id
    """
    if distribute:
        init()
        rank_id = get_rank()
        device_num = get_group_size()
    else:
        rank_id = 0
        device_num = 1
    return device_num, rank_id


def create_dataset(bs, repeat_count=1, distribute=False, do_shuffle=True, dataset_path=""):
    """create dataset like language model task"""
    device_num, rank_id = _get_rank_info(distribute)
    type_cast_op = C.TypeCast(dtype.int32)
    ds = dataset.MindDataset(dataset_path,
                             columns_list=["input_ids"],
                             shuffle=do_shuffle,
                             num_shards=device_num,
                             shard_id=rank_id)
    print("batch_size: {}".format(bs))

    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.batch(bs, drop_remainder=True)
    ds = ds.repeat(repeat_count)

    print("dataset size: {}".format(ds.get_dataset_size()))
    print("repeat count: {}".format(ds.get_repeat_count()))
    print("output shape: {}".format(ds.output_shapes()))
    print("output type: {}".format(ds.output_types()))
    print("============== create dataset successful ===============")

    return ds


def evaluate(net, dataset_path):
    ds = create_dataset(bs=batch_size, dataset_path=dataset_path)
    metrics = {"PerplexityMetric": PerplexityMetric()}
    model = ms.Model(net, metrics=metrics, eval_network=net)
    output = model.eval(ds, dataset_sink_mode=config.runner_config.sink_mode)
    print(f"PPL: {output}")


def get_args():
    """init user options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-c', type=str, required=True)
    parser.add_argument('--device_id', '-d', type=int, required=True)
    parser.add_argument('--dataset_path', '-s', type=str, required=True, help="Preprocessed dataset, "
                                                                              "must be in mindrecord format.")
    parser.add_argument('--tokenizer_path', '-t', type=str, required=True)
    args = parser.parse_args()
    print(f"-------------------------------------------------evaluate args: {args}", flush=True)
    return args


if __name__ == "__main__":
    uargs = get_args()
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    config = _set_config(uargs.config_path, uargs.device_id)
    config.processor.tokenizer.vocab_file = uargs.tokenizer_path
    network = LlamaForCausalLM(config.model.model_config)
    batch_size = config.model.model_config.batch_size
    network.set_train(False)
    network.phase = 'predict'
    print('------------ eval llama2 ------------', flush=True)
    evaluate(network, uargs.dataset_path)

    ptq = RTN()
    ptq.set_weight_only_quant(True)
    quant_network = ptq.apply(network)
    ascend_network = ptq.convert(quant_network, backend=Backend.GE_ASCEND)
    print('------------ eval W8A16 quant llama2 ------------', flush=True)
    evaluate(ascend_network, uargs.dataset_path)
