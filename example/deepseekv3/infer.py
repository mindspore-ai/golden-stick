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
"""ds infer."""
import argparse
from collections import OrderedDict

import mindspore as ms
from mindspore import dtype as msdtype
from mindspore import Model, Tensor
from mindspore.common import initializer
from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast

from mindspore_gs.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, QuantGranularity, PrecisionRecovery

from research.deepseek3.deepseek3 import DeepseekV3ForCausalLM
from research.deepseek3.deepseek3_config import DeepseekV3Config

input_questions = [['介绍下北京故宫']]


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
    from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
    ptq.decoder_layer_types.append(DeepseekV3DecodeLayer)
    return ptq


def infer(yaml_file, quant_type):
    """infer"""
    config = MindFormerConfig(yaml_file)
    build_context(config)
    build_parallel_config(config)
    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.moe_config = config.moe_config
    model_config = DeepseekV3Config(**model_config)

    tokenizer = LlamaTokenizerFast(config.processor.tokenizer.vocab_file,
                                   config.processor.tokenizer.tokenizer_file,
                                   unk_token=config.processor.tokenizer.unk_token,
                                   bos_token=config.processor.tokenizer.bos_token,
                                   eos_token=config.processor.tokenizer.eos_token,
                                   fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    network = DeepseekV3ForCausalLM(model_config)
    ptq = create_ptq(quant_type)
    ptq.apply(network)
    ptq.convert(network)
    ptq.summary(network)

    ms_model = Model(network)
    if config.load_checkpoint:
        seq_length = model_config.seq_length
        input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, ms_model, network, infer_data, do_predict=True)

    multi_inputs = []
    for question in input_questions:
        multi_inputs.append(tokenizer(question, max_length=64, padding="max_length")["input_ids"])
    for batch_input in multi_inputs:
        output = network.generate(batch_input, max_length=1024, do_sample=False, top_k=5, top_p=1, max_new_tokens=1024)
        answer = tokenizer.decode(output)
        print("answer:", answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--approach', required=True, type=str, help="awq, smoothquant, dsquant")
    args = parser.parse_args()
    infer(args.yaml_file, args.approach)
