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

from collections import OrderedDict

import mindspore as ms
from mindspore.communication.comm_func import barrier
from mindspore import dtype as msdtype
from mindspore import Model, Tensor
from mindspore.common import initializer
from mindspore.nn.utils import no_init_parameters
from mindformers import MindFormerConfig
from mindformers import build_context
from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast

from mindspore_gs.ptq import PTQ
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, QuantGranularity, PrecisionRecovery, \
    GPTQQuantConfig
from deepseekv3_weight_processor import DeepseekV3WeightProcessor

from research.deepseek3.deepseek3 import DeepseekV3ForCausalLM
from research.deepseek3.deepseek3_config import DeepseekV3Config


def create_ptq(quant_type: str, quant_mode: PTQMode):
    """create_ptq"""
    if quant_type.lower() == 'dsquant':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS,
                        opname_blacklist=['lkv2kv', 'lm_head'], precision_recovery=PrecisionRecovery.NONE,
                        act_quant_granularity=QuantGranularity.PER_TENSOR,
                        weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        ffn_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                               act_quant_dtype=msdtype.int8,
                               outliers_suppression=OutliersSuppressionType.NONE,
                               precision_recovery=PrecisionRecovery.NONE,
                               act_quant_granularity=QuantGranularity.PER_TOKEN,
                               weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        layer_policies = OrderedDict({r'.*\.feed_forward\..*': ffn_config})
    elif quant_type.lower() == 'awq-a16w4':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                        act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.AWQ,
                        opname_blacklist=['lm_head', 'lkv2kv'], weight_quant_granularity=QuantGranularity.PER_GROUP,
                        group_size=128)
        layer_policies = OrderedDict()
    elif quant_type.lower() == 'awq-a16w8':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.AWQ,
                        opname_blacklist=['lm_head', 'lkv2kv'])
    elif quant_type.lower() == 'gptq-perchannel':
        gptq_config = GPTQQuantConfig()
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                        act_quant_dtype=None, precision_recovery=PrecisionRecovery.GPTQ, algo_args=gptq_config,
                        opname_blacklist=['lm_head', 'lkv2kv'])
        layer_policies = OrderedDict()
    elif quant_type.lower() == 'gptq-pergroup':
        gptq_config = GPTQQuantConfig()
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.qint4x2,
                        algo_args=gptq_config, act_quant_dtype=None, precision_recovery=PrecisionRecovery.GPTQ,
                        weight_quant_granularity=QuantGranularity.PER_GROUP, opname_blacklist=['lm_head', 'lkv2kv'],
                        group_size=64)
        w2_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                              act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH)
        layer_policies = OrderedDict({r'.*\.feed_forward\.w2.*': w2_config,
                                      r'.*\.shared_experts.w2.*': w2_config})
    elif quant_type.lower() == 'smoothquant':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, outliers_suppression=OutliersSuppressionType.SMOOTH,
                        opname_blacklist=['lm_head', 'lkv2kv'])
        w2_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                              act_quant_dtype=msdtype.int8,
                              outliers_suppression=OutliersSuppressionType.NONE,
                              precision_recovery=PrecisionRecovery.NONE,
                              act_quant_granularity=QuantGranularity.PER_TOKEN,
                              weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        layer_policies = OrderedDict({r'.*\.w2.*': w2_config})
    elif quant_type.lower() == 'osl':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                        opname_blacklist=['lm_head', 'lkv2kv'])
        w2_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                              act_quant_dtype=msdtype.int8,
                              outliers_suppression=OutliersSuppressionType.NONE,
                              precision_recovery=PrecisionRecovery.NONE,
                              act_quant_granularity=QuantGranularity.PER_TOKEN,
                              weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        layer_policies = OrderedDict({r'.*\.w2.*': w2_config})
    elif quant_type.lower() == 'a16w8':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        opname_blacklist=['lm_head', 'lkv2kv'])
        layer_policies = OrderedDict()
    elif quant_type.lower() == 'a8dynw8':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8, act_quant_granularity=QuantGranularity.PER_TOKEN,
                        opname_blacklist=['lm_head', 'lkv2kv'])
        layer_policies = OrderedDict()
    else:
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
    ptq = PTQ(config=cfg, layer_policies=layer_policies)
    if 'awq' in quant_type.lower():
        # pylint: disable=protected-access
        ptq._config.weight_symmetric = False
    if 'smoothquant' in quant_type.lower():
        # pylint: disable=protected-access
        ptq._config.aclnn_quant_list = ["routed_experts.ffn.w_gate_hidden", "routed_experts.ffn.w1",
                                        "routed_experts.ffn.w3"]
    if 'gptq-pergroup' in quant_type.lower():
        # pylint: disable=protected-access
        ptq.layer_policies[r'.*\.feed_forward\.w2.*'].aclnn_quant_list = ["w2"]
        ptq.layer_policies[r'.*\.shared_experts.w2.*'].aclnn_quant_list = ["w2"]
    ptq._config.algorithm_cache_path = ""
    if quant_type.lower() == 'osl':
        # pylint: disable=protected-access
        ptq._config.always_use_fp_input_in_processer = True
        ptq._config.algorithm_cache_path = 'osl_cache'
    from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
    ptq.decoder_layer_types.append(DeepseekV3DecodeLayer)
    return ptq


def create_network(yaml_file, quant_type=None):
    """create_tokenizer"""
    config = MindFormerConfig(yaml_file)
    build_context(config)
    build_parallel_config(config)
    model_config = config.model.model_config
    model_config.parallel_config = config.parallel_config
    model_config.moe_config = config.moe_config
    auto_online_trans = config.auto_trans_ckpt
    print('='*50, f"if using auto_online_trans: {auto_online_trans}", flush=True)
    model_config = DeepseekV3Config(**model_config)

    with no_init_parameters():
        network = DeepseekV3ForCausalLM(model_config)
    if quant_type:
        ptq = create_ptq(quant_type, PTQMode.DEPLOY)
        ptq.apply(network)
        ptq.convert(network)

    if config.load_checkpoint:
        if auto_online_trans:
            model_parallelism = DeepseekV3WeightProcessor(config, network, quant_type is not None)
            model_parallelism.load_safetensors_shard(config.load_checkpoint)
            barrier()
        else:
            ms_model = Model(network)
            seq_length = model_config.seq_length
            input_ids = Tensor(shape=(model_config.batch_size, seq_length), dtype=ms.int32, init=initializer.One())
            infer_data = network.prepare_inputs_for_predict_layout(input_ids)
            transform_and_load_checkpoint(config, ms_model, network, infer_data, do_predict=True)

    tokenizer = LlamaTokenizerFast(config.processor.tokenizer.vocab_file,
                                   config.processor.tokenizer.tokenizer_file,
                                   unk_token=config.processor.tokenizer.unk_token,
                                   bos_token=config.processor.tokenizer.bos_token,
                                   eos_token=config.processor.tokenizer.eos_token,
                                   fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    #pylint: disable=C0301
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\\n\\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' in message %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if message['content'] is none %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{'<｜Assistant｜>' + message['content'] + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and 'tool_calls' not in message %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}"
    return tokenizer, network
