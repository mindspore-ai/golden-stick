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
"""test interfaces of post training quant."""
import argparse
import os
import sys
import shutil
from collections import OrderedDict
import numpy as np

import mindspore as ms
from mindspore.communication import get_rank
from mindspore import dtype as msdtype
from mindspore.nn.utils import no_init_parameters
from mindspore import Model, Tensor, dataset
from mindspore.common import initializer

from mindformers.trainer.utils import transform_and_load_checkpoint
from mindformers import MindFormerConfig, build_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast

from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq.ptq import PTQ
from mindspore_gs.common.utils import offload_network
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFDSV3Helper
from mindspore_gs.ptq import PTQConfig, PTQMode, OutliersSuppressionType, \
                             QuantGranularity, PrecisionRecovery, GPTQQuantConfig


def create_ds(network_helper, ds_path, ds_type_, approach, tokenizer_):
    """Create datasets."""
    if approach in ['awq-a16w8', 'awq-a16w4', 'smoothquant', 'dsquant', 'a16w8', 'a8dynw8', 'gptq-prechannel',
                    'gptq-pergroup', 'osl']:
        if not ds_path:
            raise ValueError(f"Please provide dataset_path when approach is {approach}.")
        if not ds_type_:
            raise ValueError(f"Please provide dataset_type when approach is {approach}.")
        bs_ = network_helper.get_spec('batch_size')
        seq_ = network_helper.get_spec('seq_length')
        max_decode_length = network_helper.get_spec('max_decode_length')
        ignore_token_id = network_helper.get_spec('ignore_token_id')
        ds = get_datasets(ds_type_, ds_path, "train", bs_, seq_, max_decode_length, tokenizer_, ignore_token_id,
                          1, False, n_samples=50)
        return ds
    return None


def create_ptq(quant_type: str, quant_mode: PTQMode):
    """create_ptq"""
    if quant_type.lower() == 'gptq-pergroup':
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
        ffn_config = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                               act_quant_dtype=msdtype.int8,
                               outliers_suppression=OutliersSuppressionType.NONE,
                               precision_recovery=PrecisionRecovery.NONE,
                               act_quant_granularity=QuantGranularity.PER_TOKEN,
                               weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        layer_policies = OrderedDict({r'.*\.feed_forward\..*': ffn_config})
    elif quant_type.lower() == 'a16w8':
        cfg = PTQConfig(mode=quant_mode, backend=BackendTarget.ASCEND, weight_quant_dtype=msdtype.int8,
                        opname_blacklist=['lm_head', 'lkv2kv'])
        layer_policies = OrderedDict()
    else:
        raise RuntimeError(f'Input unsupported quant type: {quant_type}.')
    ptq = PTQ(config=cfg, layer_policies=layer_policies)
    if 'smoothquant' in quant_type.lower():
        # pylint: disable=protected-access
        ptq._config.aclnn_quant_list = ["routed_experts.ffn.w_gate_hidden", "routed_experts.ffn.w1",
                                        "routed_experts.ffn.w3"]
    if 'gptq-pergroup' in quant_type.lower():
        # pylint: disable=protected-access
        ptq.layer_policies[r'.*\.feed_forward\.w2.*'].aclnn_quant_list = ["w2"]
        ptq.layer_policies[r'.*\.shared_experts.w2.*'].aclnn_quant_list = ["w2"]
    # pylint: disable=protected-access
    ptq._config.algorithm_cache_path = {}
    from research.deepseek3.deepseek3_model_infer import DeepseekV3DecodeLayer
    ptq.decoder_layer_types.append(DeepseekV3DecodeLayer)
    return ptq


def create_deepseek_network(config, quant_type=None):
    """create deepseek network"""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from research.deepseek3.deepseek3_config import DeepseekV3Config
    from research.deepseek3.deepseek3 import DeepseekV3ForCausalLM
    from deepseekv3_weight_processor import DeepseekV3WeightProcessor

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


def quant_deepseekv3(config_path_, fp16_ckpt_path_, output_dir_, quant_algo_, dataset_path_, ds_type_, example):
    """PTQ quant to quant deepseek"""
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    os.environ['MS_ENABLE_LCCL'] = 'off'
    os.environ['HCCL_OP_EXPANSION_MODE'] = 'AIV'
    os.environ['MS_DEV_RUNTIME_CONF'] = 'parallel_dispatch_kernel:True'
    os.environ['HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT'] = 'TRUE'
    os.environ['MS_ALLOC_CONF'] = 'enable_vmm:True'
    os.environ['MS_PARALLEL_DISPATCH_NUM'] = '4'
    os.environ['MS_ENABLE_SYNC_COPY_INPUT'] = '1'
    os.environ['FORCE_EAGER'] = 'true'
    config_path_ = os.path.join(config_path_, 'predict_deepseek_r1_671b_calibrate.yaml')

    config = MindFormerConfig(config_path_)
    config.load_checkpoint = fp16_ckpt_path_
    config.output_dir = output_dir_
    config.processor.tokenizer.vocab_file = os.path.join(fp16_ckpt_path_, 'tokenizer.json')
    config.processor.tokenizer.tokenizer_file = os.path.join(fp16_ckpt_path_, 'tokenizer.json')
    if quant_algo_ == 'smoothquant':
        config.seq_length = 4096
        config.qkv_concat = True
    elif quant_algo_ == 'a16w8':
        config.seq_length = 4096
        config.qkv_concat = False
    elif quant_algo_ == 'gptq-pergroup':
        config.seq_length = 2048
        config.qkv_concat = False
    helper = MFDSV3Helper(config_path_)
    tokenizer, network = create_deepseek_network(config)

    input_ids = tokenizer(example)['input_ids']
    outputs = network.generate(input_ids, max_length=1024, do_sample=False, top_k=3, top_p=1, max_new_tokens=100)

    ptq = create_ptq(quant_algo_, PTQMode.QUANTIZE)
    dataset.config.set_numa_enable(False)
    ds = create_ds(helper, dataset_path_, ds_type_, quant_algo_, tokenizer)

    ptq.apply(network, helper, datasets=ds)
    network.phase = "quant_convert"
    ptq.convert(network)
    try:
        rank_id = get_rank()
    except RuntimeError:
        rank_id = 0
    save_path = os.path.join(config.output_dir, f"rank_{rank_id}")
    os.makedirs(save_path, exist_ok=True)
    ms.save_checkpoint(network.parameters_dict(), os.path.join(save_path, f"{quant_algo_}"),
                       choice_func=lambda x: "key_cache" not in x and "value_cache" not in x and "float_weight" not in x,
                       format="safetensors")
    offload_network(network)
    os.environ.pop('FORCE_EAGER', None)
    return outputs[0]


def eval_deepseekv3(config_path_, fp16_ckpt_path_, quant_ckpt_path_, quant_algo_, example):
    """eval llama2 by float ckpt and int ckpt"""
    ms.set_context(mode=0)
    os.environ.pop('FORCE_EAGER', None)
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    config_path_ = os.path.join(config_path_, 'predict_deepseek_r1_671b_evaluate.yaml')
    config = MindFormerConfig(config_path_)
    config.load_checkpoint = quant_ckpt_path_
    config.auto_trans_ckpt = False
    config.processor.tokenizer.vocab_file = os.path.join(fp16_ckpt_path_, 'tokenizer.json')
    config.processor.tokenizer.tokenizer_file = os.path.join(fp16_ckpt_path_, 'tokenizer.json')
    if quant_algo_ == 'smoothquant':
        config.context.mode = 0
        config.seq_length = 4096
        config.qkv_concat = True
        config.model.model_config.quantization_config.quant_method = 'smoothquant'
    elif quant_algo_ == 'a16w8':
        config.context.mode = 1
        config.seq_length = 4096
        config.qkv_concat = False
    elif quant_algo_ == 'gptq-pergroup':
        config.context.mode = 0
        config.seq_length = 2048
        config.qkv_concat = False
        config.model.model_config.quantization_config.quant_method = 'gptq-pergroup'
    tokenizer, network = create_deepseek_network(config, quant_algo_)
    os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = "PagedAttention"
    input_ids = tokenizer(example)['input_ids']
    outputs = network.generate(input_ids, max_length=1024, do_sample=False, top_k=3, top_p=1, max_new_tokens=100)
    rank_id = get_rank()
    if rank_id == 0 and os.path.exists(quant_ckpt_path_):
        shutil.rmtree(quant_ckpt_path_)

    return outputs[0]


def print_output(qoutput_, foutput_):
    print(f"qoutput: {qoutput_}", flush=True)
    print(f"foutput: {foutput_}", flush=True)
    print(f"First not equal index {np.min(np.where((qoutput_ - foutput_) != 0))}", flush=True)


def ptq_deepseek_predict_2stage(config_path_, fp16_ckpt_path_,
                                quant_ckpt_path_, quant_algo_, dataset_path_, ds_type_):
    """ptq_llama2_predict_2stage"""
    example = "介绍下北京故宫"
    foutput = quant_deepseekv3(config_path_, fp16_ckpt_path_, quant_ckpt_path_,
                               quant_algo_, dataset_path_, ds_type_, example)
    qoutput = eval_deepseekv3(config_path_, fp16_ckpt_path_, quant_ckpt_path_, quant_algo_, example)
    if quant_algo_ == 'a16w8':
        ret = np.allclose(qoutput[:32], foutput[:32], 0, 0)
    elif quant_algo_ == 'smoothquant':
        ret = np.allclose(qoutput[:6], foutput[:6], 0, 0)
    elif quant_algo_ == 'gptq-pergroup':
        ret = np.allclose(qoutput[:22], foutput[:22], 0, 0)
    else:
        assert False
    if not ret:
        print_output(qoutput, foutput)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_parallel', '-m', type=int, default=1)
    parser.add_argument('--quant_algo', '-a', type=str, required=True)
    uargs = parser.parse_args()
    model_parallel = uargs.model_parallel
    quant_algo = uargs.quant_algo
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    if quant_algo == 'a16w8':
        ds_type = 'boolq'
        dataset_path = os.path.join(cur_dir, f'../../../data/boolq-dataset/dev.jsonl')
    elif quant_algo == 'smoothquant':
        ds_type = 'ceval'
        dataset_path = os.path.join(cur_dir, f'../../../data/ceval-dataset/dev')
    elif quant_algo == 'gptq-pergroup':
        ds_type = 'calibrate'
        dataset_path = os.path.join(cur_dir, f'../../../data/calibrate-dataset/calibrate.jsonl')

    if model_parallel == 4:
        config_path = os.path.join(cur_dir, "../../../data/test_deepseek")
        fp16_ckpt_path = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-bf16"
        quant_ckpt_path = os.path.join(cur_dir, f"output/parallelDeepSeek-quant-4p-{quant_algo}")
        assert ptq_deepseek_predict_2stage(config_path, fp16_ckpt_path, quant_ckpt_path,
                                           quant_algo, dataset_path, ds_type)
    else:
        raise ValueError(f"Unsupported model_parallel: {model_parallel}")
