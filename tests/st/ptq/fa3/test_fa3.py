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

"""
Test module for FA3 (Fused Attention 3) quantization functionality.

This module contains comprehensive tests for FA3 quantization applied to DeepseekV3 models.
It includes tests for:
- FA3 quantization configuration and application
- Model inference with quantized weights
- Weight processing and conversion
- Performance benchmarking of quantized models
- Integration with MindSpore and MindFormers frameworks

Note: FA3 quantization focuses on optimizing attention mechanisms and linear layers
for improved inference performance while maintaining model accuracy.
"""

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Union

from numpy.linalg import norm
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mindformers")))
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import dataset
from mindspore.nn.utils import no_init_parameters

from mindformers import MindFormerConfig
from mindformers.core.context.build_context import build_context, set_context
from mindformers.core.parallel_config import build_parallel_config
from mindformers.models import build_tokenizer
from mindformers.models.build_config import get_model_config
from mindformers.models.deepseek3.modeling_deepseek_v3_infer import InferenceDeepseekV3ForCausalLM
from mindformers.pipeline import pipeline
from mindformers.tools.utils import get_real_rank, set_strategy_save_path
from mindformers.models.utils import jit
from mindformers.generation.text_generator import GenerationMixin
from mindformers.models.llama.llama_tokenizer_fast import LlamaTokenizerFast
from research.deepseek3.deepseek3 import DeepseekV3ForCausalLM
from research.deepseek3.deepseek3_config import DeepseekV3Config

from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFDSV3Helper
from mindspore_gs.common import logger
from mindspore_gs.datasets import get_datasets
from mindspore_gs.ptq.faquant.faquant import FA3Config, FA3

from tests.st.test_utils import get_available_port
from deepseekv3_weight_processor_fa3 import DeepseekV3WeightProcessor

class FA3GenerationMixin(GenerationMixin):
    """For get hidden states."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decode_res = None
        self.decode_count = 0
    def _incremental_infer_mcore(self,
                                 model_inputs: dict,
                                 prefill,
                                 gather_decode=True):
        r"""
        mcore model forward for incremental infer.

        Args:
            model_inputs: infer model inputs.
            prefill: flag to distinguish prefill and decode.
            gather_decode: whether to gather decode logits.

        Returns:
            res: the output logits.

        """
        # Claim the first graph
        if prefill:
            self.phase = "prefill"
            if self._pre_set_phase:
                self.phase = f"prefill_{self._pre_set_phase}"
            # In dynamic shape scenarios, only the first execution of the prefill process will trigger this.
            if self._exec_add_flags:
                self.add_flags_custom_mcore(is_prefill=True)
            self.detailed_latency.start_predict_timer()
            # pylint: disable=E1102
            res, _ = self(
                **model_inputs,
            )
            self.phase = "increment"
            # first iter done, go to other iters, in dynamic shape scenarios, only the first execution
            # of the increment process will trigger this.
            if self._exec_add_flags:
                self.add_flags_custom_mcore(is_prefill=False)
                self._exec_add_flags = False

        else:
            # slice model inputs for incremental infer
            if self._pre_set_phase:
                self.phase = f"increment_{self._pre_set_phase}"
            self.detailed_latency.start_predict_timer()
            # pylint: disable=E1102
            if self.decode_count == 0:
                res, self.decode_res = self(
                    **model_inputs,
                )
            else:
                res, _ = self(
                    **model_inputs,
                )
            q_seq_lens = model_inputs.get("q_seq_lens", None)
            if gather_decode and q_seq_lens is not None:
                if q_seq_lens.max() > 1 and q_seq_lens.sum() == res.shape[0]:
                    res = self.gather(res, mint.cumsum(q_seq_lens, dim=0) - 1, 0)
            self.decode_count += 1
        return res

class FA3InferenceDeepseekV3ForCausalLM(FA3GenerationMixin, InferenceDeepseekV3ForCausalLM):
    """For get hidden states."""
    @jit
    def construct(
            self,
            input_ids,
            hidden_states=None,
            positions=None,
            batch_valid_length=None,
            context_lens_tensor=None,
            q_seq_lens=None,
            block_tables=None,
            slot_mapping=None,
            attention_mask=None,
            attn_metadata=None,
            attn_padding_idx=None,
            attn_unpadding_idx=None,
            ffn_padding_idx=None,
            ffn_unpadding_idx=None,
            key_cache=None,
            value_cache=None
    ):
        r"""
        model forward.

        Args:
            input_ids: input ids.
            hidden_states: hidden states.
            positions: position ids.
            batch_valid_length: actual seq length.
            context_lens_tensor: computed key value length.
            q_seq_lens: query sequence lengths.
            block_tables: Store mapping tables for each sequence.
            slot_mapping : Token cache physical slot index.
            attention_mask: attentino mask used for fa or pa.
            attn_metadata: attention metadata.
            attn_padding_idx: Indices mapping positions in attention output sequence to original token positions,
                used for padding attention output to fixed size.
            attn_unpadding_idx: Indices mapping valid tokens in padded attention output sequence to
                their original positions, used for removing padding in attention output.
            ffn_padding_idx: Indices mapping positions in MoE output sequence to flattened valid token positions,
                used for padding MoE output to fixed size.
            ffn_unpadding_idx: Indices mapping valid tokens in padded MoE output sequence to their original positions,
                used for removing padding in MoE output.
            key_cache: key cache for incremental inference.
            value_cache: value cache for incremental inference.

        Returns:
            logits: the output logits.

        """
        hidden_states = self.model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            positions=positions,
            batch_valid_length=batch_valid_length,
            context_lens_tensor=context_lens_tensor,
            q_seq_lens=q_seq_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
            attn_padding_idx=attn_padding_idx,
            attn_unpadding_idx=attn_unpadding_idx,
            ffn_padding_idx=ffn_padding_idx,
            ffn_unpadding_idx=ffn_unpadding_idx,
            key_cache=key_cache,
            value_cache=value_cache,
        )
        output = self.model.pre_gather_func(hidden_states, context_lens_tensor, batch_valid_length)
        # Return logits.
        logits = self.model.output_layer(output)
        logits = self.model.cast(logits.squeeze(0), mstype.float32)
        return logits, hidden_states

def create_network(yaml_file):
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

    if config.load_checkpoint:
        if auto_online_trans:
            model_parallelism = DeepseekV3WeightProcessor(config, network, False)
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

def create_ds(network_helper, ds_path, ds_type, tokenizer_):
    """Create datasets."""
    start_time = time.time()
    bs_ = network_helper.get_spec('batch_size')
    seq_ = network_helper.get_spec('seq_length')
    max_decode_length = network_helper.get_spec('max_decode_length')
    ignore_token_id = network_helper.get_spec('ignore_token_id')
    ds = get_datasets(ds_type, ds_path, "train", bs_, seq_, max_decode_length, tokenizer_, ignore_token_id,
                      1, False, n_samples=200)
    logger.info(f'Create datasets cost time is {time.time() - start_time} s.')
    return ds

def load_json_fa3(path: Path) -> Dict[str, Any]:
    """Load data from json file."""
    with path.open('rt', encoding='utf-8') as f:
        return json.load(f)

def save_json_fa3(path: Path, data: Dict[str, Any]) -> None:
    """Save data to json file."""
    with path.open('wt', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_states_for_a8w8(json_path: Union[str, Path]) -> None:
    """Configure the environmen for a8w8 infer."""
    json_path = Path(json_path)
    bak_path = json_path.with_suffix(json_path.suffix + '.bak')
    assert bak_path.exists()
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(json_path), suffix='.tmp') as tmp:
        shutil.copy(bak_path, tmp.name)
        shutil.move(tmp.name, json_path)
    data = load_json_fa3(json_path)
    data_filtered = {k: v for k, v in data.items() if 'FAQuant' not in str(v)}
    save_json_fa3(json_path, data_filtered)

    print(f"[INFO] key-values with 'FAQuant' has been deleted, bak file is {bak_path}")

def update_states_for_fa3(json_path: Union[str, Path]) -> None:
    """Configure the environmen for a8w8-fa3 infer."""
    json_path = Path(json_path)
    bak_path = json_path.with_suffix(json_path.suffix + '.bak')
    assert bak_path.exists()
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(json_path), suffix='.tmp') as tmp:
        shutil.copy(bak_path, tmp.name)
        shutil.move(tmp.name, json_path)
    print(f"[INFO] The original fa3 configuration file has been restored using the backup file {bak_path}")

def replace_fa3_params(model_path):
    """Configure the environmen for a8w8-fa3 infer."""
    para_file_path = os.path.join(model_path,
                                  "quant_model_weight_w8a8_dynamic.safetensors.index.json")
    with open(para_file_path, "r", encoding='utf-8') as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    file_to_params = {}
    current_dir = os.getcwd()
    for param_name, filename in weight_map.items():
        is_fa3_params = "self_attn.fa_q.scale" in param_name or \
                        "self_attn.fa_k.scale" in param_name or \
                        "self_attn.fa_v.scale" in param_name or \
                        "self_attn.fa_q.offset" in param_name or \
                        "self_attn.fa_k.offset" in param_name or \
                        "self_attn.fa_v.offset" in param_name
        if "model.layers" in param_name and is_fa3_params:
            if filename not in file_to_params:
                file_to_params[filename] = []
            file_to_params[filename].append(param_name)
    for filename, _ in file_to_params.items():
        filepath = os.path.join(model_path, filename)
        if not os.path.exists(filepath):
            continue
        param_dict = ms.load_checkpoint(filepath, format="safetensors")
        count = 0
        for param_name in param_dict:
            match = re.search(r"model\.layers\.([0-0])\.self_attn\.fa_(q|k|v)\.(scale|offset)", param_name)
            if not match:
                continue
            count += 1
            ori_value = param_dict[param_name]
            layer_idx, qkv, suffix = match.groups()
            prefix = qkv if qkv == 'q' else 'kv'
            npy_filename = f"network.model.layers.{layer_idx}.{prefix}_{suffix}s.npy"
            npy_path = os.path.join(current_dir, "fa3_params", "rank_0", "perhead", npy_filename)
            new_value = np.load(npy_path)
            if prefix == "q":
                new_np = np.squeeze(new_value, axis=0)
                new_tensor = ms.Tensor(new_np, dtype=ori_value.dtype)
            elif prefix == "kv":
                scalar = new_value.flat[0]
                new_tensor = ms.Tensor([[scalar]], dtype=ori_value.dtype)
            else:
                assert False
            param_dict[param_name] = ms.Parameter(new_tensor, name=ori_value.name,
                                                  requires_grad=ori_value.requires_grad)
        if count > 0:
            print("Save fa3 modified ckpt: ", filepath)
            target_dir = os.path.dirname(filepath)
            fd = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix='.safetensors', dir=target_dir, prefix='._tmp')
                os.fchmod(fd, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                os.close(fd)
                fd = None
                ms.save_checkpoint(param_dict, tmp_path, format="safetensors")
                os.replace(tmp_path, filepath)
            except Exception:
                if fd is not None:
                    os.close(fd)
                raise

def calculate(a, b):
    """Calculate cosine similarity."""
    a = a.asnumpy().ravel().astype(np.float32)
    b = b.asnumpy().ravel().astype(np.float32)
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def config_pre(yaml_file):
    """Config."""
    config = MindFormerConfig(yaml_file)
    build_context(config)
    build_parallel_config(config)
    rank_id = get_real_rank()
    config.rank_id = rank_id
    config.remove_redundancy = config.get('remove_redundancy', False)
    config.load_checkpoint = config.pretrained_model_dir
    config.model.model_config.checkpoint_name_or_path = None
    set_strategy_save_path(config.parallel)
    if config.get("pretrained_model_dir", None):
        config.model.pretrained_model_dir = config.pretrained_model_dir
    if config.get("generation_config", None):
        config.model.generation_config = config.generation_config
    return config

def infer(yaml_file):
    """Infer with mcore."""
    config = config_pre(yaml_file)
    tokenizer = build_tokenizer(config.get("processor", {}).get("tokenizer", None),
                                use_legacy=config.get("use_legacy", True),
                                pretrained_model_dir=config.get("pretrained_model_dir", None),
                                trust_remote_code=config.get("trust_remote_code", False))
    with no_init_parameters():
        default_args = {"parallel_config": config.parallel_config, "moe_config": config.moe_config}
        moe_config = default_args.pop('moe_config', {})
        default_args.update(moe_config)
        model_config = get_model_config(config.model, default_args=default_args)
        network = FA3InferenceDeepseekV3ForCausalLM(model_config)

    network.load_weights(config.load_checkpoint)
    network.init_parameters_data()
    pipe = pipeline(task='text_generation', model=network, tokenizer=tokenizer, is_full_config=True, adapter_id=None)
    pipe.model.model.return_hidden_states = True
    output = pipe("介绍下北京故宫", top_k=1)
    print("output: ", output)
    return pipe.model.decode_res

def test_fa3():
    """
    Feature: Main test case for fa3.
    Description: Main test case for fa3.
    Comparing the cosine similarity between the two inferences of the w8a8 and w8a8-fa3 models,
    the w8a8-fa3 inference requires the use of the fa3 calibration algorithm to obtain fa3 calibration parameters.
    Expectation: Cos similarity between original w8a8 and w8a8-fa3 results is supposed to be greater than 99%.
    """
    os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = "QuantBatchMatmul"
    os.environ.pop("MS_INTERNAL_ENABLE_NZ_OPS", None)
    ms.context.set_context(pynative_synchronize=True)
    yaml_path = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8-ATB-mcore-fa3/predict_deepseek3_671b.yaml"
    model_path = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8-ATB-mcore-fa3"
    index_path = os.path.join(model_path, "quant_model_description_w8a8_dynamic.json")
    update_states_for_a8w8(index_path)
    a8w8_hs = infer(yaml_path)
    # FA3
    os.environ['MS_JIT'] = "0"
    os.environ['ENFORCE_EAGER'] = "True"
    set_context(use_legacy=True)
    cal_yaml = "/home/workspace/mindspore_dataset/weight/DeepSeek-R1-W8A8-ATB-mcore-fa3/" \
            "predict_deepseek_r1_671b_calibrate.yaml"
    mfconfig = MindFormerConfig(cal_yaml)
    helper = MFDSV3Helper(cal_yaml)
    logger.info("Creating network...")
    tokenizer, network = create_network(cal_yaml)
    export_params_path = "./fa3_params"
    cfg = FA3Config(export_params_path=export_params_path, dsk_config=mfconfig.model.model_config)
    fa3 = FA3(config=cfg)
    dataset.config.set_numa_enable(False)
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    cal_datapath = os.path.join(cur_dir, "../../../data/calibrate-dataset/calibrate.jsonl")
    datasets = create_ds(helper, cal_datapath, "calibrate", tokenizer_=tokenizer)
    logger.info("Create Network End.")
    logger.info("Running FA3...")
    fa3.observe(network, helper, datasets)
    logger.info("Running FA3 Calculate End.")
    os.environ.pop('MS_JIT', None)
    os.environ.pop('ENFORCE_EAGER', None)
    set_context(use_legacy=False)
    update_states_for_fa3(index_path)
    replace_fa3_params(model_path)
    a8w8_fa3_hs = infer(yaml_path)
    res = calculate(a8w8_hs, a8w8_fa3_hs)
    assert res > 0.99, (
        f"Single-layer output cosine similarity is {res}, "
        f"a8w8_hs={a8w8_hs.asnumpy()}, a8w8_fa3_hs={a8w8_fa3_hs.asnumpy()}"
    )

def invoke_parallel(entry_func, **entry_kwargs):
    """Start 2 parallel workers, and call the entry function."""
    run_file = os.path.abspath(__file__)
    port = get_available_port()
    os.system(f'kill -9 $(lsof -i:{port} | ' + "awk '{print $2}')")
    return_code = os.system(
        f'msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 '
        f'--master_port={port} --join=True --log_dir=./test_fa3_2p_logs '
        f'python {run_file} --entry {entry_func.__name__} --entry_kwargs {repr(json.dumps(entry_kwargs))}'
    )
    if return_code != 0:
        for i in os.listdir('test_fa3_2p_logs'):
            if i.endswith('.log'):
                filepath = os.path.join('test_fa3_2p_logs', i)
                with open(filepath, 'r', encoding='utf-8') as f:
                    print(f'===================={filepath}====================')
                    print(f.read())
    os.system(f'kill -9 $(lsof -i:{port} | ' + "awk '{print $2}')")
    os.system('rm -rf test_fa3_2p_logs')
    os.system('rm -rf fa3_params')
    assert return_code == 0

def parallel_args():
    """Parse args for parallel runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry')
    parser.add_argument('--entry_kwargs', type=json.loads, default='{}')
    args = parser.parse_args()
    return args

def parallel_runner(args):
    """Call the entry function."""
    globals()[args.entry](**args.entry_kwargs)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_fa3_dual():
    """
    Feature: Comparing the cosine similarity between the two inferences of the w8a8 and w8a8-fa3 models.
    Description: Comparing the cosine similarity between the two inferences of the w8a8 and w8a8-fa3 models,
    the w8a8-fa3 inference requires the use of the fa3 calibration algorithm to obtain fa3 calibration parameters.
    Expectation: Cos similarity between original w8a8 and w8a8-fa3 results is supposed to be greater than 99%.
    """
    invoke_parallel(test_fa3)

if __name__ == '__main__':
    uargs = parallel_args()
    parallel_runner(uargs)
