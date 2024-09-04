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
"""test interfaces of MFLlama2Helper."""

from typing import List
import os
import pytest
import numpy as np

from mindformers import MindFormerConfig, LlamaForCausalLM, LlamaTokenizer
from mindformers.modules.paged_attention_mgr import PagedAttentionMgr
from mindformers.experimental.llama_demo import ParallelLlamaForCausalLM
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper, MFParallelLlama2Helper

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_llama_net_helper_inputs():
    """
    Feature: test each function inputs of MFLlama2Helper class.
    Description: MFLlama2Helper class used to create LlamaForCausalLM network and provide network detail infos.
    Expectation: correct output of each function
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_1p.yaml"
    with pytest.raises(TypeError):
        MFLlama2Helper(1)
    cfg = MindFormerConfig(config_path)
    assert isinstance(MFLlama2Helper(cfg).mf_config, MindFormerConfig)
    helper = MFLlama2Helper(config_path)
    assert isinstance(MFLlama2Helper(cfg).mf_config, MindFormerConfig)

    with pytest.raises(TypeError):
        _ = helper.get_spec(1)

    tokenizer = helper.create_tokenizer()
    assert isinstance(tokenizer, LlamaTokenizer)
    input_ids = tokenizer.encode('Hello', add_special_tokens=True)
    assert isinstance(input_ids, List)

    network = helper.create_network()
    assert isinstance(network, LlamaForCausalLM)
    with pytest.raises(TypeError, match="Type of mf_network should be "):
        helper.generate(1, input_ids, 1)
    with pytest.raises(TypeError, match="Type of input_ids should be "):
        helper.generate(network, 1, 1)
    with pytest.raises(TypeError, match="Type of max_new_tokens should be "):
        helper.generate(network, input_ids, '1')
    helper.generate(network, input_ids, 2)

    with pytest.raises(TypeError, match="Type of input_ids should be "):
        helper.assemble_inputs(1)

    with pytest.raises(TypeError):
        _ = helper.get_decoder_layers(1)

    with pytest.raises(TypeError):
        _ = helper.get_linears(1)

    with pytest.raises(TypeError):
        _ = helper.get_page_attention_mgr("1")

    with pytest.raises(TypeError):
        helper.analysis_decoder_groups("1")

    with pytest.raises(TypeError):
        _ = helper.get_pre_layer(1)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_llama_net_helper():
    """
    Feature: test each function outputs of MFLlama2Helper class.
    Description: MFLlama2Helper class used to create LlamaForCausalLM network and provide network detail infos.
    Expectation: correct output of each function
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_1p.yaml"
    with pytest.raises(TypeError):
        MFLlama2Helper(1)
    cfg = MindFormerConfig(config_path)
    assert isinstance(MFLlama2Helper(cfg).mf_config, MindFormerConfig)
    helper = MFLlama2Helper(config_path)

    helper.mf_config.model.model_config.use_past = True
    helper.mf_config.model.model_config.block_size = 16
    network = helper.create_network()
    assert isinstance(network, LlamaForCausalLM)
    layers = helper.get_decoder_layers(network)
    assert isinstance(layers, List)
    assert len(layers) == 1
    _, layer = layers[0]
    qkv_concat, ffn_concat, linears = helper.get_linears(layer)
    assert qkv_concat is False
    assert ffn_concat is False
    assert isinstance(linears, List)
    paged_attention_mgr = helper.get_page_attention_mgr(layer)
    assert isinstance(paged_attention_mgr, PagedAttentionMgr)
    helper.analysis_decoder_groups(network)
    pre_layer = helper.get_pre_layer("root.model.layers.0")
    assert pre_layer is None

    assert helper.get_spec('batch_size') == 1
    assert helper.get_spec('seq_length') == 1024
    # pylint: disable=singleton-comparison
    assert helper.get_spec('use_flash_attention') == False

    inputs = helper.assemble_inputs(np.ones((1, 1024), dtype=np.int32))
    assert isinstance(inputs, tuple)
    network(*inputs)

# @pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_parallel_llama_net_helper_inputs_1p():
    """
    Feature: test each function inputs of MFParallelLlama2Helper class with msrun.
    Description: MFParallelLlama2Helper class used to create LlamaForCausalLM network and provide network detail infos.
    Expectation: correct output of each function
    """
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_mf_llama_helpers_1p_logs "
        f"pytest test_mf_llama_helpers.py::test_mf_parallel_llama_net_helper_inputs"
    )
    if return_code != 0:
        log_file = open("./test_mf_llama_helpers_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0

@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_parallel_llama_net_helper_inputs():
    """
    Feature: test each function inputs of MFParallelLlama2Helper class.
    Description: MFParallelLlama2Helper class used to create LlamaForCausalLM network and provide network detail infos.
    Expectation: correct output of each function
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    config_path = "../../../data/test_llama2/predict_parallelLlama2_13b_1p.yaml"
    with pytest.raises(TypeError):
        MFParallelLlama2Helper(1)
    cfg = MindFormerConfig(config_path)
    assert isinstance(MFParallelLlama2Helper(cfg).mf_config, MindFormerConfig)
    helper = MFParallelLlama2Helper(config_path)
    assert isinstance(MFParallelLlama2Helper(cfg).mf_config, MindFormerConfig)

    with pytest.raises(TypeError):
        helper.get_spec(1)

    tokenizer = helper.create_tokenizer()
    assert isinstance(tokenizer, LlamaTokenizer)
    input_ids = tokenizer.encode('Hello', add_special_tokens=True)
    assert isinstance(input_ids, List)

    network = helper.create_network()
    assert isinstance(network, ParallelLlamaForCausalLM)
    with pytest.raises(TypeError, match="Type of mf_network should be "):
        helper.generate(1, input_ids, 1)
    with pytest.raises(TypeError, match="Type of input_ids should be "):
        helper.generate(network, 1, 1)
    with pytest.raises(TypeError, match="Type of max_new_tokens should be "):
        helper.generate(network, input_ids, '1')
    helper.generate(network, input_ids, 2)

    with pytest.raises(TypeError, match="Type of input_ids should be "):
        helper.assemble_inputs(1)

    with pytest.raises(TypeError):
        _ = helper.get_decoder_layers(1)

    with pytest.raises(TypeError):
        _ = helper.get_linears(1)

    with pytest.raises(TypeError):
        _ = helper.get_page_attention_mgr("1")

    with pytest.raises(TypeError):
        helper.analysis_decoder_groups("1")

    with pytest.raises(TypeError):
        _ = helper.get_pre_layer(1)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_parallel_llama_net_helper_1p():
    """
    Feature: test each function of MFParallelLlama2Helper class with msrun.
    Description: MFParallelLlama2Helper class used to create LlamaForCausalLM network and provide network detail infos.
    Expectation: correct output of each function
    """
    os.system("kill -9 $(lsof -i:10926 | awk '{print $2}')")
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10926 --join=True --log_dir=./test_mf_llama_helpers_1p_logs "
        f"pytest test_mf_llama_helpers.py::test_mf_parallel_llama_net_helper"
    )
    if return_code != 0:
        log_file = open("./test_mf_llama_helpers_1p_logs/worker_0.log", "r", encoding="utf-8")
        for line in log_file:
            print(line, flush=True)
        log_file.close()

    assert return_code == 0

@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_parallel_llama_net_helper():
    """
    Feature: test each function of MFParallelLlama2Helper class.
    Description: MFParallelLlama2Helper class used to create ParallelLlamaForCausalLM network
                 and provide network detail infos.
    Expectation: correct output of each function.
    """
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
    ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
    config_path = "../../../data/test_llama2/predict_parallelLlama2_13b_1p.yaml"
    with pytest.raises(TypeError):
        MFParallelLlama2Helper(1)
    cfg = MindFormerConfig(config_path)
    assert isinstance(MFParallelLlama2Helper(cfg).mf_config, MindFormerConfig)
    helper = MFParallelLlama2Helper(config_path)
    assert isinstance(MFParallelLlama2Helper(cfg).mf_config, MindFormerConfig)

    network = helper.create_network()
    assert isinstance(network, ParallelLlamaForCausalLM)

    layers = helper.get_decoder_layers(network)
    assert isinstance(layers, List)
    assert len(layers) == 1
    _, layer = layers[0]
    qkv_concat, ffn_concat, linears = helper.get_linears(layer)
    assert qkv_concat is True
    assert ffn_concat is True
    assert isinstance(linears, List)
    paged_attention_mgr = helper.get_page_attention_mgr(layer)
    assert isinstance(paged_attention_mgr, PagedAttentionMgr)
    helper.analysis_decoder_groups(network)
    pre_layer = helper.get_pre_layer("root.model.layers.0")
    assert pre_layer is None

    assert helper.get_spec('batch_size') == 1
    assert helper.get_spec('seq_length') == 1024
    # pylint: disable=singleton-comparison
    assert helper.get_spec('use_flash_attention') == True

    inputs = helper.assemble_inputs(np.ones((1, 1024), dtype=np.int32))
    assert isinstance(inputs, tuple)
    #FIXME ccsszz, ApplyRotaryPosEmb position_ids=None
    # network(*inputs)
