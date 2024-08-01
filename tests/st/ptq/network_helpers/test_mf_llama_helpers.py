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
import pytest
import numpy as np

from mindformers import MindFormerConfig, LlamaForCausalLM, LlamaTokenizer
from mindspore_gs.ptq.network_helpers.mf_net_helpers import MFLlama2Helper


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mf_llama_net_helper():
    """
    Feature: test float_forward in SQLinearWrapper.
    Description: Input fake data and check output of each float_forward.
    Expectation: Same with real linear output.
    """
    config_path = "../../../data/test_llama2/predict_llama2_13b_fp16_910b_1p.yaml"
    with pytest.raises(TypeError):
        MFLlama2Helper(1)
    cfg = MindFormerConfig(config_path)
    assert isinstance(MFLlama2Helper(cfg).mf_config, MindFormerConfig)
    helper = MFLlama2Helper(config_path)
    assert isinstance(MFLlama2Helper(cfg).mf_config, MindFormerConfig)

    network = helper.create_network()
    assert isinstance(network, LlamaForCausalLM)

    with pytest.raises(TypeError):
        helper.get_spec(1)
    assert helper.get_spec('batch_size') == 1
    assert helper.get_spec('seq_length') == 1024
    # pylint: disable=singleton-comparison
    assert helper.get_spec('use_flash_attention') == False

    tokenizer = helper.create_tokenizer()
    assert isinstance(tokenizer, LlamaTokenizer)
    input_ids = tokenizer.encode('Hello', add_special_tokens=True)
    assert isinstance(input_ids, List)

    with pytest.raises(TypeError, match="Type of mf_network should be "):
        helper.generate(1, input_ids, 1)
    with pytest.raises(TypeError, match="Type of input_ids should be "):
        helper.generate(network, 1, 1)
    with pytest.raises(TypeError, match="Type of max_new_tokens should be "):
        helper.generate(network, input_ids, '1')
    helper.generate(network, input_ids, 2)

    with pytest.raises(TypeError, match="Type of input_ids should be "):
        helper.assemble_inputs(1)
    inputs = helper.assemble_inputs(np.ones((1, 2), dtype=np.int32))
    assert isinstance(inputs, tuple)
    network(*inputs)
