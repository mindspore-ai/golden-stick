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
"""test razor attention search mode."""
import os
import sys
import pytest
import mindspore
from transformers import AutoConfig
# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mindone")))
from mindone.transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from mindspore_gs.common import BackendTarget
from mindspore_gs.sequence_compress.razor_attention import RAMode, RAConfig
from mindspore_gs.sequence_compress.razor_attention import RazorAttention as RA

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_razor_attention_retri_head():
    """
    Feature: razor attention search mode.
    Description: qwen3 0.6b model one layer scenario, using razor attention search mode to get retrieval head.
    Expectation: Get retrieval head for the first layer.
    """
    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    model_path = '/home/workspace/mindspore_ckpt/safetensors/Qwen3-0.6B'
    config = AutoConfig.from_pretrained(model_path)
    config.num_hidden_layers = 1
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        mindspore_dtype=mindspore.bfloat16,
        attn_implementation="flash_attention_2",
        config=config
    )

    ra_config= RAConfig(mode=RAMode.SEARCH_RETRIEVAL, backend=BackendTarget.ASCEND,\
                        echo_head_ratio=0.01, induction_head_ratio=0.14, retrieval_head_path='./head_dict.json')
    razor_compressor = RA(config=ra_config)
    result = razor_compressor.apply(model, is_saved=False)
    assert result[0] == [5]
