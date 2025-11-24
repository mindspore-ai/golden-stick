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

"""UT: Verify HF safetensors mapping for Qwen3 save path.

Feature: HF name mapping correctness in save_quantized
Description: Build DummyNetwork returning mcore names (including output_layer.weight),
mock MFModel init and QWen3 quant-type collection to avoid real model init/graph.
Expectation: Generated quantization_description.json and model.safetensors.index.json
should not contain any 'output_layer.*' keys.
"""

import os
import sys
import json
import tempfile

from unittest.mock import patch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))
import mindspore as ms
from mindspore import Parameter, Tensor, nn

from mindspore_gs.ptq.models import AutoQuantForCausalLM
from mindspore_gs.ptq.models.mindformers_models.mf_model import MFModel
from mindspore_gs.ptq.basic_functions.safetensors_mgr import SafeTensorsMgr


# Construct DummyNetwork returning real-like Qwen3 parameter names (mcore)
class _DummyQwen3(nn.Cell):
    """Dummy network returning Qwen3-style mcore parameter names for save path verification."""
    def parameters_dict(self):
        p_embed = Parameter(Tensor(ms.numpy.randn(2, 3), ms.float32), name='embedding.word_embeddings.weight')
        p_out = Parameter(Tensor(ms.numpy.randn(4, 5), ms.float32), name='output_layer.weight')
        return {
            'embedding.word_embeddings.weight': p_embed,
            'output_layer.weight': p_out,
        }


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_hf_mapping_contains_lm_head_keys_when_output_present():
    """
    Feature: HF safetensors mapping for Qwen3
    Description: Use DummyNetwork with mcore names including output_layer.weight and mock
    _get_quant_type to run save_quantized without real model init; then read generated
    quantization_description.json and model.safetensors.index.json.
    Expectation: Both files contain mapped names, and do not include any 'output_layer.*' keys.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    calibrate_yaml = os.path.join(cur_dir, "calibrate_qwen3.yaml")

    # Stub MFModel.__init__ to avoid real mindformers initialization
    with patch.object(MFModel, '__init__', return_value=None):
        model = AutoQuantForCausalLM.from_pretrained(calibrate_yaml)
        orig_dir = tempfile.mkdtemp(prefix="qwen3-orig-")
        model._original_sf_path = orig_dir # pylint: disable=protected-access
        model.network = _DummyQwen3()
    tmpdir = tempfile.mkdtemp(prefix="qwen3-mapping-")

    quant_types = {
        'embedding.word_embeddings.weight': 'float',
        'output_layer.weight': 'W8A8',
    }

    with patch.object(model, '_get_quant_type', return_value=quant_types), \
         patch.object(SafeTensorsMgr, '_copy_original_files', side_effect=lambda o, s: os.makedirs(s, exist_ok=True)), \
         patch.object(SafeTensorsMgr, '_save_safetensors', return_value=None):
        model.save_quantized(tmpdir)

    # Find quantization_description.json
    desc_json_path = None
    for fn in os.listdir(tmpdir):
        if fn.endswith(".json") and "quantization_description" in fn:
            desc_json_path = os.path.join(tmpdir, fn)
            break
    assert desc_json_path and os.path.exists(desc_json_path), "quantization_description.json missing"

    with open(desc_json_path, "r", encoding="utf-8") as fp:
        desc = json.load(fp)

    # Assert mapping consistency: no raw output_layer.* leaks in description
    assert not any("output_layer." in k for k in desc.keys()), "output_layer.* must be mapped to lm_head.*"

    # Also check the safetensors index map exists and does not leak output_layer.*
    index_json_path = os.path.join(tmpdir, "model.safetensors.index.json")
    assert os.path.exists(index_json_path), "index.json should be created"
    with open(index_json_path, "r", encoding="utf-8") as fp:
        idx = json.load(fp)
    weight_map = idx.get("weight_map", {})
    assert not any("output_layer." in k for k in weight_map.keys()), "output_layer.* must not appear in index"
