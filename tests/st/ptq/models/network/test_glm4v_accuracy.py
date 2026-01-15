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
"""Test GLM4v PTQ quantization for mindone models."""

import os
import sys
import json
from collections import OrderedDict

import numpy as np
import pytest
import mindspore as ms
from mindspore import dtype as msdtype

# pylint: disable=wrong-import-position
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindone")))

# pylint: disable=wrong-import-position
from mindspore_gs.ptq import PTQConfig
from mindspore_gs.ptq.ptq_config import OutliersSuppressionType
from mindspore_gs.ptq.utils import QuantType
from ptq_model_tester import VLPTQTester

import transformers
print('transformers.__version__:', transformers.__version__)


def convert_to_tensor(examples):
    """examples: dict[str, np.ndarray] -> dict[str, ms.Tensor]"""
    return {
        k: (ms.tensor(v, dtype=ms.int32) if isinstance(v, (np.ndarray, list)) and ms.tensor(v).dtype == ms.int64
            else ms.tensor(v) if isinstance(v, (np.ndarray, list))
            else v)
        for k, v in examples.items()
    }


class GLM4vPTQTester(VLPTQTester):
    """GLM4v PTQ Tester for mindone models"""

    def create_ptq_config(self, mode, backend):
        """Create PTQ configuration for mindone models"""
        osl_a8w8 = PTQConfig(mode=mode, backend=backend,
                             act_quant_dtype=msdtype.int8,
                             weight_quant_dtype=msdtype.int8,
                             outliers_suppression=OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                             opname_blacklist=['lm_head', 'merge'])
        layer_policies = OrderedDict({})
        return osl_a8w8, layer_policies

    def check_quant_description(self, quant_ckpt_path: str) -> bool:
        """Check if quantized checkpoint has proper description"""
        if not os.path.exists(quant_ckpt_path):
            print(f"{quant_ckpt_path} dose not exist.")
            return False
        desc_json_path = ""
        for file_name in os.listdir(quant_ckpt_path):
            if file_name.endswith(".json") and "quantization_description" in file_name:
                desc_json_path = os.path.join(quant_ckpt_path, file_name)
        if desc_json_path is None:
            print("No quant description json file.")
            return False
        with open(desc_json_path, "r", encoding="utf-8") as fp:
            desc_map = json.load(fp)

        def check(name, expect):
            cur = desc_map.get(name)
            ret = cur == expect
            if not ret:
                print(f"quant info of {name} should be {expect}, but got: {cur}.")
            return ret

        check_map = {
            'model.language_model.layers.0.self_attn.q_proj.weight': QuantType.W8A8.value,
            'model.language_model.layers.0.self_attn.k_proj.weight': QuantType.W8A8.value,
            'model.language_model.layers.0.mlp.gate_up_proj.weight': QuantType.W8A8.value,
            'model.visual.blocks.0.attn.qkv.weight': QuantType.W8A8.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        print("quant description test success.")
        return True

    def get_golden(self) -> tuple[str, str]:
        """Get golden reference for comparison"""
        return "ecurity.Xnaж$fdata-aos"


def run_glm4v_accuracy():
    """Run GLM4v PTQ test"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "/home/workspace/mindspore_dataset/weight/GLM-4.1V-9B-Thinking_1layer/"
    dataset_path = "/home/workspace/mindspore_dataset/textvqa"
    output_dir = os.path.join(cur_dir, "glm4v-quant")
    test_image_url = "/home/workspace/mindspore_dataset/images/pipeline-cat-chonk.jpeg"
    tester = GLM4vPTQTester(model_path,
                            dataset_path,
                            output_dir,
                            test_image_url)
    assert tester.run_full_test(), "GLM4v PTQ test failed!"

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_glm4v_accuracy():
    """
    Feature: test ptq quantization for glm4v model in mindone.
    Description: apply a16w4 on glm4v language part and check the generated text.
    Expectation: generated text is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_glm4v_accuracy.py")
    return_code = os.system(
        f"GSLOG=1 python {run_file} > test_glm4v_accuracy.log 2>&1"
    )
    if return_code != 0:
        with open("./test_glm4v_accuracy.log", "r", encoding="utf-8") as log_file:
            for line in log_file:
                print(line, flush=True)
    assert return_code == 0

if __name__ == "__main__":
    run_glm4v_accuracy()
