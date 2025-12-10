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
"""Test Qwen3 PTQ quantization for mindone models."""

import os
import sys
import json
from collections import OrderedDict

import pytest
from mindspore import dtype as msdtype

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindone")))

# pylint: disable=wrong-import-position
from mindspore_gs.ptq import PTQConfig
from mindspore_gs.ptq.ptq_config import OutliersSuppressionType
from mindspore_gs.ptq.utils import QuantType
from mindspore_gs.ptq.models import AutoQuantForCausalLM
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode

from ptq_model_tester import VLPTQTester


class Qwen3VLPTQTester(VLPTQTester):
    """Qwen3 PTQ Tester for mindone models"""

    def create_ptq_config(self, mode, backend):
        """Create PTQ configuration for mindone models"""
        cfg = PTQConfig(mode=mode, backend=backend,
                        weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression= OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                        opname_blacklist=['lm_head', "down_proj", "linear_fc2", "merger"]) #
        layer_policies = OrderedDict()
        return cfg, layer_policies

    def calibrate_model(self):
        """Calibrate GLM4v model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create GLM4v processor...")

        print("Create calibration datasets...")
        ds = self.create_calib_datasets()

        print("Create PTQ config...")
        cfg, layers_policy = self.create_ptq_config(
            PTQMode.QUANTIZE, BackendTarget.ASCEND
        )

        print("Create GLM4v model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        calibrate_options = {
            'algorithm_cache_path': {'osl': 'osl_cache'},
            'always_use_fp_input_in_processer': True,
            'skip_offload_in_processing': True,
        }
        print("Start calibration...")
        model.calibrate(cfg, layers_policy, datasets=ds, **calibrate_options)

        print(f"Save quantized model to {self.output_dir}")
        model.save_quantized(self.output_dir, backend=BackendTarget.ASCEND)

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
            'model.layers.0.self_attn.q_proj.weight': QuantType.W8A8.value,
            'model.layers.0.self_attn.k_proj.input_scale': QuantType.W8A8.value,
            'model.layers.0.self_attn.v_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.0.self_attn.o_proj.weight': QuantType.W8A8.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        print("quant description test success.")
        return True

    def get_golden(self) -> tuple[str, str]:
        """Get golden reference for comparison"""
        return "<think>Got it, let's describe the image. " + \
               "First, the main subject is a small cat-like animal, maybe a Pallas cat or"


def run_qwen3_accuracy():
    """Run Qwen3VL PTQ test"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "/home/workspace/mindspore_dataset/weight/Qwen3-VL-8B-Instruct/"
    dataset_path = "/home/workspace/mindspore_dataset/textvqa"
    output_dir = os.path.join(cur_dir, "qwen3vl-quant")
    test_image_url = "/home/workspace/mindspore_dataset/images/pipeline-cat-chonk.jpeg"

    tester = Qwen3VLPTQTester(model_path,
                            dataset_path,
                            output_dir, test_image_url)
    tester.run_full_test()


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_qwen3_accuracy():
    """
    Feature: test ptq quantization for qwen3 model in mindone.
    Description: apply axwx on qwen3 language part and check the generated text.
    Expectation: generated text is good.
    """
    run_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_qwen3_accuracy.py")
    return_code = os.system(
        f"GSLOG=1 python {run_file} > test_qwen3_accuracy.log"
    )
    if return_code != 0:
        with open("./test_qwen3_accuracy.log", "r", encoding="utf-8") as log_file:
            for line in log_file:
                print(line, flush=True)
    assert return_code == 0

if __name__ == "__main__":
    run_qwen3_accuracy()
