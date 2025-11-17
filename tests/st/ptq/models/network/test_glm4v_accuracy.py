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
from typing import Optional
from collections import OrderedDict

import numpy as np
import pytest
import mindspore as ms
from mindspore import dtype as msdtype
from datasets import load_dataset

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindone")))
from mindone.transformers import AutoProcessor

# pylint: disable=wrong-import-position
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode)
from mindspore_gs.ptq.models import AutoQuantForCausalLM
from mindspore_gs.ptq.utils import QuantType


def convert_to_tensor(examples):
    """examples: dict[str, np.ndarray] -> dict[str, ms.Tensor]"""
    return {
        k: (ms.tensor(v, dtype=ms.int32) if isinstance(v, (np.ndarray, list)) and ms.tensor(v).dtype == ms.int64
            else ms.tensor(v) if isinstance(v, (np.ndarray, list))
            else v)
        for k, v in examples.items()
    }

class GLM4vPTQTester:
    """GLM4v PTQ Tester for mindone models"""

    def __init__(self, model_path: str, dataset_path: str,
                 output_dir: str, test_image_url: str):
        """Initialize tester"""

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.test_image_url = test_image_url

        # Create temporary output directory if not specified
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def create_ptq_config(self, mode, backend):
        """Create PTQ configuration for mindone models"""
        cfg = PTQConfig(mode=mode, backend=backend,
                        weight_quant_dtype=msdtype.int8,
                        opname_blacklist=['lm_head', 'visual'])
        layer_policies = OrderedDict()
        return cfg, layer_policies

    def preprocess_and_tokenizer(self, example):
        """Preprocess dataset example for calibration"""
        image = example["image"]
        question = example['question']
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
        )
        return inputs

    def create_calib_datasets(self):
        """Create calibration datasets"""
        print(f"Load datasets {self.dataset_path}")
        ds = load_dataset(self.dataset_path, split="train[:1]")
        ds = ds.shuffle(seed=2025)
        print("Preprocess datasets...")
        ds = ds.map(self.preprocess_and_tokenizer, remove_columns=ds.column_names)
        ds.set_transform(convert_to_tensor)
        return ds

    def create_eval_image(self):
        """Create evaluation image input"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": self.test_image_url,
                    },
                    {
                        "type": "text",
                        "text": "Describe this image.",
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
        )

        # Convert input to Tensor
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                inputs[key] = ms.tensor(value)
            elif isinstance(value, list):
                inputs[key] = ms.Tensor(value)
            if inputs[key].dtype == ms.int64:
                inputs[key] = inputs[key].to(ms.int32)
        return inputs

    def calibrate_model(self):
        """Calibrate GLM4v model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create GLM4v processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        print("Create calibration datasets...")
        ds = self.create_calib_datasets()

        print("Create PTQ config...")
        cfg, layers_policy = self.create_ptq_config(
            PTQMode.QUANTIZE, BackendTarget.ASCEND
        )

        print("Create GLM4v model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        print("Start calibration...")
        model.calibrate(cfg, layers_policy, datasets=ds)

        print(f"Save quantized model to {self.output_dir}")
        model.save_quantized(self.output_dir)

    def evaluate_model(self):
        """Evaluate quantized GLM4v model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create GLM4v model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        print("Create evaluation config...")
        eval_cfg, eval_layers_policy = self.create_ptq_config(
            PTQMode.DEPLOY, BackendTarget.ASCEND
        )

        print("Load quantized model...")
        model.fake_quant(eval_cfg, eval_layers_policy, self.output_dir)

        print("Create evaluation input...")
        inputs = self.create_eval_image()

        print("Generate response...")
        generated_ids = model.forward(inputs, max_new_tokens=32)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(f"Generated text: {output_text}")
        return output_text[0]

    def run_full_test(self):
        """Run complete PTQ test including calibration and evaluation"""
        try:
            # Step 1: Calibrate model
            self.calibrate_model()
            assert self.check_quant_description(self.output_dir), "Quantization description test failed!"

            # Step 2: Evaluate model
            output_text = self.evaluate_model()

            # Step 3: Basic validation
            golden_output = self.get_golden()
            assert output_text is not None, "Generated text should not be None"
            assert len(output_text) > 0, "Generated text should not be empty"
            assert output_text.startswith(golden_output), f"Generated text should start with '{golden_output}'"

            print("GLM4v PTQ test completed successfully!")
            return True

        except Exception as e:
            print(f"GLM4v PTQ test failed: {str(e)}")
            return False

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
            'model.language_model.layers.0.self_attn.q_proj.weight': QuantType.W8A16.value,
            'model.language_model.layers.1.self_attn.k_proj.weight_scale': QuantType.W8A16.value,
            'model.language_model.layers.2.self_attn.v_proj.weight_offset': QuantType.W8A16.value,
            'model.language_model.layers.3.mlp.gate_up_proj.weight': QuantType.W8A16.value,
            'model.language_model.layers.4.mlp.down_proj.weight_scale': QuantType.W8A16.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        print("quant description test success.")
        return True

    def get_ds_acc_threshold(self) -> Optional[float]:
        """Get accuracy threshold for distributed training"""
        return None

    def get_golden(self) -> tuple[str, str]:
        """Get golden reference for comparison"""
        return "<think>Got it, let's describe the image. " + \
               "First, the main subject is a cat, specifically a Pallas's cat,"


def run_glm4v_accuracy():
    """Run GLM4v PTQ test"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = "/home/workspace/mindspore_dataset/weight/GLM-4.1V-9B-Thinking/"
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
        f"GSLOG=1 python {run_file} > test_glm4v_accuracy.log"
    )
    if return_code != 0:
        with open("./test_glm4v_accuracy.log", "r", encoding="utf-8") as log_file:
            for line in log_file:
                print(line, flush=True)
    assert return_code == 0

if __name__ == "__main__":
    run_glm4v_accuracy()
