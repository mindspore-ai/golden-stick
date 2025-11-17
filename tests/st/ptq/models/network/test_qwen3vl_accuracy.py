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
"""Test Qwen3VL PTQ quantization for mindone models."""

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
from mindspore_gs.ptq import (PTQConfig, PTQMode, OutliersSuppressionType)
from mindspore_gs.ptq.models import AutoQuantForCausalLM
from mindspore_gs.ptq.utils import QuantType


class Qwen3VLPTQTester:
    """Qwen3VL PTQ Tester for mindone models"""

    def __init__(self, model_path: str, dataset_path: str,
                 output_dir: str, test_image_url: str):
        """Initialize tester"""

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(cur_dir, "qwen3vl-quant_ascend_test")
        self.output_dir_ascend = output_dir
        self.test_image_url = test_image_url
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False)

        # Create temporary output directory if not specified
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def create_ptq_config(self, mode, backend):
        """Create PTQ configuration for mindone models"""
        cfg = PTQConfig(mode=mode, backend=backend,
                        weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression= OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE,
                        opname_blacklist=['lm_head', 'visual', "down_proj"])
        layer_policies = OrderedDict()
        return cfg, layer_policies

    def preprocess_and_tokenizer(self, example):
        """Preprocess dataset example for calibration"""
        image = example["image"]
        question = example['question']
        print("image:", image, " question:", question, flush=True)
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
        print(f"Load datasets {self.dataset_path}", flush=True)
        ds = load_dataset(self.dataset_path, split="train[:20]")
        # ds = ds.shuffle(seed=2025)
        print("Preprocess datasets...", flush=True)
        ds = ds.map(self.preprocess_and_tokenizer, remove_columns=ds.column_names)
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
        """Calibrate Qwen3VL model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create Qwen3VL processor...", flush=True)

        print("Create calibration datasets...", flush=True)
        ds = self.create_calib_datasets()

        print("Create PTQ config...", flush=True)
        cfg, layers_policy = self.create_ptq_config(
            PTQMode.QUANTIZE, BackendTarget.ASCEND
        )

        print("Create Qwen3VL model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        print("Start calibration...")
        calibrate_options = {
            'algorithm_cache_path': {'osl': 'osl_cache'},
            'always_use_fp_input_in_processer': True,
            'skip_offload_in_processing': True,
        }
        model.calibrate(cfg, layers_policy, datasets=ds, **calibrate_options)

        print(f"Save quantized model to {self.output_dir}")
        model.save_quantized(self.output_dir)

    def evaluate_model(self):
        """Evaluate quantized Qwen3VL model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create Qwen3VL model...")
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
        generated_ids = model.forward(inputs, max_new_tokens=20)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        model.save_quantized(self.output_dir_ascend, backend=BackendTarget.ASCEND)

        print(f"Generated text: {output_text}")
        return output_text[0]

    def run_full_test(self):
        """Run complete PTQ test including calibration and evaluation"""
        # Step 1: Calibrate model
        self.calibrate_model()
        assert self.check_quant_description(self.output_dir), "Quantization description test failed!"

        # Step 2: Evaluate model
        output_text = self.evaluate_model()
        print("output_text:", output_text, flush=True)
        # Step 3: Basic validation
        golden_output = self.get_golden()
        assert output_text is not None, "Generated text should not be None"
        assert len(output_text) > 0, "Generated text should not be empty"
        assert output_text.startswith(golden_output), f"Generated text should start with '{golden_output}'"

        print("Qwen3VL PTQ test completed successfully!")
        return True

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
            'model.language_model.layers.1.self_attn.k_proj.weight_scale': QuantType.W8A8.value,
            'model.language_model.layers.2.self_attn.v_proj.weight_offset': QuantType.W8A8.value,
            'model.language_model.layers.4.mlp.gate_proj.weight_scale': QuantType.W8A8.value,
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
        return "This is a captivating photograph of a **Pallas’s cat** (also known as the manul"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_qwen3vl_accuracy():
    """
    Feature: test ptq quantization for qwen3vl model in mindone.
    Description: apply a16w4 on qwen3vl language part and check the generated text.
    Expectation: generated text is good.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = "/home/ckpt/Qwen3-VL-8B-Instruct_1/"
    dataset_path = "/home/ckpt/textvqa"
    output_dir = os.path.join(cur_dir, "qwen3vl-quant")
    test_image_url = "/home/ckpt/pipeline-cat-chonk.jpeg"

    tester = Qwen3VLPTQTester(model_path,
                            dataset_path,
                            output_dir,
                            test_image_url)
    assert tester.run_full_test(), "Qwen3VL PTQ test failed!"
