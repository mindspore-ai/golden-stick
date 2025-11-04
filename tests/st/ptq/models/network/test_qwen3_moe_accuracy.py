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
from typing import Optional
from collections import OrderedDict

import numpy as np
import pytest
import mindspore as ms
from mindspore import dtype as msdtype
from datasets import load_dataset

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindone")))
from transformers import AutoTokenizer

# pylint: disable=wrong-import-position
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode)
from mindspore_gs.ptq.ptq_config import OutliersSuppressionType, PrecisionRecovery, QuantGranularity
from mindspore_gs.ptq.models import AutoQuantForCausalLM
from mindspore_gs.ptq.utils import QuantType


class Qwen3PTQTester:
    """Qwen3 PTQ Tester for mindone models"""

    def __init__(self, model_path: str, dataset_path: str,
                 output_dir: str):
        """Initialize tester"""

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Create temporary output directory if not specified
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def create_ptq_config(self, mode, backend):
        """Create PTQ configuration for mindone models"""
        cfg = PTQConfig(mode=mode, backend=backend,
                        weight_quant_dtype=msdtype.int8,
                        act_quant_dtype=msdtype.int8,
                        outliers_suppression= OutliersSuppressionType.SMOOTH,
                        opname_blacklist=["lm_head.", "mlp.gate"])
        ffn_config = PTQConfig(mode=mode, backend=backend, weight_quant_dtype=msdtype.int8,
                               act_quant_dtype=msdtype.int8,
                               outliers_suppression=OutliersSuppressionType.NONE,
                               precision_recovery=PrecisionRecovery.NONE,
                               act_quant_granularity=QuantGranularity.PER_TOKEN,
                               weight_quant_granularity=QuantGranularity.PER_CHANNEL)
        layer_policies = OrderedDict({r'.*\.mlp\.experts.*': ffn_config})
        return cfg, layer_policies

    def preprocess_and_tokenizer(self, example):
        """Preprocess dataset example for calibration"""
        print("example", example["question"], flush=True)
        inputs =  self.tokenizer(
            example["question"], return_tensors="np"
        )
        return inputs

    def create_calib_datasets(self, num_samples=100):
        """Create calibration datasets"""
        print(f"Load datasets {self.dataset_path}")
        ds = load_dataset(self.dataset_path, split=f"test[:{num_samples}]")
        # ds = ds.shuffle(seed=2025)
        print("before Preprocess datasets...", ds, flush=True)
        ds = ds.map(self.preprocess_and_tokenizer, remove_columns=ds.column_names)
        print("after Preprocess datasets...", ds, flush=True)
        return ds

    def create_eval_image(self):
        """Create evaluation image input"""

        inputs = "下列关于资本结构理论的说法中，不正确的是____。"
        input_ids = ms.Tensor(self.tokenizer([inputs], return_tensors="np").input_ids, ms.int32)
        return input_ids

    def calibrate_model(self):
        """Calibrate Qwen3 model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create Qwen3 processor...", flush=True)

        print("Create calibration datasets...")
        ds = self.create_calib_datasets()

        print("Create PTQ config...", flush=True)
        cfg, layers_policy = self.create_ptq_config(
            PTQMode.QUANTIZE, BackendTarget.ASCEND
        )

        print("Create Qwen3 model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        print("Start calibration...")
        model.calibrate(cfg, layers_policy, datasets=ds)

        print(f"Save quantized model to {self.output_dir}")
        model.save_quantized(self.output_dir)

    def evaluate_model(self):
        """Evaluate quantized Qwen3 model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create Qwen3 model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        print("Create evaluation config...")
        eval_cfg, eval_layers_policy = self.create_ptq_config(
            PTQMode.DEPLOY, BackendTarget.ASCEND
        )

        print("Load quantized model...")
        model.fake_quant(eval_cfg, eval_layers_policy, self.output_dir)
        # model.save_quantized(self.output_dir, ptq_config=eval_cfg, layers_policy=eval_layers_policy)

        print("Create evaluation input...")
        inputs = self.create_eval_image()
        model_inputs = {}
        model_inputs["input_ids"] = inputs
        print("Generate response...")
        generated_ids = model.forward(inputs, max_new_tokens=32)

        generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs, generated_ids)]
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(outputs)
        return outputs

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

            print("Qwen3 PTQ test completed successfully!")
            return True

        except Exception as e:
            print(f"Qwen3 PTQ test failed: {str(e)}")
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
            'model.layers.0.self_attn.q_proj.weight': QuantType.W8A8.value,
            'model.layers.0.self_attn.k_proj.input_scale': QuantType.W8A8.value,
            'model.layers.0.self_attn.v_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.0.mlp.experts.126.up_proj.weight': QuantType.W8A8_DYNAMIC.value,
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
        return "該使用者SetActive…おすす�建(Denseчь ApplicationContext.MODE.MODE."


@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_qwen3_accuracy():
    """
    Feature: test ptq quantization for qwen3 model in mindone.
    Description: apply a16w4 on qwen3 language part and check the generated text.
    Expectation: generated text is good.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = "/home/ckpt/Qwen3-30B-A3B-no-sample_1"
    dataset_path = "/home/csz/benchmark/ais_bench/datasets/ceval/formal_ceval/test/"
    output_dir = os.path.join(cur_dir, "qwen3-quant")

    tester = Qwen3PTQTester(model_path,
                            dataset_path,
                            output_dir)
    assert tester.run_full_test(), "Qwen3 PTQ test failed!"
