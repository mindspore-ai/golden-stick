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

import pytest
import numpy as np
import mindspore as ms
from mindspore import dtype as msdtype
from datasets import load_dataset

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindone")))
from transformers import AutoTokenizer

# pylint: disable=wrong-import-position
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import (PTQConfig, PTQMode)
from mindspore_gs.ptq.ptq_config import OutliersSuppressionType, QuantGranularity
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

class Qwen3PTQTester:
    """Qwen3 PTQ Tester for mindone models"""

    def __init__(self, model_path: str, dataset_path: str,
                 fake_quant_output_dir: str):
        """Initialize tester"""

        self.model_path = model_path
        self.dataset_path = dataset_path
        self.fake_quant_output_dir = fake_quant_output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Create temporary output directory if not specified
        if not os.path.exists(self.fake_quant_output_dir):
            os.makedirs(self.fake_quant_output_dir, exist_ok=True)

    def create_ptq_config(self, mode, backend):
        """Create PTQ configuration for mindone models"""
        cfg = PTQConfig(mode=mode, backend=backend,
                            weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8,
                            outliers_suppression= OutliersSuppressionType.SMOOTH,
                            opname_blacklist=["lm_head."])
        a8dynw8 = PTQConfig(mode=mode, backend=backend,
                            weight_quant_dtype=msdtype.int8,
                            act_quant_dtype=msdtype.int8,
                            act_quant_granularity=QuantGranularity.PER_TOKEN,
                            opname_blacklist=["lm_head."])
        a16w8 = PTQConfig(mode=mode, backend=backend,
                          weight_quant_dtype=msdtype.int8,
                          opname_blacklist=["lm_head."])
        layer_policies = OrderedDict({r'.*\.self_attn*': cfg,
                                      r'.*\.mlp\.gate_proj.*': a8dynw8,
                                      r'.*\.mlp\.up_proj.*': a8dynw8,
                                      r'.*\.mlp\.down_proj.*': a16w8,
                                      'not match': cfg})
        return cfg, layer_policies

    def preprocess_and_tokenizer(self, example):
        """Preprocess dataset example for calibration"""
        print("example:",example, flush=True)
        question = example["question"]
        # 构建选项文本
        options_text = f"A. {example['A']}\nB. {example['B']}\nC. {example['C']}\nD. {example['D']}"
        full_prompt = f"请回答以下单项选择题，请不要分析过程，直接在A、B、C、D四个选项中选出正确答案。\n问题: {question}\n选项:\n{options_text}\n答案:"
        print("full_prompt", full_prompt, flush=True)
        inputs =  self.tokenizer(
            full_prompt, return_tensors="np"
        )
        inputs["labels"] = self.tokenizer(
            example["answer"], return_tensors="np"
        )["input_ids"]
        return inputs

    def create_calib_datasets(self, num_samples=30):
        """Create calibration datasets"""
        print(f"Load datasets {self.dataset_path}")
        ds = load_dataset(self.dataset_path, split=f"validation[:{num_samples}]")
        print("before Preprocess datasets...", ds, flush=True)
        ds = ds.map(self.preprocess_and_tokenizer, remove_columns=ds.column_names)
        ds.set_transform(convert_to_tensor)
        print("after Preprocess datasets...", ds, flush=True)
        return ds

    def calibrate_model(self):
        """Calibrate Qwen3 model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create Qwen3 processor...", flush=True)

        print("Create calibration datasets...")
        ds = self.create_calib_datasets()
        ds = ds.select_columns(["input_ids", "attention_mask"])

        print("Create PTQ config...", flush=True)
        cfg, layers_policy = self.create_ptq_config(
            PTQMode.QUANTIZE, BackendTarget.ASCEND
        )

        print("Create Qwen3 model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        print("Start calibration...")
        model.calibrate(cfg, layers_policy, datasets=ds)

        print(f"Save quantized model to {self.fake_quant_output_dir}")
        model.save_quantized(self.fake_quant_output_dir, backend=BackendTarget.NONE)

    def evaluate(self, model):
        """evaluate 'network' with dataset from 'dataset_path'."""
        ds = self.create_calib_datasets(num_samples=100)
        correct = 0
        data_count = 0
        for _, ds_item in enumerate(ds):
            data_count += 1
            model_inputs = {"input_ids": ds_item["input_ids"]}
            generated_ids = model.forward(model_inputs, max_new_tokens=10)
            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids \
                in zip(ds_item["input_ids"], generated_ids)]
            pred_str = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            label_str = self.tokenizer.batch_decode(ds_item["labels"], skip_special_tokens=True)[0]
            question = self.tokenizer.batch_decode(ds_item["input_ids"], skip_special_tokens=True)[0]
            if label_str.lower() in pred_str.lower():
                correct += 1
                print(f"question {data_count}: {question}\n predict: {pred_str} answer: {label_str}. correct!",
                        flush=True)
            else:
                print(f"question {data_count}: {question}\n predict: {pred_str} answer: {label_str}. not correct!",
                        flush=True)
        ms.ms_memory_recycle()
        return correct / data_count

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
        model.fake_quant(eval_cfg, eval_layers_policy, self.fake_quant_output_dir)

        print("Create evaluation input...")
        res = self.evaluate(model)
        return res

    def run_full_test(self):
        """Run complete PTQ test including calibration and evaluation"""
        # Step 1: Calibrate model
        self.calibrate_model()
        assert self.check_quant_description(self.fake_quant_output_dir), "Quantization description test failed!"

        # # Step 2: Evaluate model
        threshold = self.get_ds_acc_threshold()
        score = self.evaluate_model()
        print("="*50, flush=True)
        print(f"Score {score}", flush=True)
        assert score >= threshold, f"CEval score {score:.4f} is lower than standard {threshold}"
        print("Qwen3 PTQ test completed successfully!")
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
            'model.layers.0.self_attn.q_proj.weight': QuantType.W8A8.value,
            'model.layers.0.self_attn.k_proj.input_scale': QuantType.W8A8.value,
            'model.layers.0.self_attn.v_proj.weight_offset': QuantType.W8A8.value,
            'model.layers.0.self_attn.o_proj.weight': QuantType.W8A8.value,
            'model.layers.0.mlp.gate_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.0.mlp.up_proj.weight': QuantType.W8A8_DYNAMIC.value,
            'model.layers.0.mlp.down_proj.weight': QuantType.W8A16.value,
        }
        for name, value in check_map.items():
            if not check(name, value):
                return False
        print("quant description test success.")
        return True

    def get_ds_acc_threshold(self) -> Optional[float]:
        """Get accuracy threshold for distributed training"""
        return 0.53


def run_qwen3_accuracy():
    """Run GLM4v PTQ test"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = "/home/workspace/mindspore_ckpt/safetensors/Qwen3-0.6B"
    dataset_path = "/home/workspace/mindspore_dataset/ceval/dev/"
    fake_quant_output_dir = os.path.join(cur_dir, "qwen3-0.6b-fakequant")

    tester = Qwen3PTQTester(model_path,
                            dataset_path,
                            fake_quant_output_dir)
    tester.run_full_test()

@pytest.mark.level0
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
