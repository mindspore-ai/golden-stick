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
"""test interfaces of post training quant."""

from typing import Optional
import os
import sys
import re
import time
import shutil
from abc import ABC, abstractmethod
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../mindformers")))

from datasets import load_dataset
from mindone.transformers import AutoProcessor
import mindspore as ms
from mindspore import dataset
from mindspore.communication import get_rank, get_group_size
from mindformers import MindFormerConfig
from mindspore_gs.datasets import create_ceval_dataset
from mindspore_gs.ptq.models import AutoQuantForCausalLM
from mindspore_gs.common import BackendTarget
from mindspore_gs.ptq import PTQMode
from transformers import AutoTokenizer


class PTQModelTester:
    """PTQModelTester"""
    def create_ptq_config(self):
        raise NotImplementedError

    def check_quant_description(self, quant_ckpt_path) -> bool:
        raise NotImplementedError

    def get_ds_acc_threshold(self) -> Optional[float]:
        return None

    def get_golden(self) -> tuple[str, str]:
        return "", ""

    @staticmethod
    def create_ds(ds_path, tokenizer_, mode, n_samples=-1):
        """Create datasets."""
        dataset.config.set_numa_enable(False)
        if not ds_path:
            raise ValueError("Please provide dataset_path.")
        seq_ = 200
        ignore_token_id = tokenizer_.pad_token_id
        ds = create_ceval_dataset(ds_path, mode, 1, seq_, tokenizer_, ignore_token_id,
                                  1, False, n_samples=n_samples, use_box=True)
        return ds

    @staticmethod
    def evaluate(model, ds_path, tokenizer):
        """evaluate 'network' with dataset from 'dataset_path'."""
        ds = PTQModelTester.create_ds(ds_path, tokenizer, 'eval', n_samples=50)
        pad_token_id = tokenizer.pad_token_id
        correct = 0
        data_count = 0
        for _, ds_item in enumerate(ds.create_dict_iterator()):
            input_ids = ds_item['input_ids'].asnumpy()
            labels = ds_item['labels'].asnumpy()
            batch_valid_length = []
            for j in range(input_ids.shape[0]):
                batch_valid_length.append(np.max(np.argwhere(input_ids[j] != pad_token_id)) + 1)
            batch_valid_length = np.array(batch_valid_length)
            outputs = model.forward(input_ids, max_new_tokens=10)
            output_ids = []
            for j in range(input_ids.shape[0]):
                data_count += 1
                output_ids = outputs[j][int(batch_valid_length[j]):]
                pres_str = tokenizer.decode(output_ids, skip_special_tokens=True)
                labels_str = tokenizer.decode(labels[j], skip_special_tokens=True)
                question = tokenizer.decode(input_ids[j], skip_special_tokens=True)
                match = re.search(r'\{(.*?)\}', pres_str)
                pres_answer = match.group(1) if match else ''
                if labels_str.lower() == pres_answer.lower() or labels_str.lower() in pres_str.lower():
                    correct += 1
                    print(f"question {data_count}: {question}\n predict: {pres_str} answer: {labels_str}. correct!",
                          flush=True)
                else:
                    print(f"question {data_count}: {question}\n predict: {pres_str} answer: {labels_str}. not correct!",
                          flush=True)
        ms.ms_memory_recycle()
        return correct / data_count

    def quant_model(self, config_path_, output_dir_, ds_path, fake_quant=True):
        """quant by PTQ"""
        os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
        os.environ['ENFORCE_EAGER'] = "true"
        os.environ['MS_ALLOC_CONF'] = "enable_vmm:True"
        ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
        if not ascend_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        cur_dir_ = os.path.dirname(os.path.abspath(__file__))
        config_path_ = os.path.join(cur_dir_, config_path_)

        mfconfig = MindFormerConfig(config_path_)
        tokenizer = AutoTokenizer.from_pretrained(mfconfig.pretrained_model_dir, trust_remote_code=True)

        datasets = PTQModelTester.create_ds(ds_path, tokenizer, 'train', 20)
        model = AutoQuantForCausalLM.from_pretrained(config_path_)
        cfg, layers_policy = self.create_ptq_config()
        model.calibrate(cfg, layers_policy, datasets, fake_quant=fake_quant)
        model.save_quantized(output_dir_)
        time.sleep(5)
        os.environ.pop('ENFORCE_EAGER', None)

    def eval_model(self, config_path_, ckpt_path_, ds_path):
        """eval model by float ckpt and int ckpt"""
        os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
        os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
        os.environ['MS_ALLOC_CONF'] = "enable_vmm:True"
        os.environ.pop('ENFORCE_EAGER', None)
        ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
        if not ascend_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

        mfconfig = MindFormerConfig(config_path_)
        tokenizer = AutoTokenizer.from_pretrained(mfconfig.pretrained_model_dir, trust_remote_code=True)
        model = AutoQuantForCausalLM.from_pretrained(config_path_)
        cfg, layers_policy = self.create_ptq_config()
        model.fake_quant(cfg, layers_policy, ckpt_path_)
        os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = "PagedAttention"
        return PTQModelTester.evaluate(model, ds_path, tokenizer)

    def forward_model(self, config_path_, ckpt_path_, question):
        """forward model"""
        os.environ['MS_ENABLE_INTERNAL_KERNELS'] = "on"
        os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNAL_LIST'] = "QbmmAllReduceAdd,QbmmAdd"
        os.environ['MS_ALLOC_CONF'] = "enable_vmm:True"
        os.environ.pop('ENFORCE_EAGER', None)
        ascend_path = os.environ.get("ASCEND_HOME_PATH", "")
        if not ascend_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        mfconfig = MindFormerConfig(config_path_)
        tokenizer = AutoTokenizer.from_pretrained(mfconfig.pretrained_model_dir, trust_remote_code=True)
        model = AutoQuantForCausalLM.from_pretrained(config_path_)
        cfg, layers_policy = self.create_ptq_config()
        model.fake_quant(cfg, layers_policy, ckpt_path_)
        os.environ['MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST'] = "PagedAttention"
        input_ids = tokenizer.encode(question, add_special_tokens=True)
        outputs = model.forward(input_ids, max_new_tokens=20)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def golden_accuracy(self, calibrate_config_path_, infer_config_path_, quant_ckpt_path_, ds_path):
        """golden_accuracy"""
        question, answer = self.get_golden()
        result = question is not None and answer is not None, \
                 "Please implement get_golden before invoke golden_accuracy."
        self.quant_model(calibrate_config_path_, quant_ckpt_path_, ds_path)
        result = self.check_quant_description(quant_ckpt_path_)
        if result:
            pred = self.forward_model(infer_config_path_, quant_ckpt_path_, question)
            result = pred.startswith(answer)
            print("="*50, flush=True)
            print(f"{question} predict: {pred}, answer: {answer}", "success" if result else "failed", flush=True)
        try:
            group_size = get_group_size()
        except RuntimeError:
            group_size = 0
        if group_size > 0:
            ms.mint.distributed.barrier()
        return result

    def dataset_accuracy(self, calibrate_config_path_, infer_config_path_, quant_ckpt_path_, ds_path):
        """dataset_accuracy"""
        # pylint: disable=assignment-from-none
        threshold = self.get_ds_acc_threshold()
        result = threshold is not None, "Please implement get_ds_acc_threshold before invoke dataset_accuracy."
        self.quant_model(calibrate_config_path_, quant_ckpt_path_, ds_path)
        result = self.check_quant_description(quant_ckpt_path_)
        if result:
            score = self.eval_model(infer_config_path_, quant_ckpt_path_, ds_path)
            print("="*50, flush=True)
            print(f"Score {score}", flush=True)
            result = score >= threshold
            if not result:
                print(f"CEval score is {score:.4f}, which is lower than standard f{threshold}", flush=True)
        try:
            group_size = get_group_size()
        except RuntimeError:
            group_size = 0
        if group_size > 0:
            ms.mint.distributed.barrier()
        return result

    def print_log(self, log_path_):
        """print_log"""
        try:
            rank_id = get_rank()
        except RuntimeError:
            rank_id = 0
        if rank_id > 0:
            return
        os.system(f"cat {os.path.join(log_path_, 'worker_0.log')}")
        time.sleep(5)

    # pylint: disable=unused-argument
    def tear_down(self, quant_ckpt_path_, log_path_):
        """tear_down"""
        try:
            print(f"to rm dir: {quant_ckpt_path_}", flush=True)
            shutil.rmtree(quant_ckpt_path_)
        except (OSError, FileNotFoundError):
            pass


def convert_to_tensor(examples):
    """examples: dict[str, np.ndarray] -> dict[str, ms.Tensor]"""
    return {
        k: (ms.tensor(v, dtype=ms.int32) if isinstance(v, (np.ndarray, list)) and ms.tensor(v).dtype == ms.int64
            else ms.tensor(v) if isinstance(v, (np.ndarray, list))
            else v)
        for k, v in examples.items()
    }


class VLPTQTester(ABC):
    """VL model PTQ Tester for mindone models"""

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

    @abstractmethod
    def create_ptq_config(self, mode, backend):
        """Create PTQ configuration for mindone models"""
        raise NotImplementedError

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
        """Calibrate VL model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        print("Create calibration datasets...")
        ds = self.create_calib_datasets()

        print("Create PTQ config...")
        cfg, layers_policy = self.create_ptq_config(
            PTQMode.QUANTIZE, BackendTarget.ASCEND
        )

        print("Create model...")
        model = AutoQuantForCausalLM.from_pretrained(self.model_path)

        print("Start calibration...")
        model.calibrate(cfg, layers_policy, datasets=ds)

        print(f"Save quantized model to {self.output_dir}")
        model.save_quantized(self.output_dir, backend=BackendTarget.NONE)

    def evaluate_model(self):
        """Evaluate quantized model"""
        os.environ['MS_DISABLE_INTERNAL_KERNELS_LIST'] = "FlashAttentionScore"
        print("Create model...")
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

            print("VL PTQ test completed successfully!")
            return True

        except Exception as e:
            print(f"VL PTQ test failed: {str(e)}")
            return False

    @abstractmethod
    def check_quant_description(self, quant_ckpt_path: str) -> bool:
        """Check if quantized checkpoint has proper description"""
        raise NotImplementedError

    def get_ds_acc_threshold(self) -> Optional[float]:
        """Get accuracy threshold for distributed training"""
        return None

    @abstractmethod
    def get_golden(self) -> tuple[str, str]:
        """Get golden reference for comparison"""
        raise NotImplementedError
