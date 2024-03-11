# Copyright 2023 Huawei Technologies Co., Ltd
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
"""SQuAD dataset."""
import copy
import json
import os
import pathlib
import numpy as np
from mindspore import dtype, Tensor
from mindspore import log as logger
import mindspore.dataset.transforms as C
from mindspore.dataset import GeneratorDataset


class SQuADDataset(GeneratorDataset):
    """SQuAD dataset."""
    def __init__(self, path: str, mode: str, seq_length: int, tokenizer: callable, ignore_token_id=-100):
        self.path = os.path.join(path)
        if mode not in ("eval", "train", "test"):
            raise ValueError("Input `mode` should be 'eval', 'test' or 'train', got: ", mode)
        self.mode = mode
        self.seq_len = seq_length
        self.ignore_token_id = ignore_token_id
        self.tokenizer = tokenizer
        if mode in ("eval", "test"):
            if hasattr(self.tokenizer, 'add_bos_token'):
                self.tokenizer.add_bos_token = True
            if hasattr(self.tokenizer, 'add_eos_token'):
                self.tokenizer.add_eos_token = False
        else:
            if hasattr(tokenizer, 'add_bos_token'):
                tokenizer.add_bos_token = True
            if hasattr(tokenizer, 'add_eos_token'):
                tokenizer.add_eos_token = True
        self.input_ids = []
        self.labels = []
        self._load()
        self.iter_input_ids = None
        self.iter_labels = None
        super().__init__(source=self, column_names=["input_ids", "labels"])

    def __len__(self):
        return len(self.input_ids)

    def _load(self):
        """Load and preprocess squad dataset."""
        input_file = pathlib.Path(self.path)
        with input_file.open() as f:
            file = json.load(f)
        sources = []
        targets = []
        for data in file["data"]:
            for paragraph in data["paragraphs"]:
                passage = paragraph["context"]
                query = paragraph["qas"][0]["question"]
                answer = paragraph["qas"][0]["answers"][0]["text"]

                input_str = f"Read the passage and answer the question below.\n\n### " \
                            f"Instruction:\n{passage}\n\n### Input:\n{query}\n\n### Response:"
                sources.append(input_str)
                targets.append(answer)

        total_items = 0
        total_items = self._dataset_based_on_mode(sources, targets, total_items)
        logger.info("Find %d total data items", total_items)

    def _dataset_based_on_mode(self, sources, targets, total_items):
        """create dataset based on mode"""
        self.input_ids.clear()
        self.labels.clear()
        pad_mode = 'constant'
        if self.mode == "eval":
            for prompt, answer in zip(sources, targets):
                total_items += 1
                input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                if len(input_ids) >= self.seq_len:
                    input_ids = input_ids[:self.seq_len]
                else:
                    input_ids = np.pad(input_ids, (0, self.seq_len - len(input_ids)), pad_mode,
                                       constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))

                label_id = self.tokenizer.encode(answer, add_special_tokens=False)
                label_id = np.pad(label_id, (0, self.seq_len - len(label_id)), pad_mode,
                                  constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))

                self.input_ids.append(Tensor(input_ids, dtype=dtype.int32))
                self.labels.append(Tensor(label_id, dtype=dtype.int32))
        # for train/finetune
        else:
            for prompt, answer in zip(sources, targets):
                total_items += 1
                concated_qa = prompt + answer
                input_ids = self.tokenizer.encode(concated_qa, add_special_tokens=True)
                input_ids = np.array(input_ids)

                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                prompt_ids = np.array(prompt_ids)
                prompt_length = len(prompt_ids)
                concat_length = len(input_ids)

                pad_length = self.seq_len + 1 - concat_length
                input_ids_new = np.pad(input_ids, (0, pad_length), pad_mode,
                                       constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))
                label_id_new = copy.deepcopy(input_ids_new)
                label_id_new[:prompt_length] = self.ignore_token_id
                label_id_new[-pad_length:] = self.ignore_token_id
                self.input_ids.append(Tensor(input_ids_new, dtype=dtype.int32))
                self.labels.append(Tensor(label_id_new, dtype=dtype.int32))
        return total_items

    def __next__(self):
        return next(self.iter_input_ids), next(self.iter_labels)

    def __iter__(self):
        """tokenize wikitext-2/wikitext-103 dataset"""
        self.iter_input_ids = iter(self.input_ids)
        self.iter_labels = iter(self.labels)
        return self


def create_squad_dataset(ds_path: str, mode: str, bs: int, seq_length: int, tokenizer: callable,
                         ignore_token_id=-100, repeat=1):
    """create squad dataset"""
    ds = SQuADDataset(ds_path, mode, seq_length, tokenizer, ignore_token_id)
    type_cast_op = C.TypeCast(dtype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="labels")
    ds = ds.batch(bs, drop_remainder=True)
    ds = ds.repeat(repeat)
    return ds
