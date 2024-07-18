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
"""BoolQ dataset."""


import copy
import json
import os
import pathlib
import numpy as np
from mindspore import dtype, Tensor
import mindspore.dataset.transforms as C
from mindspore.dataset import GeneratorDataset
from mindspore_gs.common import logger


class BoolQDataset(GeneratorDataset):
    """boolQ dataset."""
    def __init__(self, path: str, mode: str, seq_length: int, tokenizer: callable, ignore_token_id=-100,
                 need_pad=True, n_samples=-1):
        self.path = os.path.join(path)
        if mode not in ("eval", "train", "test"):
            raise ValueError("Input `mode` should be 'eval', 'test' or 'train', got: ", mode)
        self.mode = mode
        self.seq_len = seq_length
        self.ignore_token_id = ignore_token_id
        self.tokenizer = tokenizer
        self.need_pad = need_pad
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
        self._load(n_samples)
        self.iter_input_ids = None
        self.iter_labels = None
        super().__init__(source=self, column_names=["input_ids", "labels"])

    def __len__(self):
        return len(self.input_ids)

    def _load(self, n_samples=-1):
        """Load and preprocess squad dataset."""
        sources = []
        targets = []
        input_file = pathlib.Path(self.path)
        with open(input_file, encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                passage = data["passage"]
                query = data["question"]
                answer = 'yes' if data["answer"] else 'no'
                title = data["title"]

                input_str = f"Read the passage below and answer the question with yes or no.\n### " \
                            f"Passage:\n{title} -- {passage}\n### Question:\n{query}\n### Answer:"
                sources.append(input_str)
                targets.append(answer)
                if 0 < n_samples <= len(sources):
                    break
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
                label_id = self.tokenizer.encode(answer, add_special_tokens=False)
                if len(input_ids) >= self.seq_len:
                    input_ids = input_ids[:self.seq_len]
                if len(label_id) >= self.seq_len:
                    label_id = label_id[:self.seq_len]

                if self.need_pad:
                    input_ids = np.pad(input_ids, (0, self.seq_len - len(input_ids)), pad_mode,
                                       constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))
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

                if self.need_pad:
                    pad_length = self.seq_len + 1 - concat_length
                    input_ids = np.pad(input_ids, (0, pad_length), pad_mode,
                                       constant_values=(self.tokenizer.pad_token_id, self.tokenizer.pad_token_id))
                label_id_new = copy.deepcopy(input_ids)
                label_id_new[:prompt_length] = self.ignore_token_id
                if self.need_pad:
                    label_id_new[-pad_length:] = self.ignore_token_id
                self.input_ids.append(Tensor(input_ids, dtype=dtype.int32))
                self.labels.append(Tensor(label_id_new, dtype=dtype.int32))
        return total_items

    def __next__(self):
        return next(self.iter_input_ids), next(self.iter_labels)

    def __iter__(self):
        self.iter_input_ids = iter(self.input_ids)
        self.iter_labels = iter(self.labels)
        return self


def create_boolq_dataset(ds_path: str, mode: str, bs: int, seq_length: int, tokenizer: callable,
                         ignore_token_id=-100, repeat=1, need_pad=True, n_samples=-1):
    """create squad dataset"""
    ds = BoolQDataset(ds_path, mode, seq_length, tokenizer, ignore_token_id, need_pad, n_samples)
    type_cast_op = C.TypeCast(dtype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="labels")
    ds = ds.batch(bs, drop_remainder=True)
    ds = ds.repeat(repeat)
    return ds
