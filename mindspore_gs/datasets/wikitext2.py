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
"""Wikitext-2 dataset."""

import os
import re
import numpy as np
from mindspore import Tensor, dtype
import mindspore.dataset.transforms as C
from mindspore.dataset import GeneratorDataset


class WikiText2Dataset:
    """Wikitext-2 dataset."""
    def __init__(self, path: str, seq_length: int, max_new_tokens: int, tokenizer: callable, need_pad=False,
                 n_samples=-1, add_special_tokens=True):
        self.path = os.path.join(path)
        self.data_type = os.path.basename(self.path).split('.')[-1]
        self.seq_len = seq_length
        self.max_new_tokens = max_new_tokens
        self.add_special_tokens = add_special_tokens
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.need_pad = need_pad
        if hasattr(self.tokenizer, 'add_bos_token') and self.tokenizer.bos_token is not None:
            self.tokenizer.add_bos_token = True
        if hasattr(self.tokenizer, 'add_eos_token') and self.tokenizer.eos_token is not None:
            self.tokenizer.add_eos_token = True
        self.content = []
        if self.data_type == "parquet":
            self._load_parquet(n_samples)
        else:
            self._load(n_samples)
        self.iterator = None

    def __len__(self):
        return len(self.content)

    def _load(self, n_samples=-1):
        """_load"""
        input_content = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for para in WikiText2Dataset._clean(f.read()).split("\n\n"):
                if para and para.strip().startswith('=') is False:
                    input_content += self.tokenizer(para, add_special_tokens=self.add_special_tokens)['input_ids']
        for chunk in WikiText2Dataset._chunks(input_content, self.seq_len - self.max_new_tokens):
            if len(chunk) == self.seq_len - self.max_new_tokens:
                content = np.array(chunk, dtype=np.int32)
                if self.need_pad:
                    content = np.pad(content, (0, self.max_new_tokens), 'constant', constant_values=self.pad_token_id)
                self.content.append(Tensor(content))
                if 0 < n_samples <= len(self.content):
                    break

    def _load_parquet(self, n_samples=-1):
        """_load_parquet"""
        input_content = []
        from datasets import load_dataset
        self.data = load_dataset('parquet', data_files=self.path, split='train')
        input_content += self.tokenizer("\n\n".join(self.data['text']),
                                        add_special_tokens=self.add_special_tokens)['input_ids']
        for chunk in WikiText2Dataset._chunks(input_content, self.seq_len - self.max_new_tokens):
            if len(chunk) == self.seq_len - self.max_new_tokens:
                content = np.array(chunk, dtype=np.int32)
                if self.need_pad:
                    content = np.pad(content, (0, self.max_new_tokens), 'constant',
                                     constant_values=self.pad_token_id)
                self.content.append(Tensor(content))
                if 0 < n_samples <= len(self.content):
                    break

    @staticmethod
    def _clean(string):
        """ cleaning wikitext dataset"""
        # contractions
        string = string.replace("s '", "s'")
        string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
        # number separators
        string = string.replace(" @-@ ", "-")
        string = string.replace(" @,@ ", ",")
        string = string.replace(" @.@ ", ".")
        # punctuation
        string = string.replace(" : ", ": ")
        string = string.replace(" ; ", "; ")
        string = string.replace(" . ", ". ")
        string = string.replace(" ! ", "! ")
        string = string.replace(" ? ", "? ")
        string = string.replace(" , ", ", ")
        # double brackets
        string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
        string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
        string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
        string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
        string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
        # miscellaneous
        string = string.replace("= = = =", "====")
        string = string.replace("= = =", "===")
        string = string.replace("= =", "==")
        string = string.replace(" " + chr(176) + " ", chr(176))
        string = string.replace(" \n", "\n")
        string = string.replace("\n ", "\n")
        string = string.replace(" N ", " 1 ")
        string = string.replace(" 's", "'s")
        return string

    @staticmethod
    def _chunks(lst, n):
        """ yield n sized chunks from list"""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def __next__(self):
        return next(self.iterator)

    def __iter__(self):
        self.iterator = iter(self.content)
        return self


def create_wikitext_dataset(ds_path: str, bs: int, seq_length: int, max_new_tokens: int, tokenizer: callable,
                            repeat=1, need_pad=False, n_samples=-1, add_special_tokens=True):
    """ create wikitext dataset"""
    if max_new_tokens >= seq_length:
        raise RuntimeError(f"max_decode_len should less than seq_length, but got max_new_tokens: {max_new_tokens}, "
                           f"seq_length: {seq_length}.")
    ds = WikiText2Dataset(ds_path, seq_length, max_new_tokens, tokenizer, need_pad, n_samples, add_special_tokens)
    ds = GeneratorDataset(source=ds, column_names=["input_ids"])
    type_cast_op = C.TypeCast(dtype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.batch(bs, drop_remainder=True)
    ds = ds.repeat(repeat)
    return ds
