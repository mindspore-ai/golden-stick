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


import json
import pathlib
from mindspore import dtype
import mindspore.dataset.transforms as C
from mindspore.dataset import GeneratorDataset

from mindspore_gs.common import logger
from mindspore_gs.datasets.base import BaseDataset


class BoolQDataset(BaseDataset):
    """boolQ dataset."""
    def __init__(self, path: str, mode: str, seq_length: int, tokenizer: callable, ignore_token_id=-100,
                 need_pad=False, n_samples=-1, add_special_tokens=True):
        super().__init__(path, mode, seq_length, tokenizer, ignore_token_id, need_pad, n_samples,
                         add_special_tokens)
        self._load()

    def _load(self):
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

                input_str = f"Please read the following passage and answer the question directly with either 'yes' or" \
                            f" 'no' without analysis process.\n\n ### " \
                            f"Passage:\n{title} -- {passage}\n### Question:\n{query}\n### Answer:"
                sources.append(input_str)
                targets.append(answer)
                if 0 < self.n_samples <= len(sources):
                    break
        total_items = 0
        total_items = self._dataset_based_on_mode(sources, targets, total_items)
        logger.info("Find %d total data items", total_items)


def create_boolq_dataset(ds_path: str, mode: str, bs: int, seq_length: int, tokenizer: callable,
                         ignore_token_id=-100, repeat=1, need_pad=False, n_samples=-1, add_special_tokens=True):
    """create squad dataset"""
    ds = BoolQDataset(ds_path, mode, seq_length, tokenizer, ignore_token_id, need_pad, n_samples, add_special_tokens)
    ds = GeneratorDataset(source=ds, column_names=["input_ids", "labels"])
    type_cast_op = C.TypeCast(dtype.int32)
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.map(operations=type_cast_op, input_columns="labels")
    ds = ds.batch(bs, drop_remainder=False)
    ds = ds.repeat(repeat)
    return ds
