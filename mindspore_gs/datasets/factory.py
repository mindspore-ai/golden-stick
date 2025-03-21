# Copyright 2024 Huawei Technologies Co., Ltd
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
"""get_datasets"""


from mindspore.dataset import Dataset
from .wikitext2 import create_wikitext_dataset
from .squad import create_squad_dataset
from .boolq import create_boolq_dataset
from .ceval import create_ceval_dataset
from .calibrate import create_calibrate_dataset


def get_datasets(ds_type: str, ds_path: str, mode, batch_size, seq_length, max_new_tokens, tokenizer,
                 ignore_token_id, repeat=1, need_pad=True, n_samples=-1, add_special_tokens=True) -> Dataset:
    """get_datasets."""
    if ds_type.lower() == 'wikitext2':
        return create_wikitext_dataset(ds_path, batch_size, seq_length, max_new_tokens, tokenizer, repeat, need_pad,
                                       n_samples, add_special_tokens)
    if ds_type.lower() == 'squad1.1':
        return create_squad_dataset(ds_path, mode, batch_size, seq_length, tokenizer, ignore_token_id, repeat,
                                    need_pad, n_samples, add_special_tokens)
    if ds_type.lower() == 'boolq':
        return create_boolq_dataset(ds_path, mode, batch_size, seq_length, tokenizer, ignore_token_id, repeat,
                                    need_pad, n_samples, add_special_tokens)
    if ds_type.lower() == 'ceval':
        return create_ceval_dataset(ds_path, mode, batch_size, seq_length, tokenizer, ignore_token_id, repeat,
                                    need_pad, n_samples, add_special_tokens)
    if ds_type.lower() == 'calibrate':
        return create_calibrate_dataset(ds_path, mode, batch_size, seq_length, tokenizer, ignore_token_id, repeat,
                                        need_pad, n_samples, add_special_tokens)
    raise ValueError(f"Not supported datasets type: {ds_type}.")
