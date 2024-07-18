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
"""test squad dataset."""
import os.path

import pytest
import numpy as np
from mindspore import context, Tensor, dtype
from mindformers import LlamaTokenizer
from mindspore_gs.datasets import create_squad_dataset


def check_ds(ds_path: str, bs: int, seq_length: int, vocab_file: str, repeat):
    """Create and check squad dataset."""
    tokenizer = LlamaTokenizer(vocab_file=vocab_file)
    ds = create_squad_dataset(ds_path, "eval", bs, seq_length, tokenizer, repeat=repeat)
    assert ds.get_repeat_count() == repeat
    assert ds.output_types()[0] == np.int32
    assert ds.output_shapes()[0] == [bs, seq_length]
    assert ds.output_shapes()[1] == [bs, seq_length]
    assert ds.get_dataset_size() == 2067 // bs * repeat

    for inputs in ds.create_dict_iterator():
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        assert isinstance(input_ids, Tensor)
        assert isinstance(labels, Tensor)
        assert input_ids.shape == (bs, seq_length)
        assert labels.shape == (bs, seq_length)
        assert input_ids.dtype == dtype.int32
        assert labels.dtype == dtype.int32

    n_samples = 3
    ds = create_squad_dataset(ds_path, "eval", bs, seq_length, tokenizer, repeat=repeat, n_samples=n_samples)
    assert ds.get_dataset_size() == n_samples // bs * repeat


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend", "CPU"])
def test_squad(device):
    """
    Feature: SQuADDataset with llama2 tokenizer.
    Description: Create a SQuADDataset and iterate through it.
    Expectation: Execute successfully.
    """
    context.set_context(device_target=device)
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    squad_ds = os. path.join(cur_dir, "../../../data/squad-dataset/dev-v1.1.json")
    vocab_file = os. path.join(cur_dir, "../../../data/llama2-tokenizer.model")
    check_ds(squad_ds, 1, 500, vocab_file, 1)
    check_ds(squad_ds, 2, 501, vocab_file, 1)
    check_ds(squad_ds, 1, 502, vocab_file, 2)
