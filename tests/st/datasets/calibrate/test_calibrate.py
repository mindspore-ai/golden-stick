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
"""test calibrate dataset."""
import os.path
import math
import pytest
import numpy as np
from mindspore import context, Tensor, dtype
from mindformers import LlamaTokenizer
from mindspore_gs.datasets import create_calibrate_dataset


def check_ds(ds_path: str, bs: int, seq_length: int, vocab_file: str, repeat):
    """Create and check squad dataset."""
    tokenizer = LlamaTokenizer(vocab_file=vocab_file)
    if bs == 1:
        ds = create_calibrate_dataset(ds_path, "eval", bs, seq_length, tokenizer, repeat=repeat)
        samples = 200
        assert ds.get_repeat_count() == repeat
        assert ds.output_types()[0] == np.int32
        assert ds.output_shapes()[0] == [bs, 366]
        assert ds.output_shapes()[1] == [bs, 1]
        assert ds.get_dataset_size() == samples // bs * repeat
    ds = create_calibrate_dataset(ds_path, "eval", bs, seq_length, tokenizer, repeat=repeat, need_pad=True)
    samples = 200
    assert ds.get_repeat_count() == repeat
    assert ds.output_types()[0] == np.int32
    assert ds.output_shapes()[0] == [bs, seq_length]
    assert ds.output_shapes()[1] == [bs, seq_length]
    assert ds.get_dataset_size() == math.ceil(samples / bs * repeat)

    index = 0
    for inputs in ds.create_dict_iterator():
        index += 1
        if index > 100:
            break
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        assert isinstance(input_ids, Tensor)
        assert isinstance(labels, Tensor)
        assert input_ids.shape == (bs, seq_length)
        assert labels.shape == (bs, seq_length)
        assert input_ids.dtype == dtype.int32
        assert labels.dtype == dtype.int32

    n_samples = 3
    ds = create_calibrate_dataset(ds_path, "eval", bs, seq_length, tokenizer, repeat=repeat, n_samples=n_samples)
    assert ds.get_dataset_size() == math.ceil(n_samples / bs * repeat)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("device", ["Ascend", "CPU"])
def test_calibrate(device):
    """
    Feature: CalibrateDataset with llama2 tokenizer.
    Description: Create a CalibrateDataset and iterate through it.
    Expectation: Execute successfully.
    """
    context.set_context(device_target=device)
    cur_dir, _ = os.path.split(os.path.abspath(__file__))
    boolq_ds = os. path.join(cur_dir, "../../../data/calibrate-dataset/calibrate.jsonl")
    vocab_file = os. path.join(cur_dir, "/nfs/dataset/workspace/mindspore_vocab/llama2/llama2-tokenizer.model")
    check_ds(boolq_ds, 1, 500, vocab_file, 1)
    check_ds(boolq_ds, 2, 501, vocab_file, 1)
    check_ds(boolq_ds, 1, 502, vocab_file, 2)
