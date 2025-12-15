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
"""common utils for mixed precision network PTQ test"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, dtype as msdtype
from mindspore.dataset import GeneratorDataset


def create_test_input(hidden_size):
    """Create fixed test input with shape (10, hidden_size), data range [-1, 1]"""
    np.random.seed(42)
    data = np.zeros((10, hidden_size), dtype=np.float32)
    for i in range(10):
        for j in range(hidden_size):
            value = ((i * 1000 + j) % 2000 - 1000) / 1000.0
            data[i, j] = value

    return Tensor(data, dtype=msdtype.float32)


def create_linear_ds(batch_size, seq_length, repeat=1, is_parallel=False):
    """Create linear dataset with fixed deterministic values, data range [-1, 1]"""
    class LinearIterable:
        """Iterable dataset for linear layers with fixed deterministic values"""
        def __init__(self, batch_size, seq_length, repeat=1, is_parallel=False):
            self.index = 0
            self.data = []
            for i in range(repeat):
                if not is_parallel:
                    data = np.zeros((batch_size, seq_length), dtype=np.float16)
                    for b in range(batch_size):
                        for s in range(seq_length):
                            value = ((i * 10000 + b * 1000 + s) % 2000 - 1000) / 1000.0
                            data[b, s] = value
                    self.data.append(data)
                else:
                    data = np.zeros((batch_size, 1, seq_length), dtype=np.float16)
                    for b in range(batch_size):
                        for s in range(seq_length):
                            value = ((i * 10000 + b * 1000 + s) % 2000 - 1000) / 1000.0
                            data[b, 0, s] = value
                    self.data.append(data)

        def __next__(self):
            if self.index >= len(self.data):
                raise StopIteration
            item = (self.data[self.index],)
            self.index += 1
            return item

        def __iter__(self):
            self.index = 0
            return self

        def __len__(self):
            return len(self.data)

    return GeneratorDataset(
        source=LinearIterable(batch_size, seq_length, repeat, is_parallel),
        column_names=["input_ids"])


def convert_to_tensor(examples):
    """examples: dict[str, np.ndarray] -> dict[str, ms.Tensor]"""
    return {
        k: (ms.tensor(v, dtype=ms.int32) if isinstance(v, (np.ndarray, list)) and ms.tensor(v).dtype == ms.int64
            else ms.tensor(v) if isinstance(v, (np.ndarray, list))
            else v)
        for k, v in examples.items()
    }


def get_save_file_name(save_name):
    """Get the save file name"""
    return save_name
