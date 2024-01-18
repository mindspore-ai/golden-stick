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
"""LLMGenerateNetwork for LLM accuracy evaluation."""

import numpy as np

from mindspore.nn import Cell
from mindformers.models.base_model import BaseModel


class LLMGenerateNetwork(Cell):
    """LLMGenerateNetwork for LLM accuracy evaluation."""
    def __init__(self, network: BaseModel, do_sample, max_length, top_p, top_k, pad_token_id, tokenizer):
        """init."""
        super().__init__()
        if not isinstance(network, BaseModel):
            raise ValueError("Input network is not a BaseModel, got: ", network)
        self.network: BaseModel = network
        self.do_sample = do_sample
        self.max_length = max_length
        self.top_p = top_p
        self.top_k = top_k
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer

    def construct(self, input_ids, label):
        """construct."""
        valid_length_each_example = []
        for j in range(input_ids.shape[0]):
            # As the nonzero returns the index and we need length
            valid_length_each_example.append(np.max(np.argwhere(input_ids[j] != self.pad_token_id)) + 1)
        valid_length_each_example = np.array(valid_length_each_example)

        outputs = self.network.generate(input_ids, do_sample=self.do_sample, max_length=self.max_length,
                                        top_p=self.top_p, top_k=self.top_k)
        output_ids = []
        for j in range(input_ids.shape[0]):
            output_ids.append(outputs[j][int(valid_length_each_example[j]):])

        tokens_num = 0
        for batch_index in range(len(output_ids)):
            tokens_num += output_ids[batch_index].shape[0]

        loss = None
        # decode input_id and label to string
        pres_str = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        labels_str = self.tokenizer.decode(label, skip_special_tokens=True)
        return loss, pres_str, labels_str
