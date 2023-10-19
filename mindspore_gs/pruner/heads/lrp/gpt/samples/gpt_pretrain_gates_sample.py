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
"""
bert pretrain gates sample
"""

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype
from official.nlp.GPT.src.gpt import GPT_Head, GPT_Model


class GPTForGates(nn.Cell):
    """
    The GPT network consisting of two parts the backbone and the head

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map

    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """
    def __init__(self, config):
        super(GPTForGates, self).__init__()
        self.backbone = GPT_Model(config)
        self.head = GPT_Head(config)

    def construct(self, input_ids, input_mask):
        output_states, _, embedding_table, total_reg = self.backbone(input_ids, input_mask)
        logits = self.head(output_states, embedding_table)
        return logits, total_reg


class GPTWithLossForGates(nn.Cell):
    """
    GPT training loss

    Args:
        network: backbone network of GPT2/3
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token

    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map

    Returns:
        output: Tensor, the loss of the network
    """
    def __init__(self, network, loss, eos_token=50256):
        super(GPTWithLossForGates, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token

    def construct(self, input_ids, past=None):
        tokens = input_ids[:, :-1]
        input_mask = F.cast(F.not_equal(tokens, self.eos_token), mstype.float32)
        logits, total_reg = self.network(tokens, input_mask, past)
        labels = input_ids[:, 1:]
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output + total_reg


class GPTWithModel(nn.Cell):
    """
    The GPT network consisting of two parts the backbone and the head

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map

    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """
    def __init__(self, gpt_model, config):
        super(GPTWithModel, self).__init__()
        self.backbone = gpt_model
        self.head = GPT_Head(config)

    def construct(self, input_ids, input_mask, past=None):
        output_states, _, embedding_table = self.backbone(input_ids, input_mask, past)
        logits = self.head(output_states, embedding_table)
        return logits
