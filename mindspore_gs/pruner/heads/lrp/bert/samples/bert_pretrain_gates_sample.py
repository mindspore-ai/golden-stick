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

import mindspore
import mindspore.nn as nn
from mindspore.nn import MSELoss
import mindspore.ops as ops


class BertPreTrainingForGates(nn.Cell):
    """
    Bert pretraining network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores, seq_relationship_score.
    """

    def __init__(self, config, model):
        super(BertPreTrainingForGates, self).__init__()
        self.bert = model
        self.classifier = nn.Dense(config.hidden_size, 3)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.stack = ops.Stack()
        self.has_gates = True
        self.num_labels = 3

    def construct(self,
                  token_type_id,
                  input_mask,
                  input_ids,
                  next_sentence_labels):
        """
        forward function
        @param token_type_id:
        @param input_mask:
        @param input_ids:
        @param next_sentence_labels:
        @return:
        """
        labels = next_sentence_labels
        _, pooled_output, _, total_reg = self.bert(input_ids, token_type_id, input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = mindspore.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.has_gates:
            loss += total_reg

        return loss
