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
"""LRP Head Pruner"""
import pickle
import random
from abc import ABC, abstractmethod
from mindspore import save_checkpoint
from mindspore.nn import Cell
from mindspore_gs.common.validator import Validator
from ..abstract import AbstractHeadPruner


class AbstractHeadPrunerLRP(AbstractHeadPruner, ABC):
    """Head Pruner LRP class"""

    @abstractmethod
    def _init_head(self, model):
        """
        check if model has a head, save the model.
        Args:
            model: model to save
        """

    @abstractmethod
    def _decorate_model(self, l0_penalty=0.0015):
        """
        decorate model, repackage the model with additional functionality.
        Args:
            l0_penalty: penalty value for gate calculation.

        Returns: gated model.

        """

    @abstractmethod
    def _prune_model(self, model, save_dir_path=None):
        """
        Prune the model, after training/fine-tuning.
        Args:
            model: that has been decorated.
            save_dir_path (optional): path to save the models and heads dictionary
            input_sample (optional): dataset input sample for export MINDIR model

        Returns: pruned & clean model.

        """

    def _mask2dict(self, head_mask):
        """
        convert head mask to dictionary
        Args:
            head_mask: head mask to prune.

        Returns: dict with prune able heads.

        """
        Validator.check_value_type("head_mask", head_mask, [StubTensor], self.__class__.__name__)
        heads_dict = {}
        num_heads = len(head_mask[0])
        for i in range(len(head_mask)):
            new_array = []

            for j in range(len(head_mask[i])):

                if head_mask[i][j] == 0:
                    new_array.append(j)

            if len(new_array) == num_heads:
                num = random.randint(0, num_heads)
                new_array.pop(num)

            heads_dict[i] = new_array

        return heads_dict

    def _save_model(self, model, gates_dict, path):
        Validator.check_value_type("path", path, [str], self.__class__.__name__)
        Validator.check_value_type("gates_dict,", gates_dict, [dict], self.__class__.__name__)
        Validator.check_value_type("model", model, [Cell], self.__class__.__name__)
        save_checkpoint(model, path + '/gated_model.ckpt')

        with open(path + '/gates_dictionary', 'ab') as f:
            pickle.dump(gates_dict, f)
