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
import os
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
        """Save model and gates dictionary to specified path.

        Args:
            model: The model to save
            gates_dict: Dictionary containing gate information
            path: Directory path to save the model and gates
        """
        Validator.check_value_type("path", path, [str], self.__class__.__name__)
        Validator.check_value_type("gates_dict,", gates_dict, [dict], self.__class__.__name__)
        Validator.check_value_type("model", model, [Cell], self.__class__.__name__)

        # Normalize and validate the path
        normalized_path = os.path.normpath(path)
        if not os.path.isabs(normalized_path):
            raise ValueError("Path must be absolute")

        # Check for path traversal attempts
        if ".." in normalized_path:
            raise ValueError("Path traversal detected")

        # Ensure the directory exists
        if not os.path.exists(normalized_path):
            raise ValueError(f"Directory does not exist: {normalized_path}")

        if not os.path.isdir(normalized_path):
            raise ValueError(f"Path is not a directory: {normalized_path}")

        # Construct and validate checkpoint path
        ckpt_path = os.path.join(normalized_path, "gated_model.ckpt")
        ckpt_path = os.path.normpath(ckpt_path)

        # Ensure the checkpoint path is within the specified directory
        if not ckpt_path.startswith(normalized_path):
            raise ValueError("Invalid checkpoint path construction")

        # Construct and validate gates dictionary path
        gates_path = os.path.join(normalized_path, 'gates_dictionary')
        gates_path = os.path.normpath(gates_path)

        # Ensure the gates path is within the specified directory
        if not gates_path.startswith(normalized_path):
            raise ValueError("Invalid gates dictionary path construction")

        save_checkpoint(model, ckpt_path)
        with open(gates_path, 'ab') as f:
            pickle.dump(gates_dict, f)
