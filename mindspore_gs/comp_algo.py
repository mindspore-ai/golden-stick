# Copyright 2022 Huawei Technologies Co., Ltd
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
"""GoldenStick."""
import numbers
import os.path

import numpy as np

import mindspore.dataset
from mindspore.nn.cell import Cell
from mindspore.train.callback import Callback
from mindspore import export, Tensor
from mindspore._checkparam import Validator
from mindspore import log as logger


class CompAlgo:
    """
    Base class of algorithms in GoldenStick.

    Args:
        config (Dict): User config for network compression. Config specification is default by derived class.
    """

    def __init__(self, config):
        if config is None:
            config = {}
        Validator.check_value_type("config", config, [dict], self.__class__.__name__)
        self._config = config
        self._save_mindir = False
        self._save_mindir_path = None
        self._save_mindir_inputs = None

    def apply(self, network: Cell) -> Cell:
        """
        Define how to compress input `network`. This method must be overridden by all subclasses.

        Args:
            network (Cell): Network to be compressed.

        Returns:
            Compressed network.
        """

        return network

    def callbacks(self, *args, **kwargs) -> [Callback]:
        """
        Define what task need to be done when training. Must be called at the end of child callbacks.

        Args:
            args (Union[list, tuple, optional]): Arguments passed to the function.
            kwargs (Union[dict, optional]): The keyword arguments.

        Returns:
            List of instance of Callbacks.
        """

        cb = []
        if self._save_mindir:
            if self._save_mindir_path is None or self._save_mindir_inputs is None:
                raise RuntimeError(f"When you want to export MindIR automatically, after setting save_mindir True, "
                                   f"mindir_path and mindir_inputs should also be set, but got mindir_path "
                                   f"{self._save_mindir_path}, mindir_inputs {self._save_mindir_inputs}.")
            cb.append(ExportMindIRCallBack(self, self._save_mindir_path, self._save_mindir_inputs))
        return cb

    def set_save_mindir(self, save_mindir: bool):
        """
        Set whether to automatically export MindIR after training.

        Args:
            save_mindir (bool): If true, export MindIR automatically after training, else not.

        Raises:
            TypeError: If `need_save` is not bool.
        """
        Validator.check_bool(save_mindir, "save_mindir", self.__class__.__name__)
        self._save_mindir = save_mindir

    def set_save_mindir_path(self, save_mindir_path: str):
        """
        Set the path to export MindIR, only takes effect if `save_mindir` is True.

        Args:
            save_mindir_path (str): The path to export MindIR, the path includes the directory and file name, which can
                be a relative path or an absolute path, the user needs to ensure write permission.

        Raises:
            TypeError: if `save_mindir_path` is not str.
        """
        if not self._save_mindir:
            logger.warning("When you want to export MindIR automatically, 'save_mindir' should be set True before "
                           "setting MindIR path")
        if not isinstance(save_mindir_path, str):
            raise TypeError(f"For {self.__class__.__name__}, 'save_mindir_path' should be string but got "
                            f"{type(save_mindir_path).__name__}.")
        self._save_mindir_path = os.path.realpath(save_mindir_path)

    @staticmethod
    def _check_input_dataset(*dataset, dataset_type):
        """Input dataset check."""
        if not dataset:
            return False
        for item in dataset:
            if not isinstance(item, dataset_type):
                return False
        return True

    @staticmethod
    def _check_inputs(*inputs):
        """Check inputs."""
        if CompAlgo._check_input_dataset(*inputs, dataset_type=mindspore.dataset.Dataset):
            if len(inputs) != 1:
                raise RuntimeError(
                    f"You can only serialize one dataset into MindIR, got " + str(len(inputs)) + " datasets")
            shapes, types, columns = inputs[0].output_shapes(), inputs[0].output_types(), inputs[0].get_col_names()
            only_support_col = "image"
            inputs_col = list()
            for c, s, t in zip(columns, shapes, types):
                if only_support_col != c:
                    continue
                inputs_col.append(Tensor(np.random.uniform(-1.0, 1.0, size=s).astype(t)))
            if not inputs_col:
                raise RuntimeError(f"Only supports parse \"image\" column from dataset now, given dataset has columns: "
                                   + str(columns))
        elif inputs[0] is None:
            raise RuntimeError("When you want to set inputs for exporting MindIR, inputs should not be None")
        else:
            if not isinstance(*inputs, (Tensor, list, tuple, numbers.Number, bool)):
                raise RuntimeError(f"When you want to set inputs for exporting MindIR, inputs should be tensor, list, "
                                   f"type, Number, bool or Dataset, but got {type(*inputs).__class__.__name__}")

    def set_save_mindir_inputs(self, *inputs):
        """
        Takes effect when `save_mindir` is True, set inputs for exporting MindIR.

        Args:
            inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
                 of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
                 it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
                 In second situation, you should adjust batch size of dataset script manually which will impact on
                 the batch size of 'net' input. Only supports parse "image" column from dataset currently.

        Raises:
            ValueError: If `inputs` is None.
        """
        if not self._save_mindir:
            logger.warning("When you want to export MindIR automatically, 'save_mindir' should be set True before "
                           "setting MindIR inputs")
        CompAlgo._check_inputs(*inputs)
        self._save_mindir_inputs = inputs[0]

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a compressed network to a standard network before exporting to MindIR.

        Args:
            net_opt (Cell): Network to be converted which is transformed by `CompAlgo.apply`.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is a empty string which means not loading
                checkpoint file to `net_opt`.

        Returns:
            An instance of Cell represents converted network.
        """

        return net_opt

    def loss(self, loss_fn: callable) -> callable:
        """
        Define how to adjust loss-function for algorithm. Subclass is not need to overridden this method if current
        algorithm not care loss-function.

        Args:
            loss_fn (callable): Original loss function.

        Returns:
            Adjusted loss function.
        """

        return loss_fn


class ExportMindIRCallBack(Callback):
    """Export MindIR after training automatically."""

    def __init__(self, algo: CompAlgo, save_mindir_path: str, inputs):
        """
        Init callback.

        Args:
            algo (CompAlgo): Mindspore Golden stick algorithm.
            save_mindir_path (str): The path to export MindIR, the path includes the directory and file name, which can
                be a relative path or an absolute path, the user needs to ensure write permission.
            inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
                 of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
                 it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
                 In second situation, you should adjust batch size of dataset script manually which will impact on
                 the batch size of 'net' input. Only supports parse "image" column from dataset currently.

        Raises:
            RuntimeError: If `save_mindir_path` is not set.
            ValueError: If `inputs` is None.
        """
        self._algo = algo
        self._save_mindir_path = save_mindir_path
        self._inputs = inputs

    def on_train_end(self, run_context):
        """Called on train end, convert net and export MindIR."""
        cb_params = run_context.original_args()
        net_deploy = self._algo.convert(cb_params.network)
        export(net_deploy, self._inputs, file_name=self._save_mindir_path, file_format="MINDIR")
