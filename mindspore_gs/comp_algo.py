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
import os.path

from mindspore.nn.cell import Cell
from mindspore.train.callback import Callback
from mindspore import export


class CompAlgo:
    """
    Base class of algorithms in GoldenStick.

    Args:
        config (dict): User config for network compression. Config specification is default by derived class.
    """

    def __init__(self, config):
        self._config = config
        self._save_mindir = False
        self._save_mindir_path = ""
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
            cb.append(ExportMindIRCallBack(self, self._save_mindir_path, self._save_mindir_inputs))
        return cb

    def set_save_mindir(self, save_mindir: bool):
        """
        Set whether to automatically export MindIR after training.

        Args:
            save_mindir (bool): If true, export MindIR automatically after training, else not.

        Raises:
            TypeError: If `need_save` is not bool.

        Examples:
            >>> from mindspore import Tensor, nn
            >>> from mindspore_gs.comp_algo import CompAlgo
            >>> import numpy as np
            >>> ## 1) Define network to be trained
            >>> network = LeNet(10)
            >>> ## 2) Define MindSpore Golden Stick Algorithm, here we use base algorithm.
            >>> algo = CompAlgo({})
            >>> ## 3) Enable automatically export MindIR after training.
            >>> algo.set_save_mindir(save_mindir=True)
            >>> ## 4) Set MindIR output path.
            >>> algo.set_save_mindir_path(save_mindir_path="./lenet")
            >>> ## 5) Set the inputs of the net for exoprting MindIR.
            >>> input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32))
            >>> algo.set_save_mindir_inputs(input_tensor)
            >>> ## 6) Apply MindSpore Golden Stick algorithm to origin network.
            >>> network = algo.apply(network)
            >>> ## 7) Then you can start training and MindIR with be exported automatically after training ends.
        """
        if not isinstance(save_mindir, bool):
            raise TypeError("Input `need_save` should be bool.")
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
        if not isinstance(save_mindir_path, str):
            raise TypeError("Input `save_mindir_path` should be str.")
        if not self._save_mindir:
            return
        self._save_mindir_path = os.path.realpath(save_mindir_path)

    def set_save_mindir_inputs(self, inputs):
        """
        Takes effect when `save_mindir` is True, set inputs for exporting MindIR.

        Args:
            inputs (Union[Tensor, Dataset, List, Tuple, Number, Bool]): It represents the inputs
                 of the `net`, if the network has multiple inputs, set them together. While its type is Dataset,
                 it represents the preprocess behavior of the `net`, data preprocess operations will be serialized.
                 In second situation, you should adjust batch size of dataset script manually which will impact on
                 the batch size of 'net' input. Only supports parse "image" column from dataset currently.

        Raises:
            RuntimeError: If `inputs` is None.
            RuntimeError: If `inputs` has more than one Dataset.
            RuntimeError: If `inputs` is Dataset but the column is not "image".
            RuntimeError: If type of `inputs` is not in [Tensor, Dataset, List, Tuple, Number, Bool].
        """
        if inputs is None:
            raise ValueError("Inputs is None.")
        self._save_mindir_inputs = inputs

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a compressed network to a standard network before exporting to MindIR.

        Args:
            net_opt (Cell): Network to be converted which is transformed by `CompAlgo.apply`.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is a empty string which means not loading
                checkpoint file to `net_opt`.

        Returns:
            An instance of Cell represents converted network.

        Examples:
            >>> from mindspore_gs.comp_algo import CompAlgo
            >>> ## 1) Define network to be trained
            >>> network = LeNet(10)
            >>> ## 2) Define MindSpore Golden Stick Algorithm, here we use base algorithm.
            >>> algo = CompAlgo({})
            >>> ## 3) Apply MindSpore Golden Stick algorithm to origin network.
            >>> network = algo.apply(network)
            >>> ## 4) Then you can start training, after which you can convert a compressed network to a standard
            >>> ##    network, there are two ways to do that.
            >>> ## 4.1) Convert without checkpoint.
            >>> net_deploy = algo.convert(network)
            >>> ## 4.2) Convert with checkpoint.
            >>> net_deploy = algo.convert(network, ckpt_path)
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
        if not save_mindir_path:
            raise RuntimeError("Save MindIR path is not set when creating ExportMindIRCallBack.")
        self._save_mindir_path = save_mindir_path
        if inputs is None:
            raise ValueError("Inputs is none when creating ExportMindIRCallBack.")
        self._inputs = inputs

    def on_train_end(self, run_context):
        """Called on train end, convert net and export MindIR."""
        cb_params = run_context.original_args()
        net_deploy = self._algo.convert(cb_params.network)
        export(net_deploy, self._inputs, file_name=self._save_mindir_path, file_format="MINDIR")
