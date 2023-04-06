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

import abc
import os.path

from mindspore.nn.cell import Cell
from mindspore.train.callback import Callback
from mindspore import export, context
from mindspore import log as logger
from mindspore_gs.validator import Validator


class CompAlgo(abc.ABC):
    """
    Base class of algorithms in GoldenStick.

    Args:
        config (dict): User config for network compression, default is None. Algorithm config specification is default
            by derived class, base attributes are listed below:

            - save_mindir (bool): If true, export MindIR automatically after training, else not. Default: False.
            - save_mindir_path (str): The path to export MindIR, the path includes the directory and file name, which
              can be a relative path or an absolute path, the user needs to ensure write permission.
              Default: './network'.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        Validator.check_value_type("config", config, [dict], self.__class__.__name__)
        self._is_cpu = context.get_context('device_target') == "CPU"
        self._config = None
        self._create_config()
        self._update_common_config(config)
        self._update_config_from_dict(config)

    def _create_config(self):
        """Create base config. If derived class has extra attributes, Should be over-writed."""
        self._config = CompAlgoConfig()

    def _update_common_config(self, config: dict):
        """Create base config from a dict."""
        self.set_save_mindir(config.get("save_mindir", False))
        if self._config.save_mindir:
            self.set_save_mindir_path(config.get("save_mindir_path", "./network"))

    def _update_config_from_dict(self, config: dict):
        """Update config for specific algo. If derived class has extra attributes, Should be over-writed."""

    @abc.abstractmethod
    def apply(self, network: Cell) -> Cell:
        """
        Define how to compress input `network`. This method must be overridden by all subclasses.

        Args:
            network (Cell): Network to be compressed.

        Returns:
            Compressed network.
        """

        raise NotImplementedError

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
        if self._config.save_mindir:
            cb.append(ExportMindIRCallBack(self, os.path.realpath(self._config.save_mindir_path)))
        return cb

    def set_save_mindir(self, save_mindir: bool):
        """
        Set whether to automatically export MindIR after training.

        Args:
            save_mindir (bool): If true, export MindIR automatically after training, else not.

        Raises:
            TypeError: If `need_save` is not bool.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore_gs.quantization import SimulatedQuantizationAwareTraining as SimQAT
            >>> import numpy as np
            >>> ## 1) Define network to be trained
            >>> network = LeNet(10)
            >>> ## 2) Define MindSpore Golden Stick Algorithm, here we use base algorithm.
            >>> algo = SimQAT()
            >>> ## 3) Enable automatically export MindIR after training.
            >>> algo.set_save_mindir(save_mindir=True)
            >>> ## 4) Set MindIR output path.
            >>> algo.set_save_mindir_path(save_mindir_path="./lenet")
            >>> ## 5) Apply MindSpore Golden Stick algorithm to origin network.
            >>> network = algo.apply(network)
            >>> ## 6) Set up Model.
            >>> train_dataset = create_custom_dataset()
            >>> net_loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
            >>> net_opt = ms.nn.Momentum(network.trainable_params(), 0.01, 0.9)
            >>> model = ms.Model(network, net_loss, net_opt, metrics={"Accuracy": ms.train.Accuracy()})
            >>> ## 7) Config callback in model.train, start training, then MindIR will be exported.
            >>> model.train(1, train_dataset, callbacks=algo.callbacks())
        """
        Validator.check_bool(save_mindir, "save_mindir", self.__class__.__name__)
        self._config.save_mindir = save_mindir

    def set_save_mindir_path(self, save_mindir_path: str):
        """
        Set the path to export MindIR, only takes effect if `save_mindir` is True.

        Args:
            save_mindir_path (str): The path to export MindIR, the path includes the directory and file name, which can
                be a relative path or an absolute path, the user needs to ensure write permission.

        Raises:
            ValueError: if `save_mindir_path` is not Non-empty str.
        """
        if not self._config.save_mindir:
            logger.warning("When you want to export MindIR automatically, 'save_mindir' should be set True before "
                           "setting MindIR path")
        if save_mindir_path is None or not isinstance(save_mindir_path, str) or save_mindir_path.strip() == "":
            raise ValueError(f"For {self.__class__.__name__}, 'save_mindir_path' should be Non-empty string but got"
                             f" {save_mindir_path}.")
        self._config.save_mindir_path = os.path.realpath(save_mindir_path)

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
            >>> from mindspore_gs.quantization import SimulatedQuantizationAwareTraining as SimQAT
            >>> ## 1) Define network to be trained
            >>> network = LeNet(10)
            >>> ## 2) Define MindSpore Golden Stick Algorithm, here we use base algorithm.
            >>> algo = SimQAT()
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


class CompAlgoConfig:
    """
    Config for CompAlgo.
    """

    def __init__(self):
        """Init with default value."""
        self.save_mindir = False
        self.save_mindir_path = "./network"


class ExportMindIRCallBack(Callback):
    """Export MindIR after training automatically."""

    def __init__(self, algo: CompAlgo, save_mindir_path: str):
        """
        Init callback.

        Args:
            algo (CompAlgo): Mindspore Golden stick algorithm.
            save_mindir_path (str): The path to export MindIR, the path includes the directory and file name, which can
                be a relative path or an absolute path, the user needs to ensure write permission.
        """
        self._algo = algo
        self._save_mindir_path = save_mindir_path

    def on_train_end(self, run_context):
        """Called on train end, convert net and export MindIR."""
        cb_params = run_context.original_args()
        net_deploy = self._algo.convert(cb_params.network)
        export(net_deploy, cb_params.train_dataset, file_name=self._save_mindir_path, file_format="MINDIR")
