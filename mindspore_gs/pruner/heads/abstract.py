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
definition of pruner abstract class
"""
from abc import ABC, abstractmethod
from mindspore.nn import Cell
from mindspore_gs import CompAlgo
from mindspore_gs.validator import Validator


class AbstractHeadPruner(CompAlgo, ABC):
    """
    Abstract head prunner class, defines behaviour only.
    """

    def __init__(self, config):
        """
        Initialize Head Pruner
        Args:
            config: configuration parameters
        """
        super(AbstractHeadPruner, self).__init__(config)
        Validator.check_value_type("config", config, [dict], self.__class__.__name__)
        self.model_config = config['model_config']
        self.l0_penalty = config['l0_penalty']

    def apply(self, network: Cell, **kwargs) -> Cell:
        """
        Define how to compress input `network`. This method must be overridden by all subclasses.

        Args:
            network (Cell): Network to be compressed.

        Returns:
            Compressed network.
        """
        Validator.check_value_type("network", network, [Cell])
        self._init_head(network)
        return self._decorate_model(self.l0_penalty)

    def convert(self, net_opt: Cell, ckpt_path="") -> Cell:
        """
        Define how to convert a compressed network to a standard network before exporting to MindIR.

        Args:
            net_opt (Cell): Network to be converted which is transformed by `AbstractHeadPruner.apply`.
            ckpt_path (str): Path to checkpoint file for `net_opt`. Default is ``""``, which means not loading
                checkpoint file to `net_opt`.

        Returns:
            An instance of Cell represents converted network.

        Raises:
            TypeError: If `net_opt` is not Cell.
            TypeError: If `ckpt_path` is not string.
            ValueError: If `ckpt_path` is not empty and invalid.

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
        Validator.check_value_type("ckpt_path", ckpt_path, [str], self.__class__.__name__)
        Validator.check_value_type("net_opt", net_opt, [Cell])
        return self._prune_model(net_opt, ckpt_path)

    @abstractmethod
    def _init_head(self, model):
        """
            check if model has a head, save the model.
            Args:
                model: model to save
        """

    @abstractmethod
    def _decorate_model(self, l0_penalty):
        """
        decorate model, repackage the model with additional functionality.
        Args:
            l0_penalty: penalty value for gate calculation.

        Returns:

        """

    @abstractmethod
    def _prune_model(self, model, save_dir_path=None):
        """
        Prune the model, after training/fine-tuning.
        Args:
            model: that has been decorated.
            save_dir_path (optional): path to save the models and heads dictionary

        Returns: pruned & clean model.

        """
