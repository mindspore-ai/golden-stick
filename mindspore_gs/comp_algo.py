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

from mindspore.nn.cell import Cell
from mindspore.train.callback import Callback


class CompAlgo:
    """
    Base class of algorithms in GoldenStick.

    Args:
        config (Dict): User config for network compression. Config specification is default by derived class.
    """

    def __init__(self, config):
        self._config = config

    def apply(self, network: Cell) -> Cell:
        """
        Define how to compress input `network`. This method must be overridden by all subclasses.

        Args:
            network (Cell): Network to be compressed.

        Returns:
            Compressed network.
        """

        return network

    def callback(self) -> Callback:
        """
        Define what task need to be done when training for QAT.

        Returns:
            Instance of Callback
        """

        return Callback()

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
