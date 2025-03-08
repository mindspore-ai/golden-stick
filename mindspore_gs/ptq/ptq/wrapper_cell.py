# Copyright 2024 Huawei Technologies Co., Ltd
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
"""ptq wrapper cell base class."""

import abc
import dataclasses
from mindspore.nn import Cell
from mindspore import ops as msops
from mindspore_gs.ptq.context import InnerPTQConfig
from mindspore_gs.ptq.network_helpers import NetworkHelper


class Checker:
    def check(self, config: InnerPTQConfig):
        raise NotImplementedError


@dataclasses.dataclass
class SearchInputs:
    layer: Cell
    layer_args: []
    layer_kwargs: {}


class WrapperCell(abc.ABC, Cell):
    """WrapperCell"""

    # pylint: disable=W0613
    def __init__(self, layer_name: str, layer, context: InnerPTQConfig, cfg: InnerPTQConfig,
                 network_helper: NetworkHelper, **kwargs):
        super().__init__()
        self.cfg = cfg
        self._layer_name = layer_name
        self._layer = layer
        self.net_helper = network_helper
        self.samples = []
        self.cat_samples = None
        self.group_list = None
        context.report_quant_info(layer_name, self._quant_info())

    def _quant_info(self) -> str:
        raise NotImplementedError

    @property
    def layer(self):
        """layer"""
        return self._layer

    @property
    def layer_name(self):
        """layer_name"""
        return self._layer_name

    def process(self):
        if not self.samples:
            raise RuntimeError(f"Please catch matmul inputs before quantization.")
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        self.samples.clear()

    def add_hook(self):
        """add_hook"""
        raise NotImplementedError

    def remove_hook(self):
        """remove_hook"""
        raise NotImplementedError

    @abc.abstractmethod
    def deploy(self):
        """deploy"""
        raise NotImplementedError

    def construct(self, x, *args, **kwargs):
        """construct"""
        return self._layer(x, *args, **kwargs)
