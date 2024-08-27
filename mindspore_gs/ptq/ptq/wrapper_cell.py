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
from mindspore.nn import Cell
from mindspore import ops as msops
from mindspore_gs.ptq.ptq_config import InnerPTQConfig, PTQMode
from mindspore_gs.ptq.network_helpers import NetworkHelper


class WrapperCell(abc.ABC, Cell):
    """WrapperCell"""

    class MatmulCell(Cell):
        def __init__(self, matmul):
            super().__init__()
            self.mm = matmul

        def construct(self, *args, **kwargs):
            return self.mm(*args, **kwargs)

    def __init__(self, layer_name: str, layer, cfg: InnerPTQConfig, network_helper: NetworkHelper):
        super().__init__()
        self.cfg = cfg
        self._layer_name = layer_name
        self._layer = layer
        self.net_helper = network_helper
        self.samples = []
        self.cat_samples = None
        if self.cfg.mode == PTQMode.QUANTIZE:
            self._layer.matmul = WrapperCell.MatmulCell(self._layer.matmul)

    @property
    def layer(self):
        return self._layer

    @property
    def layer_name(self):
        return self._layer_name

    def process(self):
        if not self.samples:
            raise RuntimeError(f"Please catch matmul inputs before quantization.")
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        self.samples.clear()

    def add_hook(self):
        def hook_fn(_, inps):
            x = inps[0]
            self.samples.append(msops.squeeze(x))
        self._layer.matmul.register_forward_pre_hook(hook_fn)

    def remove_hook(self):
        self._layer.matmul = WrapperCell.MatmulCell(self._layer.matmul.mm)

    @abc.abstractmethod
    def deploy(self):
        raise NotImplementedError

    def construct(self, x, **kwargs):
        """construct"""
        return self._layer(x, **kwargs)
