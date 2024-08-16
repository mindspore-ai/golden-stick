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
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.network_helpers import NetworkHelper


class WrapperCell(abc.ABC, Cell):
    """WrapperCell"""
    def __init__(self, cfg: InnerPTQConfig):
        super().__init__()
        self.cfg = cfg

    @abc.abstractmethod
    def process(self):
        raise NotImplementedError

    @abc.abstractmethod
    def deploy(self):
        raise NotImplementedError


class WrapperLinearCell(WrapperCell):
    """WrapperLinearCell"""
    def __init__(self, linear_name: str, linear, cfg: InnerPTQConfig, network_helper: NetworkHelper):
        super().__init__(cfg)
        self._linear_name = linear_name
        self._linear = linear
        self.net_helper = network_helper
        self.samples = []
        self.cat_samples = None

    @property
    def linear(self):
        return self._linear

    @property
    def linear_name(self):
        return self._linear_name

    def process(self):
        if not self.samples:
            raise RuntimeError(f"Please catch matmul inputs before quantization.")
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        self.samples.clear()

    @abc.abstractmethod
    def deploy(self):
        raise NotImplementedError

    def construct(self, x, **kwargs):
        """construct"""
        class CatchInputMatmul(Cell):
            def __init__(self, matmul, samples):
                super().__init__()
                self.mm = matmul
                self.samples = samples

            def construct(self, x, weight):
                self.samples.append(x)
                return self.mm(x, weight)

        class MatmulCell(Cell):
            def __init__(self, matmul):
                super().__init__()
                self.mm = matmul

            def construct(self, *args, **kwargs):
                return self.mm(*args, **kwargs)

        self._linear.matmul = CatchInputMatmul(self._linear.matmul, self.samples)
        output = self._linear(x, **kwargs)
        self._linear.matmul = MatmulCell(self._linear.matmul.mm)
        return output
