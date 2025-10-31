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
from mindspore import dtype
from mindspore.nn import Cell
from mindspore import ops as msops
from mindspore_gs.ptq.context import InnerPTQConfig, OutliersSuppressionType, QuantGranularity


class Checker:
    def check(self, config: InnerPTQConfig):
        """check"""
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
                 **kwargs):
        super().__init__()
        self.context = context
        self.cfg = cfg
        self._layer_name = layer_name
        self._layer = layer
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
        """process"""
        if not self.samples:
            raise RuntimeError(f"Please catch matmul inputs before quantization.")
        # for moe matmul in the mindone, x shape may be(0, ic) or (ic) or (bs, ic)
        # so we only need tensor.shape[0] > 0, and we also need to reshape (ic) to (1, ic),
        # for the concat ops.
        self.samples = [(tensor if len(tensor.shape) > 1 else tensor.reshape(1, -1))
                        for tensor in self.samples if tensor.shape[0] > 0]
        # in the moe matmul in the mindone, len(self.samples) may be 0 after filter of
        # "tensor.shape[0] > 0"
        if len(self.samples) == 0:
            if (self.cfg.act_quant_dtype == dtype.int8 and self.cfg.act_quant_granularity != \
                QuantGranularity.PER_TOKEN) or self.cfg.outliers_suppression != OutliersSuppressionType.NONE:
                raise ValueError("when act_quant_dtype is dtype.int8 and act_quant_granularity != QuantGranularity.PER_TOKEN,"
                                 "or outliers_suppression isn't OutliersSuppressionType.NONE,"
                                 f"len(self.samples) of {self._layer_name} can't be 0.")
            else:
                self.cat_samples = None
                return
        self.cat_samples = msops.cat(tuple(self.samples), axis=0)
        self.samples.clear()

    def add_hook(self, experimental=False):
        """add_hook"""
        raise NotImplementedError

    def remove_hook(self, experimental=False):
        """remove_hook"""
        raise NotImplementedError

    @abc.abstractmethod
    def deploy(self):
        """deploy"""
        raise NotImplementedError

    def construct(self, x, *args, **kwargs):
        """construct"""
        return self._layer(x, *args, **kwargs)
