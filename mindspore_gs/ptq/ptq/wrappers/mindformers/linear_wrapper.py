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
"""mindformers linear wrapper cell."""
import abc
from typing import Optional

from mindspore import ops as msops
from mindspore.common.hook_handle import HookHandle
from mindspore_gs.ptq.ptq_config import InnerPTQConfig
from mindspore_gs.ptq.ptq.wrapper_cell import WrapperCell
from mindspore_gs.ptq.network_helpers import NetworkHelper
from mindspore_gs.ptq.ptq.quant_cell_units import MatmulCellForHook


class WrapperLinearCell(WrapperCell, abc.ABC):
    """WrapperCell"""
    def __init__(self, layer_name: str, layer, cfg: InnerPTQConfig, network_helper: NetworkHelper):
        super().__init__(layer_name, layer, cfg, network_helper)
        self.hook_handle: Optional[HookHandle] = None

    def add_hook(self):
        def hook_fn(_, inps):
            x = inps[0]
            self.samples.append(msops.squeeze(x))
        # mindspore can only hook cell.
        if isinstance(self._layer.matmul, msops.Primitive):
            self._layer.matmul = MatmulCellForHook(self._layer.matmul)
        self.hook_handle = self._layer.matmul.register_forward_pre_hook(hook_fn)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()
        # mindspore not support replace a cell with primitive, so MatmulCellForHook can not be removed here.

    @abc.abstractmethod
    def deploy(self):
        """deploy"""
        raise NotImplementedError
