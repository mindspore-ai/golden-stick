# Copyright 2025 Huawei Technologies Co., Ltd
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
"""mindone linear wrapper cell."""
import abc

from mindspore import ops as msops

from mindspore_gs.ptq.quant_cells.quant_cell import QuantCell


class WrapperLinearCell(QuantCell, abc.ABC):
    """WrapperCell"""

    def __init__(self, linear_name, linear, context, cfg, **kwargs):
        super().__init__(linear_name, linear, context, cfg, **kwargs)
        self.transpose_b = True

    # pylint: disable=arguments-differ
    def add_hook(self):
        """add_hook"""
        class HookMatMul(msops.MatMul):
            def __call__(self, *args, **kwargs):
                x = args[0]
                self.samples.append(msops.squeeze(x))
                return msops.MatMul.__call__(self, *args, **kwargs)

        class HookDense(msops.Dense):
            def __call__(self, *args, **kwargs):
                x = args[0]
                self.samples.append(msops.squeeze(x))
                return msops.Dense.__call__(self, *args, **kwargs)

        # what Python really does to __call__:
        # type(a).__call__(a)
        # as such, if I want to override the __call__ method, I must override the __call__ of a class
        # but if I don't want to affect behaviour of other instances of the same class,
        # I need to create a new class with the overridden __call__ method.
        matmul = self.layer.matmul if hasattr(self.layer, "matmul") else self.layer.dense
        if isinstance(matmul, msops.MatMul):
            matmul.__class__ = HookMatMul
            matmul.layer_name = self.layer_name
            matmul.samples = self.samples
        elif isinstance(matmul, msops.Dense):
            matmul.__class__ = HookDense
            matmul.layer_name = self.layer_name
            matmul.samples = self.samples
        else:
            raise RuntimeError(f"Unsupported matmul type for hook: {type(matmul)}")

    # pylint: disable=arguments-differ
    def remove_hook(self):
        """remove_hook"""
        matmul = self.layer.matmul if hasattr(self.layer, "matmul") else self.layer.dense
        if isinstance(matmul, msops.MatMul):
            matmul.__class__ = msops.MatMul
        elif isinstance(matmul, msops.Dense):
            matmul.__class__ = msops.Dense
        else:
            raise RuntimeError(f"Unsupported matmul type for hook: {type(matmul)}")
