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
"""functional unit cells."""

from mindspore.nn import Cell
from mindspore import Parameter
from mindspore import ops as msops
from mindspore.common.initializer import initializer


class MatmulCellForHook(Cell):
    """MatmulCellForHook"""
    def __init__(self, matmul):
        super().__init__()
        self.mm = matmul

    def construct(self, *args, **kwargs):
        return self.mm(*args, **kwargs)


class SmoothMatmul(Cell):
    """SmoothMatmul"""
    def __init__(self, mm, smooth_scale_):
        super().__init__()
        self.mm = mm
        self.smooth_scale = Parameter(msops.div(1, smooth_scale_))

    def construct(self, x, weight):
        x = msops.mul(x, self.smooth_scale)
        return self.mm(x, weight)


class SmoothMatmulForDeploy(Cell):
    """SmoothMatmulForDeploy"""
    def __init__(self, mm, ic_, compute_dtype_):
        super().__init__()
        self.mm = mm
        self.smooth_scale = Parameter(initializer('ones', (ic_,), dtype=compute_dtype_))

    def construct(self, x, weight):
        x = msops.mul(x, self.smooth_scale)
        return self.mm(x, weight)


def matmul_cell_for_hook2smooth_matmul(src: MatmulCellForHook, smooth_scale):
    """matmul_cell_for_hook2smooth_matmul"""
    if not isinstance(src.mm, msops.MatMul):
        raise ValueError(f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
    return SmoothMatmul(src.mm, smooth_scale)


def matmul_cell_for_hook2smooth_matmul_for_deploy(src: MatmulCellForHook, ic, compute_dtype):
    """matmul_cell_for_hook2smooth_matmul_for_deploy"""
    if not isinstance(src.mm, msops.MatMul):
        raise ValueError(f'matmul of MatmulCellForHook should be an instance of {msops.MatMul}, but got {src.mm}.')
    return SmoothMatmulForDeploy(src.mm, ic, compute_dtype)


def create_matmul_wrapper(src, dst_type, **kwargs):
    """create_matmul_wrapper"""
    transform_map = {
        (MatmulCellForHook, SmoothMatmul): matmul_cell_for_hook2smooth_matmul,
        (MatmulCellForHook, SmoothMatmulForDeploy): matmul_cell_for_hook2smooth_matmul_for_deploy
    }
    fn = transform_map.get((type(src), dst_type))
    if not fn:
        raise RuntimeError(f"Not support transform cell from {type(src)} to {dst_type}")
    return fn(src, **kwargs)
