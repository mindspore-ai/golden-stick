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
"""
MindSpore Golden Stick nn ops.
"""

from .act_quant import ActQuant
from .batchnorm_fold_cell import BatchNormFoldCell
from .conv2d_bn_fold_quant import Conv2dBnFoldQuant
from .conv2d_bn_without_fold_quant import Conv2dBnWithoutFoldQuant
from .conv2d_bn_fold_quant_one_conv import Conv2dBnFoldQuantOneConv
from .conv2d_quant import Conv2dQuant
from .dense_quant import DenseQuant
from .fake_quant_with_min_max_observer import FakeQuantWithMinMaxObserver
from .mul_quant import MulQuant
from .tensor_add_quant import TensorAddQuant


__all__ = [
    'FakeQuantWithMinMaxObserver',
    'Conv2dBnFoldQuantOneConv',
    'Conv2dBnFoldQuant',
    'Conv2dBnWithoutFoldQuant',
    'Conv2dQuant',
    'DenseQuant',
    'ActQuant',
    'TensorAddQuant',
    'MulQuant',
]
