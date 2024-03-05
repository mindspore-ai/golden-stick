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
"""
enum classes in golden-stick
"""

from enum import Enum


class PTQApproach(Enum):
    """
    PTQ approach enums
    """
    SMOOTH_QUANT = 'smooth_quant'
    RTN = 'rtn'
    GPTQ = 'gptq'


class QATApproach(Enum):
    """
    QAT approach enums
    """
    LSQ = 'lsq'
    SLB = 'slb'


class QuantCellType(Enum):
    """
    supported quant cell type enums
    """
    LINEAR = 'linear'
    CONV2D = 'conv2d'
    MF_LINEAR = 'mf_linear'


class GSQuantDtype(Enum):
    """
    supported quant dtype
    """
    int4 = 'INT4'
    uint4 = 'UINT4'
    int8 = 'INT8'
    uint8 = 'UINT8'
    int16 = 'INT16'
    UINT16 = 'UINT16'


class PTQMode(Enum):
    """
    Mode for ptq quantizer.
    QUANTIZE: indicate ptq quantizer in quantize mode.
    DEPLOY: indicate ptq quantizer in deploy mode.
    """
    QUANTIZE = 'quantize'
    DEPLOY = 'deploy'


class BackendTarget(Enum):
    """
    Mindspore backend target for cell convert.
    NONE: indicate target cell is not for specific backend.
    ASCEND: indicate target cell is for ascend backend.
    """
    NONE = 'none'
    ASCEND = 'ascend'
