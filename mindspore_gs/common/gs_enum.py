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


class QATApproach(Enum):
    """
    QAT approach enums
    """
    LSQ = 'lsq'
    SLB = 'slb'


class BackendTarget(Enum):
    """
    Mindspore backend target for cell convert.

    - ``NONE``: indicate target cell is not for specific backend.
    - ``ASCEND``: indicate target cell is for ascend backend.
    """
    NONE = 'none'
    ASCEND = 'ascend'
