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
"""algorithm related configs."""

from dataclasses import dataclass
from enum import Enum

from mindspore_gs.common.utils import value_check
from mindspore_gs.common.gs_enum import BackendTarget

class RAMode(Enum):
    """
    Mode for razorattention algorithm.

    - ``SEARCH_RETRIEVAL``: indicate razorattention in SEARCH_RETRIEVAL mode.
    - ``DEPLOY``: indicate razorattention in deploy mode.
    """
    SEARCH_RETRIEVAL = 'search_retrieval'
    DEPLOY = 'deploy'


@dataclass
class RAConfig:
    """
    Config for razorattention algorithm.

    Args:

    Raises:

    Examples:
        >>> from mindspore_gs.ptq import RAConfig, RAMode
        >>> from mindspore_gs.common import BackendTarget
        >>> RAConfig(mode=RAMode.DSEARCH_RETRIEVAL, backend=BackendTarget.ASCEND)
        RAConfig(mode=<RAMode.DEPLOY: 'deploy'>,
        backend=<BackendTarget.ASCEND: 'ascend'>,
        retrieval_head_path="path/to/retrieval_head")
    """
    mode: RAMode = RAMode.SEARCH_RETRIEVAL
    backend: BackendTarget = BackendTarget.ASCEND
    echo_head_ratio: float = 0.01
    induction_head_ratio: float = 0.14
    retrieval_head_path: str = ""
    sink_size: int = 4
    local_capacity: int = 256
    use_virtual_token: bool = True

    def __post_init__(self):
        if self.mode not in RAMode.__members__.values():
            raise ValueError(f'mode shall be in {RAMode.__members__.values()}')
        if self.backend not in BackendTarget.__members__.values():
            raise ValueError(f'backend shall be in {BackendTarget.__members__.values()}')
        value_check('echo_head_ratio', self.echo_head_ratio, float)
        value_check('induction_head_ratio', self.induction_head_ratio, float)
        value_check('retrieval_head_path', self.retrieval_head_path, str)
        value_check('sink_size', self.sink_size, int)
        value_check('local_capacity', self.local_capacity, int)
        value_check('use_virtual_token', self.use_virtual_token, bool)
