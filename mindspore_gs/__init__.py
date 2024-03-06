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
MindSpore golden stick module.
"""

from .version import __version__, mindspore_version_check
mindspore_version_check()

from .comp_algo import CompAlgo, Backend
from .quantization import SimulatedQuantizationAwareTraining, SlbQuantAwareTraining
from .pruner import PrunerKfCompressAlgo, PrunerFtCompressAlgo, UniPruner
from .ghost import GhostAlgo
from .ptq.ptq_config import PTQConfig
from .common.gs_enum import PTQMode, BackendTarget

__all__ = ["SimulatedQuantizationAwareTraining", "SlbQuantAwareTraining", "PrunerKfCompressAlgo",
           "PrunerFtCompressAlgo", "UniPruner", "CompAlgo", "GhostAlgo", 'PTQConfig', 'PTQMode',
           'BackendTarget']
__all__.extend(__version__)
