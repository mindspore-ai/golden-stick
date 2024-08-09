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
"""sq quant cell."""


from mindspore import nn
from mindspore_gs.ptq.quant_cell import PTQCell


class SQCell(PTQCell):
    """SQCell"""
    def to_next_phase(self) -> nn.Cell:
        """to_next_phase"""
        raise NotImplementedError
