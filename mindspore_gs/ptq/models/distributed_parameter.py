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
"""DistributedParameter"""

from mindspore import Parameter, ops as msops
from mindspore.common.dtype import type_size_in_bytes
from mindspore_gs.common import logger


class DistributedParameter:
    """DistributedParameter"""
    def __init__(self, param: Parameter, axis=None):
        self.param: Parameter = param
        self.axis = axis
        self._is_comm_ed = False

    def comm(self):
        """comm"""
        if self._is_comm_ed:
            return
        self._is_comm_ed = True
        if self.axis is None:
            logger.debug(f"parameter {self.param.name} no need to tensor-parallel-merge")
            return
        logger.debug(f"tensor-parallel-merge for parameter {self.param.name} by axis {self.axis}")
        perm = []
        for i in range(len(self.param.shape)):
            if i == 0:
                perm.append(self.axis)
            elif i == self.axis:
                perm.append(0)
            else:
                perm.append(i)
        perm = tuple(perm)
        param = msops.transpose(self.param, perm)
        param = msops.AllGather()(param)
        param = msops.transpose(param, perm)
        self.param = Parameter(param, self.param.name)

    def size(self):
        """_cal_size"""
        result = type_size_in_bytes(self.param.dtype)
        for ele in self.param.shape:
            result = result * ele
        return result

    def __str__(self):
        return f"DistributedParameter(param({self.param.name}, " \
               f"{self.param.shape}), axis({self.axis}))"

    def __repr__(self):
        return self.__str__()
