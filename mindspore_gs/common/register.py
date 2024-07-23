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
registers for ptq
"""
from typing import Dict

from mindformers.modules.layers import Linear
import mindspore.log as logger
import mindspore.nn as nn
from mindspore_gs.common.gs_enum import QuantCellType


cell_type_dicts = {
    QuantCellType.LINEAR.value: nn.Dense,
    QuantCellType.CONV2D.value: nn.Conv2d,
    QuantCellType.MF_LINEAR.value: Linear
}


class RegisterMachine:
    """register machine class"""
    def __init__(self, name: str = None):
        self._name = name
        self._name_method_dict = dict()

    def __getitem__(self, item):
        return self._name_method_dict.get(item, None)

    def __repr__(self):
        return f'Registered methods: {self._name_method_dict}'

    @property
    def registered_method(self) -> Dict:
        """registered method for register machine"""
        return self._name_method_dict

    def register(self, name=None):
        """register method for register machine"""
        def wrapper(func):
            if name is None:
                func_name = func.__name__
            else:
                func_name = name
            if func_name not in self._name_method_dict:
                self._name_method_dict[func_name] = func
            else:
                logger.warning(f'{name} is already registered, would skip this register')
            return func
        return wrapper
