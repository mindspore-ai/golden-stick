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
"""algorithm related configs"""

from dataclasses import dataclass, field, is_dataclass, asdict
from typing import List

from mindspore_gs.common.config import GSBaseConfig
from mindspore_gs.common.utils import value_check
from mindspore_gs.common.register import RegisterMachine
from mindspore_gs.common.gs_enum import GSPTQApproach, GSQuantCellType, GSQuantDtype

algo_cfg_register = RegisterMachine()


@algo_cfg_register.register(GSPTQApproach.SMOOTH_QUANT.value)
@dataclass
class SmoothQuantConfig:
    """config for smooth quant algorithm"""
    alpha: float = 0.5
    is_deploy: bool = False

    def __post_init__(self):
        value_check('alpha', self.alpha, float)
        value_check('is_deploy', self.is_deploy, bool)


@algo_cfg_register.register(GSPTQApproach.RTN.value)
@dataclass
class RTNConfig:
    """
    Config for round to nearest algorithms
    """


@dataclass
class QuantizerConfig(GSBaseConfig):
    """ quantize related config """
    bit_num: int = 8
    optypes_exclude_output_quant: List[str] = field(default_factory=lambda: [])
    algo_args: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        value_check('bit_num', self.bit_num, int)
        value_check('optypes_exclude_output_quant', self.optypes_exclude_output_quant, str)
        value_check('algo_args', self.algo_args, dict)


@dataclass
class PTQConfig(QuantizerConfig):
    """
    config for post-trainning-quantizer
    Example:
        >>> from mindspore_gs import PTQConfig
        >>> smooth_quant_config = PTQConfig(approach='smooth_quant',
        >>>                                 algo_args={'alpha': 0.5, 'is_deploy': True})
    """
    approach: str = field(default=GSPTQApproach.RTN.value,
                          metadata={'valid_values': [
                              item.value for item in GSPTQApproach.__members__.values()
                          ]})
    calibration_sampling_size: int = 0
    act_quant_dtype: str = field(default=GSQuantDtype.int8.value,
                                 metadata={'valid_values': [
                                     item.value for item in GSQuantDtype.__members__.values()
                                 ]})
    weight_quant_dtype: str = field(default=GSQuantDtype.int8.value,
                                    metadata={'valid_values': [
                                        item.value for item in GSQuantDtype.__members__.values()
                                    ]})
    weight_only: bool = False
    act_per_channel: bool = False
    weight_per_channel: bool = True
    act_symmetric: bool = False
    weight_symmetric: bool = True
    act_narrow_range: bool = False
    weight_narrow_range: bool = False
    op_types: List[str] = field(default_factory=lambda: [GSQuantCellType.MF_LINEAR.value],
                                metadata={'choices': [
                                    item.value for item in GSQuantCellType.__members__.values()
                                ]})

    def __post_init__(self):
        value_check('calibration_sampling_size', self.calibration_sampling_size, int)
        value_check('act_quant_dtype', self.act_quant_dtype, str)
        value_check('weight_quant_dtype', self.weight_quant_dtype, str)
        value_check('weight_only', self.weight_only, bool)
        value_check('act_per_channel', self.act_per_channel, bool)
        value_check('weight_per_channel', self.weight_per_channel, bool)
        value_check('act_symmetric', self.weight_symmetric, bool)
        value_check('act_narrow_range', self.act_narrow_range, bool)
        value_check('weight_narrow_range', self.weight_narrow_range, bool)
        if self.approach not in {item.value for item in GSPTQApproach.__members__.values()}:
            raise ValueError(f'Invalid approach: {self.approach}')
        support_op_types = {
            item.value for item in GSQuantCellType.__members__.values()
        }
        for op_type in self.op_types:
            if op_type not in support_op_types:
                raise ValueError(f'{op_type} is not supported, all support type is {support_op_types}')
        if not self.algo_args:
            args_config = algo_cfg_register[self.approach]
            if args_config is not None and is_dataclass(args_config):
                self.algo_args.update(asdict(args_config()))

    def value_check(self):
        self.__post_init__()
