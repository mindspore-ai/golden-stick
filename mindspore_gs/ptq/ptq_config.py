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
from enum import Enum
from typing import List, Union

from mindspore import QuantDtype
from mindspore import dtype as msdtype

from mindspore_gs.common.config import GSBaseConfig
from mindspore_gs.common.utils import value_check
from mindspore_gs.common.register import RegisterMachine
from mindspore_gs.common.gs_enum import QuantCellType, BackendTarget

algo_cfg_register = RegisterMachine()


class PTQApproach(Enum):
    """
    PTQ approach enums
    """
    SMOOTH_QUANT = 'smooth_quant'
    RTN = 'rtn'
    GPTQ = 'gptq'


class PTQMode(Enum):
    """
    Mode for ptq quantizer.

    - ``QUANTIZE``: indicate ptq quantizer in quantize mode.
    - ``DEPLOY``: indicate ptq quantizer in deploy mode.
    """
    QUANTIZE = 'quantize'
    DEPLOY = 'deploy'


@algo_cfg_register.register(PTQApproach.SMOOTH_QUANT)
@dataclass
class SmoothQuantConfig:
    """config for smooth quant algorithm"""
    alpha: float = 0.5

    def __post_init__(self):
        value_check('alpha', self.alpha, float)


@algo_cfg_register.register(PTQApproach.RTN)
@dataclass
class RTNConfig:
    """
    Config for round to nearest algorithms.
    """


@dataclass
class PTQConfig:
    """
    Config for post trainning quantization.

    Args:
        mode (:class:`mindspore_gs.ptq.PTQMode`): Flag for ptq mode, ``QUANTIZATION`` for quantization mode,
            ``DEPLOY`` for deploy mode.
        backend (:class:`mindspore_gs.ptq.BackendTarget`): Flag for backend target, ``NONE`` for no specific backend,
            ``ASCEND`` for ascend backend.
        opname_blacklist (List[str]): Blacklist of opname. Layers in network with name fuzzy matched with this blacklist
            will not being quanted.

    Raises:
        ValueError: If `mode` is not in PTQMode's members.
        ValueError: If `backend` is not in BackendTarget's members.
        ValueError: if `opname_blacklist` is not a list of str.

    Example:
        >>> from mindspore_gs.ptq import PTQConfig, PTQMode
        >>> from mindspore_gs.common import BackendTarget
        >>> PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=['layer0'])
        PTQConfig(mode=<PTQMode.DEPLOY: 'deploy'>, backend=<BackendTarget.ASCEND: 'ascend'>, opname_blacklist=['layer0'], algo_args={})
    """
    mode: PTQMode = PTQMode.QUANTIZE
    backend: BackendTarget = BackendTarget.ASCEND
    opname_blacklist: List[str] = field(default_factory=list)
    algo_args: Union[dict, dataclass] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in PTQMode.__members__.values():
            raise ValueError(f'mode shall be in {PTQMode.__members__.values()}')
        if self.backend not in BackendTarget.__members__.values():
            raise ValueError(f'backend shall be in {BackendTarget.__members__.values()}')
        value_check('opname_blacklist', self.opname_blacklist, str)
        if is_dataclass(self.algo_args):
            self.algo_args = asdict(self.algo_args)


class YamlLoader:
    """Loader for some special item in yaml."""
    def __call__(self, src: str):
        raise NotImplementedError


class MSDTypeLoader(YamlLoader):
    """Loader for `mindspore.dtype` in yaml."""
    def __init__(self):
        self.dtype_dict = {
            "Bool": msdtype.bool_,
            "Int": msdtype.int_,
            "Int8": msdtype.int8,
            "Int16": msdtype.int16,
            "Int32": msdtype.int32,
            "Int64": msdtype.int64,
            "UInt8": msdtype.uint8,
            "UInt16": msdtype.uint16,
            "UInt32": msdtype.uint32,
            "UInt64": msdtype.uint64,
            "Float": msdtype.float_,
            "Float16": msdtype.float16,
            "Float32": msdtype.float32,
            "Float64": msdtype.float64,
            "BFloat16": msdtype.bfloat16,
            "Complex64": msdtype.complex64,
            "Complex128": msdtype.complex128,
        }

    def __call__(self, src: str):
        ms_dtype = self.dtype_dict.get(src, None)
        if not ms_dtype:
            raise ValueError(f"Unrecognized dtype: {src}")
        return ms_dtype


@dataclass
class InnerPTQConfig(GSBaseConfig, PTQConfig):
    """
    config for post-trainning-quantizer
    """
    approach: PTQApproach = field(default=PTQApproach.RTN)
    calibration_sampling_size: int = 0
    act_quant_dtype: QuantDtype = QuantDtype.INT8
    weight_quant_dtype: QuantDtype = QuantDtype.INT8
    weight_only: bool = True
    enable_kvcache_int8: bool = False
    act_per_channel: bool = False
    weight_per_channel: bool = True
    act_symmetric: bool = False
    weight_symmetric: bool = True
    act_narrow_range: bool = False
    weight_narrow_range: bool = False
    op_types: List[str] = field(default_factory=lambda: [QuantCellType.MF_LINEAR.value],
                                metadata={'choices': [
                                    item.value for item in QuantCellType.__members__.values()
                                ]})

    def __post_init__(self):
        value_check('calibration_sampling_size', self.calibration_sampling_size, int)
        value_check('act_quant_dtype', self.act_quant_dtype, QuantDtype)
        value_check('weight_quant_dtype', self.weight_quant_dtype, QuantDtype)
        value_check('weight_only', self.weight_only, bool)
        value_check('enable_kvcache_int8', self.enable_kvcache_int8, bool)
        value_check('act_per_channel', self.act_per_channel, bool)
        value_check('weight_per_channel', self.weight_per_channel, bool)
        value_check('act_symmetric', self.weight_symmetric, bool)
        value_check('act_narrow_range', self.act_narrow_range, bool)
        value_check('weight_narrow_range', self.weight_narrow_range, bool)
        if self.approach not in PTQApproach.__members__.values():
            raise ValueError(f'Invalid approach: {self.approach}')
        support_op_types = {
            item.value for item in QuantCellType.__members__.values()
        }
        for op_type in self.op_types:
            if op_type not in support_op_types:
                raise ValueError(f'{op_type} is not supported, all support type is {support_op_types}')
        if not self.algo_args:
            args_config = algo_cfg_register[self.approach]
            if args_config is not None and is_dataclass(args_config):
                self.algo_args.update(asdict(args_config()))

    def _parse_dict(self):
        """ parse data class to readable dicts"""
        parsed_dict = self.__dict__
        parsed_dict['act_quant_dtype'] = str(self.act_quant_dtype)
        parsed_dict['weight_quant_dtype'] = str(self.weight_quant_dtype)
        parsed_dict['backend'] = self.backend.name
        parsed_dict['mode'] = self.mode.name
        parsed_dict['approach'] = self.approach.name
        parsed_dict['opname_blacklist'] = self.opname_blacklist
        return parsed_dict

    def _unparse_dict(self, data_dict):
        """ convert readable dicts to data config"""
        def update_dict(key, decode_fn):
            nonlocal data_dict
            if key not in data_dict:
                raise ValueError(f'{key} shall in yaml, but not found')
            if isinstance(decode_fn, YamlLoader):
                data_dict[key] = decode_fn(data_dict[key])
            else:
                data_dict[key] = decode_fn[data_dict[key]]
        unparse_list = [
            ('act_quant_dtype', QuantDtype),
            ('weight_quant_dtype', QuantDtype),
            ('mode', PTQMode),
            ('backend', BackendTarget),
            ('approach', PTQApproach)
        ]
        for item in unparse_list:
            update_dict(*item)
        self.__dict__.update(data_dict)

    @staticmethod
    def inner_config(cfg: PTQConfig, approach=None):
        """convert PTQConfig to InnerConfig"""
        if not isinstance(cfg, PTQConfig):
            raise TypeError(f'input config shall be PTQConfig, but got {type(cfg)}')
        if not approach:
            inner_cfg = InnerPTQConfig()
        else:
            inner_cfg = InnerPTQConfig(approach=approach)
        for key, val in asdict(cfg).items():
            if key == "algo_args":
                inner_cfg.algo_args.update(val)
            else:
                setattr(inner_cfg, key, val)
        return inner_cfg
