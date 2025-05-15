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

from typing import List
from mindspore import dtype as msdtype
from mindspore.communication import get_group_size, get_rank

from mindspore_gs.common.config import GSBaseConfig
from mindspore_gs.common.dumper import Dumper
from mindspore_gs.common.utils import value_check
from mindspore_gs.common.register import RegisterMachine
from mindspore_gs.common.gs_enum import BackendTarget
from .ptq_config import PTQConfig, QuantGranularity, PrecisionRecovery, OutliersSuppressionType, PTQMode

algo_cfg_register = RegisterMachine()


class PTQApproach(Enum):
    """
    PTQ approach enums
    """
    SMOOTH_QUANT = 'smooth_quant'
    RTN = 'rtn'
    GPTQ = 'gptq'
    PTQ = 'ptq'
    AWQ = 'awq'


class LayerQuantizeAlgo(Enum):
    """
    Quantization algorithm for each layer.

    - ``A16W8``: apply.
    """
    A16W8 = 'a16w8'
    A8W8 = 'a8w8'
    A8DYNAMICW8 = 'a8dynamic-w8'


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
            "Int4": msdtype.qint4x2,
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
        if src == "None":
            return None
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
    act_per_channel: bool = False
    weight_per_channel: bool = True
    kvcache_per_head: bool = True
    act_symmetric: bool = False
    weight_symmetric: bool = True
    kvcache_symmetric: bool = True
    act_narrow_range: bool = False
    weight_narrow_range: bool = False
    kvcache_narrow_range: bool = False
    enable_deploy_fusion: bool = True
    kvcache_calibrate_max_new_tokens: int = 10
    reflash_inputs_after_each_processor: bool = False
    fallback_blacklist: dict = field(default_factory=dict)
    aclnn_quant_list: List[str] = field(default_factory=list)
    tp_size: int = 1
    rank_id: int = 0
    layer_quant_info_collect: dict = field(default_factory=dict)
    algorithm_cache_path: str = ''
    always_use_fp_input_in_processer: bool = False
    use_inner_osp: bool = False
    skip_offload_in_processing: bool = False

    dump_path: str = ""
    dumper: Dumper = Dumper()

    def report_quant_info(self, layer_name: str, quant_type: str):
        info = self.layer_quant_info_collect.get(layer_name)
        if info:
            info += f"-{quant_type}"
        else:
            info = quant_type
        self.layer_quant_info_collect[layer_name] = info

    def update_comm_info(self):
        try:
            self.tp_size = get_group_size()
        except RuntimeError:
            self.tp_size = 1
        try:
            self.rank_id = get_rank()
        except RuntimeError:
            self.rank_id = 0

    def set_dump_path(self, dump_path: str):
        self.dumper.set_dump_path(dump_path)

    def __post_init__(self):
        self.update_comm_info()
        value_check('act_per_channel', self.act_per_channel, bool)
        value_check('weight_per_channel', self.weight_per_channel, bool)
        value_check('kvcache_per_head', self.kvcache_per_head, bool)
        value_check('act_symmetric', self.act_symmetric, bool)
        value_check('weight_symmetric', self.weight_symmetric, bool)
        value_check('kvcache_symmetric', self.kvcache_symmetric, bool)
        value_check('act_narrow_range', self.act_narrow_range, bool)
        value_check('weight_narrow_range', self.weight_narrow_range, bool)
        value_check('enable_deploy_fusion', self.enable_deploy_fusion, bool)
        value_check('kvcache_calibrate_max_new_tokens', self.kvcache_calibrate_max_new_tokens, int)
        value_check('fallback_blacklist', self.fallback_blacklist, dict)
        value_check('reflash_inputs_after_each_processor', self.reflash_inputs_after_each_processor, bool)
        if self.approach not in PTQApproach.__members__.values():
            raise ValueError(f'Invalid approach: {self.approach}')
        self._check_quant_granularity()
        self._check_rtn()
        if list(set(self.fallback_blacklist.keys()) & set(self.opname_blacklist)):
            raise ValueError("There should be no repetition between opname_blacklist and fallback_a16w8_blacklist,"
                             f"now opname_blacklist={self.opname_blacklist},"
                             f"fallback_a16w8_blacklist={self.fallback_blacklist}")
        if not self.algo_args:
            args_config = algo_cfg_register[self.approach]
            if args_config is not None and is_dataclass(args_config):
                self.algo_args.update(asdict(args_config()))

    def _check_rtn(self):
        """_check_rtn"""
        if self.approach is PTQApproach.RTN and self.act_quant_dtype == msdtype.int8 and self.act_quant_granularity is not QuantGranularity.PER_TOKEN:
            raise ValueError(f"{self.approach} is not support act_quant_dtype == mindspore.dtype.int8 when act_quant_granularity is not QuantGranularity.PER_TOKEN.")
        if self.approach is PTQApproach.RTN and self.weight_quant_dtype is None and self.kvcache_quant_dtype is None:
            raise ValueError(f"weight_quant_dtype and kvcache_quant_dtype are None, {self.approach} can't take effect.")
        if self.approach is PTQApproach.RTN and self.weight_quant_dtype == msdtype.int8 and self.kvcache_quant_dtype == msdtype.int8 \
                        and self.kvcache_quant_granularity is not QuantGranularity.PER_TOKEN:
            raise ValueError("when self.kvcache_quant_granularity not QuantGranularity.PER_TOKEN, weight_quant_dtype and kvcache_quant_dtype are mindspore.dtype.int8, "
                             f"{self.approach} isn't supported.")
        if self.approach is PTQApproach.RTN and self.kvcache_quant_dtype == msdtype.int8:
            raise ValueError("The 'kvcache-int8' quant in RTN is deprecated. Please replace the value of the 'approach' parameter with 'PTQ'.")

    def _check_quant_granularity(self):
        """_check_quant_granularity"""
        if self.approach is PTQApproach.RTN and (self.act_quant_granularity is QuantGranularity.PER_TOKEN or \
                                                 self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN) and self.mode is PTQMode.QUANTIZE:
            raise ValueError("self.mode is PTQMode.QUANTIZE, self.act_quant_granularity is QuantGranularity.PER_TOKEN or "
                             f"self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, {self.approach} can't take effect.")
        if (self.act_quant_granularity is QuantGranularity.PER_TOKEN or self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN) and \
            self.approach is not PTQApproach.RTN and self.approach is not PTQApproach.PTQ:
            raise ValueError("self.act_quant_granularity is QuantGranularity.PER_TOKEN or "
                             f"self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, {self.approach} can't take effect.")
        if self.act_quant_granularity is QuantGranularity.PER_TOKEN and self.weight_symmetric is False:
            raise ValueError("when self.act_quant_granularity is QuantGranularity.PER_TOKEN, self.weight_symmetric must be True.")

    def _parse_dict(self):
        """ parse data class to readable dicts"""
        parsed_dict = self.__dict__
        parsed_dict['backend'] = self.backend.name
        parsed_dict['mode'] = self.mode.name
        parsed_dict['approach'] = self.approach.name
        parsed_dict['opname_blacklist'] = self.opname_blacklist
        parsed_dict['kvcache_quant_dtype'] = str(self.kvcache_quant_dtype)
        parsed_dict['weight_quant_dtype'] = str(self.weight_quant_dtype)
        parsed_dict['act_quant_dtype'] = str(self.act_quant_dtype)
        parsed_dict['precision_recovery'] = self.precision_recovery.name
        parsed_dict['outliers_suppression'] = self.outliers_suppression.name
        parsed_dict['act_quant_granularity'] = self.act_quant_granularity.name
        parsed_dict['kvcache_quant_granularity'] = self.kvcache_quant_granularity.name
        parsed_dict['weight_quant_granularity'] = self.weight_quant_granularity.name
        parsed_dict.pop('dumper')
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
            ('mode', PTQMode),
            ('backend', BackendTarget),
            ('approach', PTQApproach),
            ('outliers_suppression', OutliersSuppressionType),
            ('precision_recovery', PrecisionRecovery),
            ('kvcache_quant_dtype', MSDTypeLoader()),
            ('weight_quant_dtype', MSDTypeLoader()),
            ('act_quant_dtype', MSDTypeLoader()),
            ('act_quant_granularity', QuantGranularity),
            ('kvcache_quant_granularity', QuantGranularity),
            ('weight_quant_granularity', QuantGranularity),
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
        inner_cfg.__post_init__()
        return inner_cfg
