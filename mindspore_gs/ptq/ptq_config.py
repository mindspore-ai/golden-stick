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

from mindspore import dtype as msdtype

from mindspore_gs.common.config import GSBaseConfig
from mindspore_gs.common.utils import value_check, list_value_check
from mindspore_gs.common.register import RegisterMachine
from mindspore_gs.common.gs_enum import BackendTarget

algo_cfg_register = RegisterMachine()


class PTQApproach(Enum):
    """
    PTQ approach enums
    """
    SMOOTH_QUANT = 'smooth_quant'
    RTN = 'rtn'
    GPTQ = 'gptq'
    OMNI_QUANT = 'omni_quant'
    PTQ = 'ptq'


class PTQMode(Enum):
    """
    Mode for ptq quantizer.

    - ``QUANTIZE``: indicate ptq quantizer in quantize mode.
    - ``DEPLOY``: indicate ptq quantizer in deploy mode.
    """
    QUANTIZE = 'quantize'
    DEPLOY = 'deploy'


class OutliersSuppressionType(Enum):
    """
    Outliers suppression type for ptq quantizer.

    - ``SMOOTH``: apply smooth scale between weight and activate.
    - ``NONE``: not doing any outliers suppression.
    """
    SMOOTH = 'smooth'
    NONE = 'none'


class LayerQuantizeAlgo(Enum):
    """
    Quantization algorithm for each layer.

    - ``A16W8``: apply.
    """
    A16W8 = 'a16w8'
    A8W8 = 'a8w8'


@algo_cfg_register.register(PTQApproach.OMNI_QUANT)
@dataclass
class OmniQuantConfig:
    """config for omni quant algorithm"""
    pre_clip_ratio: Union[list, float] = 1.0
    post_clip_ratio: Union[list, float] = 1.0
    smooth_alpha: Union[list, float] = 0.5
    is_revert_by_loss: bool = False

    def __post_init__(self):
        value_check('pre_clip_ratio', self.pre_clip_ratio, Union[list, float])
        value_check('post_clip_ratio', self.post_clip_ratio, Union[list, float])
        value_check('smooth_alpha', self.smooth_alpha, Union[list, float])
        value_check('is_revert_by_loss', self.is_revert_by_loss, bool)
        if (not isinstance(self.pre_clip_ratio, type(self.post_clip_ratio))) or \
            (not isinstance(self.pre_clip_ratio, type(self.smooth_alpha))) or \
            (not isinstance(self.post_clip_ratio, type(self.smooth_alpha))):
            raise ValueError(f"pre_clip_ratio, post_clip_ratio and smooth_alpha should have same type,"
                             f"but got pre_clip_ratio: {type(self.pre_clip_ratio)},"
                             f"post_clip_ratio: {type(self.post_clip_ratio)},"
                             f"smooth_alpha: {type(self.smooth_alpha)}.")


@algo_cfg_register.register(PTQApproach.PTQ)
@dataclass
class PTQQuantConfig:
    """config for omni quant algorithm"""


@algo_cfg_register.register(PTQApproach.SMOOTH_QUANT)
@dataclass
class SmoothQuantConfig:
    """config for smooth quant algorithm"""
    alpha: float = 0.5

    def __post_init__(self):
        value_check('alpha', self.alpha, float)


class QuantGranularity(Enum):
    """quant granularity."""
    PER_TENSOR = 'per_tensor'
    PER_CHANNEL = 'per_channel'
    PER_TOKEN = 'per_token'
    PER_GROUP = 'per_group'


@dataclass
class PTQConfig:
    """
    Config for post trainning quantization.

    Args:
        mode (:class:`mindspore_gs.ptq.PTQMode`): Flag for ptq mode, ``QUANTIZATION`` for quantization mode,
            ``DEPLOY`` for deploy mode.
        backend (:class:`mindspore_gs.common.BackendTarget`): Flag for backend target, ``NONE`` for no specific backend,
            ``ASCEND`` for ascend backend.
        opname_blacklist (List[str]): Blacklist of opname. Layers in network with name fuzzy matched with this blacklist
            will not being quanted.
        algo_args (Union[dict, dataclass]): Used to configure hyperparameters of algorithms such as RTN, SmoothQuant,
            and OmniQuant.
        act_quant_dtype (mindspore.dtype): Used to configure the quantization type of activation. mindspore.dtype.int8
            indicates that the activation is quantized by 8 bits, and None indicates that it is not quantized.
        weight_quant_dtype (mindspore.dtype): Used to configure the quantization type of weight. mindspore.dtype.int8
            indicates that the weight is quantized by 8 bits, and None indicates that it is not quantized.
        kvcache_quant_dtype (mindspore.dtype): Used to configure the quantization type of kvcache. mindspore.dtype.int8
            indicates that the kvcache is quantized by 8 bits, and None indicates that it is not quantized.
        outliers_suppression (:class:`mindspore_gs.ptq.OutliersSuppressionType`): Used to configure outliers suppression
            method before quantization. OutliersSuppressionType.SMOOTH indicates using smooth method from SmoothQuant
            to suppress outliers, and OutliersSuppressionType.NONE as default indicates doing nothing for outliers.
        act_quant_granularity: (:class:`mindspore_gs.ptq.QuantGranularity`): Used to configure the quantization granularity of activation.
            Currently only QuantGranularity.PER_TENSOR and QuantGranularity.PER_TOKEN are supported.
        kvcache_quant_granularity: (:class:`mindspore_gs.ptq.QuantGranularity`): Used to configure the quantization granularity of kvcache.
            Currently only QuantGranularity.PER_CHANNEL and QuantGranularity.PER_TOKEN are supported.
        weight_quant_granularity: (:class:`mindspore_gs.ptq.QuantGranularity`): Used to configure the quantization granularity of weight.
            Currently only QuantGranularity.PER_CHANNEL and QuantGranularity.PER_GROUP are supported.
        group_size (int): group_size of per_group quantization, suggest using 64 or 128.

    Raises:
        ValueError: If `mode` is not PTQMode.QUANTIZE or PTQMode.DEPLOY.
        ValueError: If `backend` is not BackendTarget.NONE or BackendTarget.ASCEND.
        TypeError: If `opname_blacklist` is not a list of str.
        ValueError: If `weight_quant_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `kvcache_quant_dtype` is not mindspore.dtype.int8 or None.
        ValueError: If `act_quant_dtype` is not mindspore.dtype.int8 or None.
        TypeError: If `outliers_suppression` is not a OutliersSuppressionType.
        ValueError: If `act_quant_granularity` is not QuantGranularity.PER_TENSOR or QuantGranularity.PER_TOKEN.
        ValueError: If `kvcache_quant_granularity` is not QuantGranularity.PER_CHANNEL or QuantGranularity.PER_TOKEN.
        ValueError: If `act_quant_granularity` is QuantGranularity.PER_TOKEN but weight_quant_dtype != msdtype.int8 or act_quant_dtype != msdtype.int8.
        ValueError: If `kvcache_quant_granularity` is QuantGranularity.PER_TOKEN but kvcache_quant_dtype != msdtype.int8.
        ValueError: If `weight_quant_granularity` is not QuantGranularity.PER_CHANNEL or QuantGranularity.PER_GROUP.
        ValueError: If `weight_quant_granularity` is QuantGranularity.PER_GROUP but `group_size` is not in [64, 128].
        ValueError: If `weight_quant_granularity` is not QuantGranularity.PER_GROUP but `group_size` != 0.
        TypeError: If `group_size` is not Int.

    Examples:
        >>> from mindspore_gs.ptq import PTQConfig, PTQMode
        >>> from mindspore_gs.common import BackendTarget
        >>> PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, opname_blacklist=['layer0'])
        PTQConfig(mode=<PTQMode.DEPLOY: 'deploy'>, backend=<BackendTarget.ASCEND: 'ascend'>, opname_blacklist=['layer0'], algo_args={})
    """
    mode: PTQMode = PTQMode.QUANTIZE
    backend: BackendTarget = BackendTarget.ASCEND
    opname_blacklist: List[str] = field(default_factory=list)
    algo_args: Union[dict, dataclass] = field(default_factory=dict)
    weight_quant_dtype: msdtype = msdtype.int8
    kvcache_quant_dtype: msdtype = None
    act_quant_dtype: msdtype = None
    outliers_suppression: OutliersSuppressionType = OutliersSuppressionType.NONE
    weight_quant_granularity: QuantGranularity = QuantGranularity.PER_CHANNEL
    kvcache_quant_granularity: QuantGranularity = QuantGranularity.PER_CHANNEL
    act_quant_granularity: QuantGranularity = QuantGranularity.PER_TENSOR
    group_size: int = 0

    def __post_init__(self):
        weight_support = [msdtype.int8, msdtype.qint4x2, None]
        act_support = [msdtype.int8, None]
        kvcache_support = [msdtype.int8, None]
        if self.mode not in PTQMode.__members__.values():
            raise ValueError(f'mode shall be in {PTQMode.__members__.values()}')
        if self.backend not in BackendTarget.__members__.values():
            raise ValueError(f'backend shall be in {BackendTarget.__members__.values()}')
        if self.weight_quant_dtype not in weight_support:
            raise ValueError(f'weight_quant_dtype support {weight_support}, but got {self.weight_quant_dtype}.')
        if self.kvcache_quant_dtype not in kvcache_support:
            raise ValueError(f'kvcache_quant_dtype support {kvcache_support}, but got {self.kvcache_quant_dtype}.')
        if self.act_quant_dtype not in act_support:
            raise ValueError(f'act_quant_dtype support {act_support}, but got {self.act_quant_dtype}.')
        list_value_check('opname_blacklist', self.opname_blacklist, str)
        value_check('outliers_suppression', self.outliers_suppression, OutliersSuppressionType)
        self._check_quant_granularity()
        value_check('group_size', self.group_size, int)
        if not isinstance(self.algo_args, dict) and not is_dataclass(self.algo_args):
            raise ValueError(f"algo_args's type should be dict or dataclass, but now is {type(self.algo_args)}")
        if self.algo_args and is_dataclass(self.algo_args):
            self.algo_args = asdict(self.algo_args)

    def _check_quant_granularity(self):
        '''check_quant_granularity'''
        if self.act_quant_granularity != QuantGranularity.PER_TENSOR and self.act_quant_granularity != \
            QuantGranularity.PER_TOKEN:
            raise ValueError(f'self.act_quant_granularity {self.act_quant_granularity} must be \
                             QuantGranularity.PER_CHANNEL or QuantGranularity.PER_TOKEN.')
        if self.weight_quant_granularity != QuantGranularity.PER_CHANNEL and self.weight_quant_granularity != \
            QuantGranularity.PER_GROUP:
            raise ValueError(f'self.weight_quant_granularity {self.weight_quant_granularity} must be \
                             QuantGranularity.PER_CHANNEL or QuantGranularity.PER_GROUP.')
        if self.kvcache_quant_granularity != QuantGranularity.PER_CHANNEL and self.kvcache_quant_granularity != \
            QuantGranularity.PER_TOKEN:
            raise ValueError(f'self.kvcache_quant_granularity {self.kvcache_quant_granularity} must be \
                             QuantGranularity.PER_CHANNEL or QuantGranularity.PER_TOKEN.')
        if (self.weight_quant_dtype != msdtype.int8 or self.act_quant_dtype != msdtype.int8) and \
            self.act_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError('when self.act_quant_granularity is QuantGranularity.PER_TOKEN, self.weight_quant_dtype: {self.weight_quant_dtype} \
                             and self.act_quant_dtype: {self.act_quant_dtype} must be mindspore.dtype.int8.')
        if self.kvcache_quant_dtype != msdtype.int8 and self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError('when self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, self.kvcache_quant_dtype must be mindspore.dtype.int8.')
        if self.weight_quant_granularity != QuantGranularity.PER_GROUP and self.group_size != 0:
            raise ValueError("group_size should equal to 0 when not to do pre_group quantize.")
        if self.weight_quant_granularity == QuantGranularity.PER_GROUP and self.group_size not in [64, 128]:
            raise ValueError("group_size should be in [64, 128] when doing pre_group quantize.")

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
    fallback_blacklist: dict = field(default_factory=dict)

    def __post_init__(self):
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
        if self.approach is PTQApproach.RTN and self.act_quant_dtype == msdtype.int8 and self.act_quant_granularity is not QuantGranularity.PER_TOKEN:
            raise ValueError(f"{self.approach} is not support act_quant_dtype == mindspore.dtype.int8 when act_quant_granularity is not QuantGranularity.PER_TOKENNNNN.")
        if self.approach is PTQApproach.RTN and self.weight_quant_dtype is None and self.kvcache_quant_dtype is None:
            raise ValueError(f"weight_quant_dtype and kvcache_quant_dtype are None, {self.approach} can't take effect.")
        if self.approach is PTQApproach.RTN and self.weight_quant_dtype == msdtype.int8 and self.kvcache_quant_dtype == msdtype.int8 \
                        and self.kvcache_quant_granularity is not QuantGranularity.PER_TOKEN:
            raise ValueError(f"when self.kvcache_quant_granularity not QuantGranularity.PER_TOKEN, weight_quant_dtype and kvcache_quant_dtype are mindspore.dtype.int8, \
                             {self.approach} isn't supported.")

    def _check_quant_granularity(self):
        if self.approach is PTQApproach.RTN and (self.act_quant_granularity is QuantGranularity.PER_TOKEN or \
                                                 self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN) and self.mode is PTQMode.QUANTIZE:
            raise ValueError(f"self.mode is PTQMode.QUANTIZE, self.act_quant_granularity is QuantGranularity.PER_TOKEN or \
                             self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, {self.approach} can't take effect.")
        if (self.act_quant_granularity is QuantGranularity.PER_TOKEN or self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN) and \
            self.approach is not PTQApproach.RTN and self.approach is not PTQApproach.PTQ:
            raise ValueError(f"self.act_quant_granularity is QuantGranularity.PER_TOKEN or \
                              self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, {self.approach} can't take effect.")

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
        parsed_dict['outliers_suppression'] = self.outliers_suppression.name
        parsed_dict['act_quant_granularity'] = self.act_quant_granularity.name
        parsed_dict['kvcache_quant_granularity'] = self.kvcache_quant_granularity.name
        parsed_dict['weight_quant_granularity'] = self.weight_quant_granularity.name
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
        return inner_cfg
