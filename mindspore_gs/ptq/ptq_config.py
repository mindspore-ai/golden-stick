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
from mindspore.communication import get_group_size

from mindspore_gs.common import logger
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
    PTQ = 'ptq'
    AWQ = 'awq'


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
    - ``AWQ``: apply awq algo to search best smooth_scale and clip parameter.
    - ``NONE``: not doing any outliers suppression.
    """
    SMOOTH = 'smooth'
    AWQ = 'awq'
    NONE = 'none'

    @classmethod
    def from_str(cls, name: str):
        """outlier_formatter"""
        if name.lower() == 'smooth':
            return OutliersSuppressionType.SMOOTH
        if name.lower() == 'awq':
            return OutliersSuppressionType.AWQ
        return OutliersSuppressionType.NONE


class LayerQuantizeAlgo(Enum):
    """
    Quantization algorithm for each layer.

    - ``A16W8``: apply.
    """
    A16W8 = 'a16w8'
    A8W8 = 'a8w8'


class PrecisionRecovery(Enum):
    """
    Precision recovery algorithms.

    - ``GPTQ``: apply gptq algorithm to recovery precision.
    - ``NONE``: not doing any precision recovery.
    """
    GPTQ = 'gptq'
    NONE = 'none'

    @classmethod
    def from_str(cls, name: str):
        """outlier_formatter"""
        if name.lower() == 'gptq':
            return PrecisionRecovery.GPTQ
        return PrecisionRecovery.NONE


@algo_cfg_register.register(PTQApproach.GPTQ)
@dataclass
class GPTQQuantConfig:
    """config for gptq quant algorithm"""
    block_size: int = 128
    desc_act: bool = False
    damp_percent: float = 0.01
    static_groups: bool = False

    def __post_init__(self):
        value_check('block_size', self.block_size, int)
        value_check('desc_act', self.desc_act, bool)
        value_check('damp_percent', self.damp_percent, float)
        value_check('static_groups', self.static_groups, bool)
        if self.block_size < 0:
            raise ValueError("GPTQConfig block_size should >=0, " f"but got {self.block_size}.")
        if self.damp_percent < 0 or self.damp_percent > 1:
            raise ValueError("GPTQConfig damp_percent should >=0 and <=1, " f"but got {self.damp_percent}.")


@algo_cfg_register.register(PTQApproach.SMOOTH_QUANT)
@dataclass
class SmoothQuantConfig:
    """config for smooth quant algorithm"""
    alpha: float = 0.5

    def __post_init__(self):
        value_check('alpha', self.alpha, float)


@algo_cfg_register.register(PTQApproach.AWQ)
@dataclass
class AWQConfig:
    """
    config for awq quant algorithm

    - `duo_scaling`: use activation and weight to compute scale.
    - `smooth_alpha`: the hyper-parameter of smooth search.
    - `weight_clip_ratio`: the hyper-parameter of clip search.
    """
    duo_scaling: bool = True
    smooth_alpha: Union[list, float] = field(default_factory=lambda: [i/20 for i in range(20)])
    weight_clip_ratio: Union[list, float] = field(default_factory=lambda: [1- i/20 for i in range(10)])

    def __post_init__(self):
        value_check('duo_scaling', self.duo_scaling, bool)
        if not isinstance(self.smooth_alpha, (float, list)):
            raise TypeError("AWQConfig smooth_alpha only support float or list, "
                            f"but got {type(self.smooth_alpha)}")
        if not isinstance(self.weight_clip_ratio, (float, list)):
            raise TypeError("AWQConfig weight_clip_ratio only support float or list, "
                            f"but got {type(self.weight_clip_ratio)}")
        if isinstance(self.smooth_alpha, float) and (self.smooth_alpha < 0 or self.smooth_alpha > 1):
            raise ValueError("AWQConfig smooth_alpha should >=0 and <=1, "
                             f"but got {self.smooth_alpha}.")
        if isinstance(self.weight_clip_ratio, float) and (self.weight_clip_ratio < 0 or self.weight_clip_ratio > 1):
            raise ValueError("AWQConfig weight_clip_ratio should >=0 and <=1, "
                             f"but got {self.weight_clip_ratio}.")
        if isinstance(self.smooth_alpha, list) and any(alpha < 0 or alpha > 1 for alpha in self.smooth_alpha):
            raise ValueError("AWQConfig smooth_alpha should >=0 and <=1, "
                             f"but got {self.smooth_alpha}.")
        if isinstance(self.weight_clip_ratio, list) and any(alpha < 0 or alpha > 1 for alpha in self.weight_clip_ratio):
            raise ValueError("AWQConfig weight_clip_ratio should >=0 and <=1, "
                             f"but got {self.weight_clip_ratio}.")


class QuantGranularity(Enum):
    """quant granularity."""
    PER_TENSOR = 'per_tensor'
    PER_CHANNEL = 'per_channel'
    PER_TOKEN = 'per_token'
    PER_GROUP = 'per_group'

    @classmethod
    def from_str(cls, name: str):
        """quant_granularity_formatter"""
        if name.lower() == 'per_token':
            return QuantGranularity.PER_TOKEN
        if name.lower() == 'per_tensor':
            return QuantGranularity.PER_TENSOR
        if name.lower() == 'per_channel':
            return QuantGranularity.PER_CHANNEL
        if name.lower() == 'per_group':
            return QuantGranularity.PER_GROUP
        return None


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
        precision_recovery (:class:`mindspore_gs.ptq.PrecisionRecovery`): Used to precision compensation of
            weights during quantization. PrecisionRecovery.GPTQ indicates using GPTQ method to compensate precision,
            and PrecisionRecovery.NONE as default indicates doing nothing for precision recovery.
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
        TypeError: If `precision_recovery` is not a PrecisionRecovery.
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
    precision_recovery: PrecisionRecovery = PrecisionRecovery.NONE
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
        self._check_precision_recovery()
        self._check_quant_granularity()
        value_check('group_size', self.group_size, int)
        if not isinstance(self.algo_args, dict) and not is_dataclass(self.algo_args):
            raise ValueError(f"algo_args's type should be dict or dataclass, but now is {type(self.algo_args)}")
        if self.algo_args and is_dataclass(self.algo_args):
            self.algo_args = asdict(self.algo_args)

    def _check_precision_recovery(self):
        '''check precision recovery'''
        if self.precision_recovery == PrecisionRecovery.GPTQ and self.algo_args == {}:
            self.algo_args = GPTQQuantConfig()
            logger.warning('GPTQConfig is not configured, it will apply the default parameters.')
        if self.precision_recovery != PrecisionRecovery.GPTQ and isinstance(self.algo_args, GPTQQuantConfig):
            logger.warning(f'GPTQConfig is configured, but the precision recovery is {self.precision_recovery}.')
        value_check('precision_recovery', self.precision_recovery, PrecisionRecovery)

    def _check_quant_granularity(self):
        """check_quant_granularity"""
        if self.act_quant_granularity != QuantGranularity.PER_TENSOR and self.act_quant_granularity != \
            QuantGranularity.PER_TOKEN:
            raise ValueError(f'self.act_quant_granularity {self.act_quant_granularity} must be '
                             'QuantGranularity.PER_CHANNEL or QuantGranularity.PER_TOKEN.')
        if self.weight_quant_granularity != QuantGranularity.PER_CHANNEL and self.weight_quant_granularity != \
            QuantGranularity.PER_GROUP:
            raise ValueError(f'self.weight_quant_granularity {self.weight_quant_granularity} must be '
                             'QuantGranularity.PER_CHANNEL or QuantGranularity.PER_GROUP.')
        if self.kvcache_quant_granularity != QuantGranularity.PER_CHANNEL and self.kvcache_quant_granularity != \
            QuantGranularity.PER_TOKEN:
            raise ValueError(f'self.kvcache_quant_granularity {self.kvcache_quant_granularity} must be '
                             'QuantGranularity.PER_CHANNEL or QuantGranularity.PER_TOKEN.')
        if (self.weight_quant_dtype != msdtype.int8 or self.act_quant_dtype != msdtype.int8) and \
            self.act_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError(f'when self.act_quant_granularity is QuantGranularity.PER_TOKEN, self.weight_quant_dtype: {self.weight_quant_dtype} '
                             f'and self.act_quant_dtype: {self.act_quant_dtype} must be mindspore.dtype.int8.')
        if self.kvcache_quant_dtype != msdtype.int8 and self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError('when self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, self.kvcache_quant_dtype must be mindspore.dtype.int8.')
        if self.mode == PTQMode.QUANTIZE and self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN:
            logger.warning('self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN, not need quantize for kvcache.')
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
    reflash_inputs_after_each_processor: bool = False
    fallback_blacklist: dict = field(default_factory=dict)
    tp_size: int = 1

    def update_tp_size(self):
        try:
            self.tp_size = get_group_size()
        except RuntimeError:
            pass

    def __post_init__(self):
        self.update_tp_size()
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
        if self.approach is PTQApproach.RTN and self.act_quant_dtype == msdtype.int8 and self.act_quant_granularity is not QuantGranularity.PER_TOKEN:
            raise ValueError(f"{self.approach} is not support act_quant_dtype == mindspore.dtype.int8 when act_quant_granularity is not QuantGranularity.PER_TOKENNNNN.")
        if self.approach is PTQApproach.RTN and self.weight_quant_dtype is None and self.kvcache_quant_dtype is None:
            raise ValueError(f"weight_quant_dtype and kvcache_quant_dtype are None, {self.approach} can't take effect.")
        if self.approach is PTQApproach.RTN and self.weight_quant_dtype == msdtype.int8 and self.kvcache_quant_dtype == msdtype.int8 \
                        and self.kvcache_quant_granularity is not QuantGranularity.PER_TOKEN:
            raise ValueError("when self.kvcache_quant_granularity not QuantGranularity.PER_TOKEN, weight_quant_dtype and kvcache_quant_dtype are mindspore.dtype.int8, "
                             f"{self.approach} isn't supported.")

    def _check_quant_granularity(self):
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
        return inner_cfg
