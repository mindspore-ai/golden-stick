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

from mindspore_gs.common import logger
from mindspore_gs.common.utils import value_check, list_value_check
from mindspore_gs.common.gs_enum import BackendTarget


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
    - ``AWQ``: apply awq algorithm to outliers suppression.
    - ``OUTLIER_SUPPRESSION_PLUS``: apply OUTLIER_SUPPRESSION_PLUS algorithm to outliers suppression.
    - ``NONE``: not doing any outliers suppression.
    """
    SMOOTH = 'smooth'
    AWQ = 'awq'
    OUTLIER_SUPPRESSION_PLUS = 'outlier-suppression+'
    OMNIQUANT_GRID = 'omniquant-grid'
    NONE = 'none'

    @classmethod
    def from_str(cls, name: str):
        """
        Convert name to outliers suppression algorithm type.

        Args:
            name (str): the string name of the outliers suppression algorithm.
        """
        if name.lower() == 'smooth':
            return OutliersSuppressionType.SMOOTH
        if name.lower() == 'awq':
            return OutliersSuppressionType.AWQ
        if name.lower() == 'outlier-suppression+':
            return OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS
        if name.lower() == 'omniquant-grid':
            return OutliersSuppressionType.OMNIQUANT_GRID
        return OutliersSuppressionType.NONE


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
        """
        Convert name to precision recovery algorithm type.

        Args:
            name (str): the string name of the precision recovery algorithm.
        """
        if name.lower() == 'gptq':
            return PrecisionRecovery.GPTQ
        return PrecisionRecovery.NONE


@dataclass
class GPTQQuantConfig:
    """
    Config for gptq quant algorithm.

    Args:
        block_size (int, optional): The size of block compensation in precision recovery. Default value: ``128``.
        desc_act (bool, optional): Whether to perform importance sorting on the Hessian matrix. Default value: ``False``.
        damp_percent (float, optional): The percentage of the average of the diagonal elements of the Hessian matrix during numerical stable computations. Default value: ``0.01``.
        static_groups (bool, optional): Whether to perform per_group calculation before precision recovery in the GPTQ algorithm. Default value: ``False``.

    Raises:
        TypeError: If `block_size` is not type int.
        TypeError: If `desc_act` is not type bool.
        TypeError: If `damp_percent` is not type float.
        TypeError: If `static_groups` is not type bool.
        ValueError: If `block_size` is less than 0.
        ValueError: If `damp_percent` is less than 0 or greater than 1.

    Examples:
        >>> from mindspore_gs.ptq import GPTQQuantConfig
        >>> GPTQQuantConfig(block_size=128, desc_act=False, damp_percent=0.01, static_groups=False)
    """
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


@dataclass
class SmoothQuantConfig:
    """config for smooth quant algorithm"""
    alpha: float = 0.5

    def __post_init__(self):
        value_check('alpha', self.alpha, float)


@dataclass
class AWQConfig:
    """
    AWQConfig(duo_scaling=True, smooth_alpha=[i/20 for i in range(20)], weight_clip_ratio=[1-i/20 for i in range(10)])

    Config for awq quant algorithm.

    Args:
        duo_scaling (bool, optional): Use activation and weight to compute scale. Default value: ``True``.
        smooth_alpha (List[float], optional): The hyper-parameter of smooth search. Default value: ``[i/20 for i in range(20)]``.
        weight_clip_ratio (List[float], optional): The hyper-parameter of clip search. Default value: ``[1-i/20 for i in range(10)]``.

    Raises:
        TypeError: If `duo_scaling` is not type bool.
        TypeError: If `smooth_alpha` is not type float or list.
        TypeError: If `weight_clip_ratio` is not float or list.
        ValueError: If `smooth_alpha` is less than 0 or greater than 1.
        ValueError: If `weight_clip_ratio` is less than 0 or greater than 1.

    Examples:
        >>> from mindspore_gs.ptq import AWQConfig
        >>> AWQConfig(duo_scaling=True, smooth_alpha=[i/20 for i in range(20)], weight_clip_ratio=[1-i/20 for i in range(10)])
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
    """
    Quant granularity for ptq quantizer.

    - ``PER_TENSOR``: apply quant granularity to per_tensor.
    - ``PER_CHANNEL``: apply quant granularity to per_channel.
    - ``PER_TOKEN``: apply quant granularity to per_token.
    - ``PER_GROUP``: apply quant granularity to per_group.
    """
    PER_TENSOR = 'per_tensor'
    PER_CHANNEL = 'per_channel'
    PER_TOKEN = 'per_token'
    PER_GROUP = 'per_group'

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, name: str):
        """
        Convert name to quant granularity type.

        Args:
            name (str): the string name of the quant granularity.
        """
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
    PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND, opname_blacklist=<class 'list'>, algo_args=<class 'dict'>, weight_quant_dtype=Int8, kvcache_quant_dtype=None, act_quant_dtype=None, outliers_suppression=OutliersSuppressionType.NONE, precision_recovery=PrecisionRecovery.NONE, weight_quant_granularity=QuantGranularity.PER_CHANNEL, group_size=0, act_quant_granularity=QuantGranularity.PER_TENSOR, kvcache_quant_granularity=QuantGranularity.PER_CHANNEL)

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
        act_quant_granularity (:class:`mindspore_gs.ptq.QuantGranularity`): Used to configure the quantization granularity of activation.
            Currently only QuantGranularity.PER_TENSOR and QuantGranularity.PER_TOKEN are supported.
        kvcache_quant_granularity (:class:`mindspore_gs.ptq.QuantGranularity`): Used to configure the quantization granularity of kvcache.
            Currently only QuantGranularity.PER_CHANNEL and QuantGranularity.PER_TOKEN are supported.
        weight_quant_granularity (:class:`mindspore_gs.ptq.QuantGranularity`): Used to configure the quantization granularity of weight.
            Currently only QuantGranularity.PER_CHANNEL and QuantGranularity.PER_GROUP are supported.
        group_size (int, optional): group_size of per_group quantization, suggest using 64 or 128. Default value: ``0``.

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
        value_check('group_size', self.group_size, int)
        self._check_quant_granularity()
        self._check_precision_recovery()
        if not isinstance(self.algo_args, dict) and not is_dataclass(self.algo_args):
            raise ValueError(f"algo_args's type should be dict or dataclass, but now is {type(self.algo_args)}")
        if self.algo_args and is_dataclass(self.algo_args):
            self.algo_args = asdict(self.algo_args)

    def _check_precision_recovery(self):
        '''check precision recovery'''
        if self.precision_recovery == PrecisionRecovery.GPTQ and self.algo_args == {}:
            self.algo_args = GPTQQuantConfig()
            logger.warning('GPTQQuantConfig is not configured, it will apply the default parameters.')
        if self.precision_recovery != PrecisionRecovery.GPTQ and isinstance(self.algo_args, GPTQQuantConfig):
            logger.warning(f'GPTQQuantConfig is configured, but the precision recovery is {self.precision_recovery}.')
        if self.precision_recovery == PrecisionRecovery.GPTQ and \
                self.weight_quant_granularity == QuantGranularity.PER_GROUP and \
                (self.algo_args.desc_act and not self.algo_args.static_groups):
            raise ValueError('when using GPTQ algorithm with per_group quantization, '
                             'if desc_act is True ,then static_groups must be true.')
        value_check('precision_recovery', self.precision_recovery, PrecisionRecovery)

    def _check_quant_granularity(self):
        """check_quant_granularity"""
        act_quant_granularity_support = [QuantGranularity.PER_TENSOR, QuantGranularity.PER_TOKEN]
        weight_quant_granularity_support = [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_GROUP]
        kvcache_quant_granularity_support = [QuantGranularity.PER_CHANNEL, QuantGranularity.PER_TOKEN]
        if self.act_quant_granularity not in act_quant_granularity_support:
            raise ValueError(f'act_quant_granularity support {act_quant_granularity_support}, '
                             f'but got {self.act_quant_granularity}.')
        if self.weight_quant_granularity not in weight_quant_granularity_support:
            raise ValueError(f'weight_quant_granularity support {weight_quant_granularity_support}, '
                             f'but got {self.weight_quant_granularity}.')
        if self.kvcache_quant_granularity not in kvcache_quant_granularity_support:
            raise ValueError(f'kvcache_quant_granularity support {kvcache_quant_granularity_support}, '
                             f'but got {self.kvcache_quant_granularity}.')
        if (self.weight_quant_dtype != msdtype.int8 or self.act_quant_dtype != msdtype.int8) and \
            self.act_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError(f'when act_quant_granularity is QuantGranularity.PER_TOKEN, weight_quant_dtype: {self.weight_quant_dtype} '
                             f'and act_quant_dtype: {self.act_quant_dtype} must be mindspore.dtype.int8.')
        if (self.weight_quant_dtype == msdtype.int8 and self.act_quant_dtype == msdtype.int8) and \
            self.weight_quant_granularity != QuantGranularity.PER_CHANNEL:
            raise ValueError(f'when weight_quant_dtype is int8 and act_quant_dtype is int8, the weight_quant_granularity must be QuantGranularity.PER_CHANNEL,'
                             f' but got {self.weight_quant_granularity}.')
        if self.kvcache_quant_dtype != msdtype.int8 and self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN:
            raise ValueError('when kvcache_quant_granularity is QuantGranularity.PER_TOKEN, kvcache_quant_dtype must be mindspore.dtype.int8.')
        if self.mode == PTQMode.QUANTIZE and self.kvcache_quant_granularity is QuantGranularity.PER_TOKEN:
            logger.warning('kvcache_quant_granularity is QuantGranularity.PER_TOKEN, not need quantize for kvcache.')
        if self.weight_quant_granularity != QuantGranularity.PER_GROUP and self.group_size != 0:
            raise ValueError("group_size should equal to 0 when not to do pre_group quantize.")
        if self.weight_quant_granularity == QuantGranularity.PER_GROUP and self.group_size not in [64, 128]:
            raise ValueError("group_size should be in [64, 128] when doing pre_group quantize.")
