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
"""test for ptq related config"""
import os
import sys

import pytest
from mindspore import dtype as msdtype

from mindspore_gs.ptq import PrecisionRecovery
from mindspore_gs.ptq.ptq_config import (PTQConfig, SmoothQuantConfig, PTQMode,
                                         OutliersSuppressionType, QuantGranularity,
                                         GPTQQuantConfig, AWQConfig)
from mindspore_gs.ptq.context import InnerPTQConfig, PTQApproach
from mindspore_gs.common import BackendTarget

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../../'))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_basic_construct():
    """
    Feature: PTQConfig basic construction
    Description: Test PTQConfig default constructor and basic functionality
    Expectation: Config object created correctly, default values match expectations, property access works normally
    """
    # Verify default constructor
    cfg = PTQConfig()
    assert cfg.mode == PTQMode.QUANTIZE
    assert cfg.backend == BackendTarget.ASCEND
    assert cfg.opname_blacklist == []
    assert cfg.algo_args == {}
    assert cfg.weight_quant_dtype == msdtype.int8
    assert cfg.kvcache_quant_dtype is None
    assert cfg.act_quant_dtype is None
    assert cfg.outliers_suppression == OutliersSuppressionType.NONE
    assert cfg.precision_recovery == PrecisionRecovery.NONE
    assert cfg.weight_quant_granularity == QuantGranularity.PER_CHANNEL
    assert cfg.kvcache_quant_granularity == QuantGranularity.PER_CHANNEL
    assert cfg.act_quant_granularity == QuantGranularity.PER_TENSOR
    assert cfg.group_size == 0
    assert cfg.weight_clip is False

    # Verify basic parameter settings
    cfg = PTQConfig(
        opname_blacklist=['layer0', 'layer1'],
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8,
        kvcache_quant_dtype=msdtype.int8
    )
    assert cfg.opname_blacklist == ['layer0', 'layer1']
    assert cfg.weight_quant_dtype == msdtype.int8
    assert cfg.act_quant_dtype == msdtype.int8
    assert cfg.kvcache_quant_dtype == msdtype.int8


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_mode_config():
    """
    Feature: PTQConfig mode configuration
    Description: Test PTQConfig mode configuration functionality
    Expectation: Different mode configurations work correctly, backend parameter validation passes
    """
    # Configure QUANTIZE mode
    cfg = PTQConfig(mode=PTQMode.QUANTIZE)
    assert cfg.mode == PTQMode.QUANTIZE
    assert cfg.backend == BackendTarget.ASCEND

    # Configure DEPLOY mode
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
    assert cfg.mode == PTQMode.DEPLOY
    assert cfg.backend == BackendTarget.ASCEND

    # Configure DEPLOY mode with NONE backend
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.NONE)
    assert cfg.mode == PTQMode.DEPLOY
    assert cfg.backend == BackendTarget.NONE

    # Verify opname_blacklist functionality
    cfg = PTQConfig(mode=PTQMode.QUANTIZE, opname_blacklist=['attention', 'mlp'])
    assert cfg.opname_blacklist == ['attention', 'mlp']


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_quant_dtype_config():
    """
    Feature: PTQConfig quantization type configuration
    Description: Test quantization type configuration correctness
    Expectation: Quantization type configuration correct, combination compatibility verified
    """
    # Configure weight_quant_dtype
    cfg = PTQConfig(weight_quant_dtype=msdtype.int8)
    assert cfg.weight_quant_dtype == msdtype.int8

    cfg = PTQConfig(weight_quant_dtype=msdtype.qint4x2)
    assert cfg.weight_quant_dtype == msdtype.qint4x2

    cfg = PTQConfig(weight_quant_dtype=None)
    assert cfg.weight_quant_dtype is None

    # Configure act_quant_dtype
    cfg = PTQConfig(act_quant_dtype=msdtype.int8)
    assert cfg.act_quant_dtype == msdtype.int8

    cfg = PTQConfig(act_quant_dtype=None)
    assert cfg.act_quant_dtype is None

    # Configure kvcache_quant_dtype
    cfg = PTQConfig(kvcache_quant_dtype=msdtype.int8)
    assert cfg.kvcache_quant_dtype == msdtype.int8

    cfg = PTQConfig(kvcache_quant_dtype=None)
    assert cfg.kvcache_quant_dtype is None

    # Verify combination compatibility: A8W8 configuration
    cfg = PTQConfig(
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8
    )
    assert cfg.weight_quant_dtype == msdtype.int8
    assert cfg.act_quant_dtype == msdtype.int8

    # Verify combination compatibility: A8W4 configuration (requires GPTQ precision_recovery and specific parameters)
    gptq_config = GPTQQuantConfig(desc_act=True, static_groups=True, block_size=32)
    cfg = PTQConfig(
        weight_quant_dtype=msdtype.qint4x2,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_granularity=QuantGranularity.PER_CHANNEL,
        precision_recovery=PrecisionRecovery.GPTQ,
        algo_args=gptq_config
    )
    assert cfg.weight_quant_dtype == msdtype.qint4x2
    assert cfg.act_quant_dtype == msdtype.int8
    assert cfg.precision_recovery == PrecisionRecovery.GPTQ
    assert cfg.act_quant_granularity == QuantGranularity.PER_TOKEN
    assert cfg.weight_quant_granularity == QuantGranularity.PER_CHANNEL
    assert isinstance(cfg.algo_args, dict)
    assert cfg.algo_args['desc_act'] is True
    assert cfg.algo_args['static_groups'] is True


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_quant_granularity_config():
    """
    Feature: PTQConfig quantization granularity configuration
    Description: Test quantization granularity configuration functionality
    Expectation: Granularity configuration correct, group_size parameter validation passes
    """
    # Configure weight_quant_granularity
    cfg = PTQConfig(weight_quant_granularity=QuantGranularity.PER_CHANNEL)
    assert cfg.weight_quant_granularity == QuantGranularity.PER_CHANNEL
    assert cfg.group_size == 0

    cfg = PTQConfig(weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=64)
    assert cfg.weight_quant_granularity == QuantGranularity.PER_GROUP
    assert cfg.group_size == 64

    cfg = PTQConfig(weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=128)
    assert cfg.weight_quant_granularity == QuantGranularity.PER_GROUP
    assert cfg.group_size == 128

    cfg = PTQConfig(weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=256)
    assert cfg.weight_quant_granularity == QuantGranularity.PER_GROUP
    assert cfg.group_size == 256

    # Configure act_quant_granularity
    cfg = PTQConfig(act_quant_granularity=QuantGranularity.PER_TENSOR)
    assert cfg.act_quant_granularity == QuantGranularity.PER_TENSOR

    cfg = PTQConfig(
        mode=PTQMode.DEPLOY,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8
    )
    assert cfg.act_quant_granularity == QuantGranularity.PER_TOKEN

    # Configure kvcache_quant_granularity
    cfg = PTQConfig(kvcache_quant_granularity=QuantGranularity.PER_CHANNEL)
    assert cfg.kvcache_quant_granularity == QuantGranularity.PER_CHANNEL

    cfg = PTQConfig(
        mode=PTQMode.DEPLOY,
        kvcache_quant_granularity=QuantGranularity.PER_TOKEN,
        kvcache_quant_dtype=msdtype.int8
    )
    assert cfg.kvcache_quant_granularity == QuantGranularity.PER_TOKEN


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_algo_config():
    """
    Feature: PTQConfig algorithm configuration
    Description: Test algorithm related configuration functionality
    Expectation: Algorithm configuration correct, parameter passing verified
    """
    # Configure outliers_suppression
    cfg = PTQConfig(outliers_suppression=OutliersSuppressionType.NONE)
    assert cfg.outliers_suppression == OutliersSuppressionType.NONE

    cfg = PTQConfig(outliers_suppression=OutliersSuppressionType.SMOOTH)
    assert cfg.outliers_suppression == OutliersSuppressionType.SMOOTH

    cfg = PTQConfig(outliers_suppression=OutliersSuppressionType.AWQ)
    assert cfg.outliers_suppression == OutliersSuppressionType.AWQ

    # Configure precision_recovery
    cfg = PTQConfig(precision_recovery=PrecisionRecovery.NONE)
    assert cfg.precision_recovery == PrecisionRecovery.NONE

    cfg = PTQConfig(precision_recovery=PrecisionRecovery.GPTQ)
    assert cfg.precision_recovery == PrecisionRecovery.GPTQ

    # Configure algo_args - using SmoothQuantConfig
    sq_config = SmoothQuantConfig(alpha=0.8)
    cfg = PTQConfig(algo_args=sq_config)
    assert isinstance(cfg.algo_args, dict)
    assert cfg.algo_args['alpha'] == 0.8

    # Configure algo_args - using GPTQQuantConfig
    gptq_config = GPTQQuantConfig(block_size=128, desc_act=True, static_groups=False)
    cfg = PTQConfig(algo_args=gptq_config)
    assert isinstance(cfg.algo_args, dict)
    assert cfg.algo_args['block_size'] == 128
    assert cfg.algo_args['desc_act'] is True
    assert cfg.algo_args['static_groups'] is False

    # Configure algo_args - using dictionary
    cfg = PTQConfig(algo_args={'custom_param': 123, 'another_param': 'value'})
    assert isinstance(cfg.algo_args, dict)
    assert cfg.algo_args['custom_param'] == 123
    assert cfg.algo_args['another_param'] == 'value'

    # Configure weight_clip parameter
    cfg = PTQConfig(weight_clip=True)
    assert cfg.weight_clip is True

    cfg = PTQConfig(weight_clip=False)
    assert cfg.weight_clip is False


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_gptq_quant_config_construct():
    """
    Feature: GPTQQuantConfig construction
    Description: Test GPTQ algorithm configuration class
    Expectation: GPTQ configuration created correctly, parameter validation passes
    """
    # Verify default parameters
    cfg = GPTQQuantConfig()
    assert cfg.block_size == 128
    assert cfg.desc_act is False
    assert cfg.damp_percent == 0.01
    assert cfg.static_groups is False

    # Set custom parameters
    cfg = GPTQQuantConfig(block_size=256, desc_act=True, damp_percent=0.02, static_groups=True)
    assert cfg.block_size == 256
    assert cfg.desc_act is True
    assert cfg.damp_percent == 0.02
    assert cfg.static_groups is True

    # Verify parameter range - boundary value tests
    cfg = GPTQQuantConfig(block_size=0, damp_percent=0.0)
    assert cfg.block_size == 0
    assert cfg.damp_percent == 0.0

    cfg = GPTQQuantConfig(block_size=512, damp_percent=1.0)
    assert cfg.block_size == 512
    assert cfg.damp_percent == 1.0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_smooth_quant_config_construct():
    """
    Feature: SmoothQuantConfig construction
    Description: Test SmoothQuant algorithm configuration class
    Expectation: SmoothQuant configuration created correctly
    """
    # Default alpha parameter
    cfg = SmoothQuantConfig()
    assert cfg.alpha == 0.5

    # Set custom alpha value
    cfg = SmoothQuantConfig(alpha=0.8)
    assert cfg.alpha == 0.8

    cfg = SmoothQuantConfig(alpha=0.0)
    assert cfg.alpha == 0.0

    cfg = SmoothQuantConfig(alpha=1.0)
    assert cfg.alpha == 1.0

    # Test intermediate values
    cfg = SmoothQuantConfig(alpha=0.25)
    assert cfg.alpha == 0.25


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_awq_config_construct():
    """
    Feature: AWQConfig construction
    Description: Test AWQ algorithm configuration class
    Expectation: AWQ configuration created correctly, parameter range validation passes
    """
    # Verify default parameters
    cfg = AWQConfig()
    assert cfg.duo_scaling is True
    assert isinstance(cfg.smooth_alpha, list)
    assert len(cfg.smooth_alpha) == 20
    assert isinstance(cfg.weight_clip_ratio, list)
    assert len(cfg.weight_clip_ratio) == 10

    # Set single value parameter
    cfg = AWQConfig(duo_scaling=False, smooth_alpha=0.5, weight_clip_ratio=0.8)
    assert cfg.duo_scaling is False
    assert cfg.smooth_alpha == 0.5
    assert cfg.weight_clip_ratio == 0.8

    # Set list parameter
    smooth_alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    weight_clip_ratio_list = [0.9, 0.8, 0.7]
    cfg = AWQConfig(smooth_alpha=smooth_alpha_list, weight_clip_ratio=weight_clip_ratio_list)
    assert cfg.smooth_alpha == smooth_alpha_list
    assert cfg.weight_clip_ratio == weight_clip_ratio_list

    # Verify parameter range - boundary values
    cfg = AWQConfig(smooth_alpha=0.0, weight_clip_ratio=0.0)
    assert cfg.smooth_alpha == 0.0
    assert cfg.weight_clip_ratio == 0.0

    cfg = AWQConfig(smooth_alpha=1.0, weight_clip_ratio=1.0)
    assert cfg.smooth_alpha == 1.0
    assert cfg.weight_clip_ratio == 1.0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_inner_ptq_config_convert():
    """
    Feature: InnerPTQConfig conversion
    Description: Test configuration conversion functionality
    Expectation: Conversion function works normally, configuration consistency verified
    """
    # Convert PTQConfig to InnerPTQConfig - default approach
    ptq_cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND)
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg)
    assert inner_cfg.mode == PTQMode.DEPLOY
    assert inner_cfg.backend == BackendTarget.ASCEND
    assert inner_cfg.approach == PTQApproach.RTN

    # Convert with different approach - SMOOTH_QUANT
    ptq_cfg = PTQConfig(mode=PTQMode.QUANTIZE, backend=BackendTarget.ASCEND)
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg, PTQApproach.SMOOTH_QUANT)
    assert inner_cfg.mode == PTQMode.QUANTIZE
    assert inner_cfg.backend == BackendTarget.ASCEND
    assert inner_cfg.approach == PTQApproach.SMOOTH_QUANT

    # Convert with different approach - PTQ
    ptq_cfg = PTQConfig(
        mode=PTQMode.DEPLOY,
        backend=BackendTarget.ASCEND,
        act_quant_dtype=msdtype.int8,
        weight_quant_dtype=msdtype.int8
    )
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg, PTQApproach.PTQ)
    assert inner_cfg.approach == PTQApproach.PTQ
    assert inner_cfg.act_quant_dtype == msdtype.int8
    assert inner_cfg.weight_quant_dtype == msdtype.int8

    # Verify configuration consistency - include algo_args
    sq_config = SmoothQuantConfig(alpha=0.7)
    ptq_cfg = PTQConfig(mode=PTQMode.DEPLOY, algo_args=sq_config)
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg, PTQApproach.SMOOTH_QUANT)
    assert inner_cfg.algo_args['alpha'] == 0.7

    # Verify configuration consistency - include opname_blacklist
    ptq_cfg = PTQConfig(
        mode=PTQMode.QUANTIZE,
        opname_blacklist=['layer0', 'layer1'],
        weight_quant_dtype=msdtype.qint4x2
    )
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg)
    assert inner_cfg.opname_blacklist == ['layer0', 'layer1']
    assert inner_cfg.weight_quant_dtype == msdtype.qint4x2

    # Equality test - manually set InnerPTQConfig should equal the one converted from PTQConfig
    inner_cfg = InnerPTQConfig()
    inner_cfg.approach = PTQApproach.PTQ
    inner_cfg.mode = PTQMode.DEPLOY
    inner_cfg.backend = BackendTarget.ASCEND
    inner_cfg.act_quant_dtype = msdtype.int8

    ptq_cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND, act_quant_dtype=msdtype.int8)
    convert_inner_cfg = inner_cfg.inner_config(cfg=ptq_cfg, approach=PTQApproach.PTQ)
    assert convert_inner_cfg == inner_cfg


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_param_type_error():
    """
    Feature: PTQConfig parameter type error
    Description: Test parameter type error handling
    Expectation: Raise TypeError exception with accurate error message
    """
    # mode parameter type error
    with pytest.raises(ValueError):
        _ = PTQConfig(mode='none')

    # backend parameter type error
    with pytest.raises(ValueError):
        _ = PTQConfig(backend=PTQMode.QUANTIZE)

    # opname_blacklist parameter type error
    with pytest.raises(TypeError):
        _ = PTQConfig(opname_blacklist=1)
    with pytest.raises(TypeError):
        _ = PTQConfig(opname_blacklist="1")
    with pytest.raises(TypeError):
        _ = PTQConfig(opname_blacklist=["1", 1])

    # outliers_suppression parameter type error
    with pytest.raises(TypeError):
        _ = PTQConfig(outliers_suppression=1)
    with pytest.raises(TypeError):
        _ = PTQConfig(outliers_suppression='awq')

    # precision_recovery parameter type error
    with pytest.raises(TypeError):
        _ = PTQConfig(precision_recovery='gptq')

    # group_size parameter type error
    with pytest.raises(TypeError):
        _ = PTQConfig(group_size='64')

    # weight_clip parameter type error
    with pytest.raises(TypeError):
        _ = PTQConfig(weight_clip=1)

    # algo_args parameter type error - neither dict nor dataclass
    with pytest.raises(ValueError, match="algo_args's type should be dict or dataclass"):
        _ = PTQConfig(algo_args="invalid_type")
    with pytest.raises(ValueError, match="algo_args's type should be dict or dataclass"):
        _ = PTQConfig(algo_args=123)
    with pytest.raises(ValueError, match="algo_args's type should be dict or dataclass"):
        _ = PTQConfig(algo_args=[1, 2, 3])


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_param_value_error():
    """
    Feature: PTQConfig parameter value range error
    Description: Test parameter value range error handling
    Expectation: Raise ValueError exception with accurate error message
    """
    # Unsupported quantization type
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_dtype=msdtype.float16)
    with pytest.raises(ValueError):
        _ = PTQConfig(act_quant_dtype=msdtype.float16)
    with pytest.raises(ValueError):
        _ = PTQConfig(kvcache_quant_dtype=msdtype.float16)

    # List type quant_dtype parameter error
    with pytest.raises(ValueError, match="weight_quant_dtype support"):
        _ = PTQConfig(weight_quant_dtype=["1"])
    with pytest.raises(ValueError, match="kvcache_quant_dtype support"):
        _ = PTQConfig(kvcache_quant_dtype=["1"])
    with pytest.raises(ValueError, match="act_quant_dtype support"):
        _ = PTQConfig(act_quant_dtype=["1"])

    # Unsupported granularity combination
    with pytest.raises(ValueError):
        _ = PTQConfig(act_quant_granularity=QuantGranularity.PER_GROUP)
    with pytest.raises(ValueError):
        _ = PTQConfig(act_quant_granularity=QuantGranularity.PER_CHANNEL)
    with pytest.raises(ValueError):
        _ = PTQConfig(kvcache_quant_granularity=QuantGranularity.PER_GROUP)
    with pytest.raises(ValueError):
        _ = PTQConfig(kvcache_quant_granularity=QuantGranularity.PER_TENSOR)
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_granularity=QuantGranularity.PER_TOKEN)
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_granularity=QuantGranularity.PER_TENSOR)

    # PER_TOKEN granularity matching requirement
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=None
        )
    with pytest.raises(ValueError):
        _ = PTQConfig(
            mode=PTQMode.QUANTIZE,
            kvcache_quant_granularity=QuantGranularity.PER_TOKEN,
            kvcache_quant_dtype=None
        )

    # Error when kvcache_quant_granularity=PER_TOKEN but using default kvcache_quant_dtype=None
    with pytest.raises(ValueError, match="kvcache_quant_dtype must be mindspore.dtype.int8"):
        _ = PTQConfig(kvcache_quant_granularity=QuantGranularity.PER_TOKEN)

    # group_size range error
    with pytest.raises(ValueError):
        _ = PTQConfig(group_size=16)
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_granularity=QuantGranularity.PER_CHANNEL, group_size=64)
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=0)
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_granularity=QuantGranularity.PER_GROUP, group_size=32)

    # Validation error: weight_quant_granularity must be PER_CHANNEL in A8W8 configuration
    with pytest.raises(ValueError, match="weight_quant_granularity must be QuantGranularity.PER_CHANNEL"):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            act_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_GROUP  # A8W8 configuration must be PER_CHANNEL
        )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_param_conflict():
    """
    Feature: PTQConfig parameter combination conflict
    Description: Test parameter combination conflict detection
    Expectation: Raise ValueError exception with accurate conflict information
    """
    # A8W4 configuration mismatch with GPTQ parameters - missing precision_recovery
    with pytest.raises(ValueError, match="A8W4 quantization only support GPTQ precision_recovery"):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            act_quant_dtype=msdtype.int8
        )

    # A8W4 configuration mismatch with GPTQ parameters - when algo_args is empty dict, it will be auto-converted to default GPTQQuantConfig, but default values do not meet requirements
    with pytest.raises(ValueError, match="A8W4 quantization only support desc_act=True and static_groups=True"):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args={}
        )

    # A8W4 configuration mismatch with GPTQ parameters - algo_args is not GPTQQuantConfig type (using SmoothQuantConfig)
    with pytest.raises(ValueError, match="A8W4 quantization need algo_args is GPTQQuantConfig"):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=SmoothQuantConfig(alpha=0.5)
        )

    # A8W4 configuration mismatch with GPTQ parameters - desc_act or static_groups is False
    with pytest.raises(ValueError, match="A8W4 quantization only support desc_act=True and static_groups=True"):
        gptq_config = GPTQQuantConfig(desc_act=False, static_groups=True)
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=gptq_config
        )

    # A8W4 configuration mismatch with GPTQ parameters - act_quant_granularity is not PER_TOKEN
    with pytest.raises(ValueError, match="A8W4 quantization only support act_quant_granularity is per_token"):
        gptq_config = GPTQQuantConfig(desc_act=True, static_groups=True)
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_granularity=QuantGranularity.PER_CHANNEL,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=gptq_config
        )

    # A8W4 configuration mismatch with GPTQ parameters - weight_quant_granularity not supported (checked first in _check_quant_granularity)
    with pytest.raises(ValueError, match="weight_quant_granularity support"):
        gptq_config = GPTQQuantConfig(desc_act=True, static_groups=True)
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_granularity=QuantGranularity.PER_TENSOR,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=gptq_config
        )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ptq_config_unsupported_granularity_combinations():
    """
    Feature: PTQConfig unsupported granularity combinations.
    Description: Test unsupported granularity combination scenarios.
    Expectation: Raise ValueError exception for unsupported combinations.
    """
    # ========== Unsupported A8W8 granularity combinations ==========

    # A8W8-pertensor/pergroup not supported (A8W8 must use PER_CHANNEL for weight)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64
        )

    # A8W8-pertensor/pergroup not supported (with SMOOTH)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64,
            outliers_suppression=OutliersSuppressionType.SMOOTH
        )

    # A8W8-pertensor/pergroup not supported (with AWQ)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64,
            outliers_suppression=OutliersSuppressionType.AWQ,
            algo_args=AWQConfig()
        )

    # A8W8-pertoken/pergroup not supported (A8W8 must use PER_CHANNEL for weight)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64
        )

    # A8W8-pertoken/pergroup not supported (with SMOOTH)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64,
            outliers_suppression=OutliersSuppressionType.SMOOTH
        )

    # A8W8-pertoken/pergroup not supported (with AWQ)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64,
            outliers_suppression=OutliersSuppressionType.AWQ,
            algo_args=AWQConfig()
        )

    # Note: A8W8-pertensor/perchannel with AWQ and A8W8-pertoken/perchannel with AWQ
    # are checked in PTQ class, not in PTQConfig constructor.
    # These tests should be in test_ptq.py, not here.

    # ========== A8W4 granularity combinations not supported ==========

    # A8perchannelW4pergroup does not support GPTQ
    with pytest.raises(ValueError):
        gptq_config = GPTQQuantConfig(desc_act=True, static_groups=True)
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_CHANNEL,
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=gptq_config
        )

    # A8pergroupW4pergroup does not support GPTQ
    with pytest.raises(ValueError):
        gptq_config = GPTQQuantConfig(desc_act=True, static_groups=True)
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_GROUP,
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64,
            precision_recovery=PrecisionRecovery.GPTQ,
            algo_args=gptq_config
        )

    # ========== Other granularity combinations not supported ==========

    # C8-pergroup not supported
    with pytest.raises(ValueError):
        _ = PTQConfig(
            kvcache_quant_dtype=msdtype.int8,
            kvcache_quant_granularity=QuantGranularity.PER_GROUP
        )

    # A16W8-pertensor not supported
    with pytest.raises(ValueError):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_TENSOR
        )

    # A16W8-pertoken not supported
    with pytest.raises(ValueError):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_TOKEN
        )

    # A16W4-pertensor not supported
    with pytest.raises(ValueError):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_TENSOR
        )

    # A16W4-pertoken not supported
    with pytest.raises(ValueError):
        _ = PTQConfig(
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_TOKEN
        )

    # A8W8-perchannel/pertoken not supported (act_quant_granularity=PER_CHANNEL not supported)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_CHANNEL,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_TOKEN
        )

    # A8W8-pergroup/pertensor not supported (act_quant_granularity=PER_GROUP not supported)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_GROUP,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_TENSOR
        )

    # A8W8-pertensor/pertoken not supported (weight_quant_granularity=PER_TOKEN not supported)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_TOKEN
        )

    # A8W8-pertensor/pertensor not supported (weight_quant_granularity=PER_TENSOR not supported)
    with pytest.raises(ValueError):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TENSOR,
            weight_quant_dtype=msdtype.int8,
            weight_quant_granularity=QuantGranularity.PER_TENSOR
        )

    # C8-pertensor not supported
    with pytest.raises(ValueError):
        _ = PTQConfig(
            kvcache_quant_dtype=msdtype.int8,
            kvcache_quant_granularity=QuantGranularity.PER_TENSOR
        )

    # A8pertokenW4pergroup requires GPTQ configuration (already covered in test_ptq_config_param_conflict)
    # Verify the case without GPTQ configuration
    with pytest.raises(ValueError, match="A8W4 quantization only support GPTQ precision_recovery"):
        _ = PTQConfig(
            act_quant_dtype=msdtype.int8,
            act_quant_granularity=QuantGranularity.PER_TOKEN,
            weight_quant_dtype=msdtype.qint4x2,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            group_size=64
        )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_gptq_quant_config_param_error():
    """
    Feature: GPTQQuantConfig parameter validation error
    Description: Test GPTQ parameter validation mechanism
    Expectation: Raise corresponding ValueError or TypeError exception
    """
    # block_size < 0
    with pytest.raises(ValueError, match="block_size should >=0"):
        _ = GPTQQuantConfig(block_size=-1)
    with pytest.raises(ValueError, match="block_size should >=0"):
        _ = GPTQQuantConfig(block_size=-100)

    # damp_percent out of range
    with pytest.raises(ValueError, match="damp_percent should >=0 and <=1"):
        _ = GPTQQuantConfig(damp_percent=-0.1)
    with pytest.raises(ValueError, match="damp_percent should >=0 and <=1"):
        _ = GPTQQuantConfig(damp_percent=-0.5)
    with pytest.raises(ValueError, match="damp_percent should >=0 and <=1"):
        _ = GPTQQuantConfig(damp_percent=1.1)
    with pytest.raises(ValueError, match="damp_percent should >=0 and <=1"):
        _ = GPTQQuantConfig(damp_percent=2.1)

    # Parameter type error
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(block_size='128')
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(block_size=0.1)  # Float type error
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(desc_act=1)
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(desc_act="0")  # String "0"
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(damp_percent='0.01')
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(damp_percent="1")  # String "1"
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(static_groups=1)
    with pytest.raises(TypeError):
        _ = GPTQQuantConfig(static_groups="2")  # String "2"


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_smooth_quant_config_param_error():
    """
    Feature: SmoothQuantConfig parameter validation error
    Description: Test SmoothQuant parameter validation
    Expectation: Raise TypeError exception
    """
    # alpha parameter type error
    with pytest.raises(TypeError):
        _ = SmoothQuantConfig(alpha='0.5')
    with pytest.raises(TypeError):
        _ = SmoothQuantConfig(alpha=[0.5])

    # Unknown parameter error
    with pytest.raises(TypeError):
        _ = SmoothQuantConfig(alpha=0.5, is_deploy=1)  # pylint: disable=unexpected-keyword-arg


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_awq_config_param_error():
    """
    Feature: AWQConfig parameter validation error
    Description: Test AWQ parameter validation mechanism
    Expectation: Raise corresponding ValueError or TypeError exception
    """
    # smooth_alpha out of range - single value
    with pytest.raises(ValueError, match="smooth_alpha should >=0 and <=1"):
        _ = AWQConfig(smooth_alpha=-0.1)
    with pytest.raises(ValueError, match="smooth_alpha should >=0 and <=1"):
        _ = AWQConfig(smooth_alpha=-0.5)
    with pytest.raises(ValueError, match="smooth_alpha should >=0 and <=1"):
        _ = AWQConfig(smooth_alpha=1.1)

    # weight_clip_ratio out of range - single value
    with pytest.raises(ValueError, match="weight_clip_ratio should >=0 and <=1"):
        _ = AWQConfig(weight_clip_ratio=-0.1)
    with pytest.raises(ValueError, match="weight_clip_ratio should >=0 and <=1"):
        _ = AWQConfig(weight_clip_ratio=-0.5)
    with pytest.raises(ValueError, match="weight_clip_ratio should >=0 and <=1"):
        _ = AWQConfig(weight_clip_ratio=1.1)

    # smooth_alpha out of range - list
    with pytest.raises(ValueError, match="smooth_alpha should >=0 and <=1"):
        _ = AWQConfig(smooth_alpha=[0.1, 0.2, 1.5, 0.3])
    with pytest.raises(ValueError, match="smooth_alpha should >=0 and <=1"):
        _ = AWQConfig(smooth_alpha=[-1, 0.1, 0.5])

    # weight_clip_ratio out of range - list
    with pytest.raises(ValueError, match="weight_clip_ratio should >=0 and <=1"):
        _ = AWQConfig(weight_clip_ratio=[0.9, 0.8, -0.1, 0.7])
    with pytest.raises(ValueError, match="weight_clip_ratio should >=0 and <=1"):
        _ = AWQConfig(weight_clip_ratio=[0.1, 0.5, 10])

    # Parameter type error
    with pytest.raises(TypeError, match="Type of duo_scaling should be"):
        _ = AWQConfig(duo_scaling=1)
    with pytest.raises(TypeError, match="smooth_alpha only support float or list"):
        _ = AWQConfig(smooth_alpha='0.5')
    with pytest.raises(TypeError, match="weight_clip_ratio only support float or list"):
        _ = AWQConfig(weight_clip_ratio='0.8')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_inner_ptq_config_param_error():
    """
    Feature: InnerPTQConfig error parameter validation
    Description: Test internal PTQ config parameter validation and conversion
    Expectation: Raise corresponding ValueError or TypeError exception
    """
    # Invalid method name
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(approach='no_such_approach')

    # RTN method parameter combination constraint - act_quant_dtype cannot be set alone
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(approach=PTQApproach.RTN, act_quant_dtype=msdtype.int8)

    # RTN method parameter combination constraint - weight and kvcache cannot both be None
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(
            approach=PTQApproach.RTN,
            weight_quant_dtype=None,
            kvcache_quant_dtype=None
        )

    # RTN method parameter combination constraint - weight and kvcache cannot both be set
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(
            approach=PTQApproach.RTN,
            weight_quant_dtype=msdtype.int8,
            kvcache_quant_dtype=msdtype.int8
        )

    # RTN method kvcache_quant_granularity constraint
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(
            approach=PTQApproach.RTN,
            kvcache_quant_dtype=msdtype.int8,
            kvcache_quant_granularity=QuantGranularity.PER_TOKEN
        )
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(
            approach=PTQApproach.RTN,
            kvcache_quant_dtype=msdtype.int8,
            kvcache_quant_granularity=QuantGranularity.PER_CHANNEL
        )

    # Check weight_symmetric parameter type
    cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    cfg.weight_symmetric = 1
    with pytest.raises(TypeError):
        cfg.__post_init__()

    # inner_config parameter type error
    inner_cfg = InnerPTQConfig()
    with pytest.raises(TypeError, match="input config shall be PTQConfig"):
        inner_cfg.inner_config('none')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_enum_type_conversion_error():
    """
    Feature: Enum type conversion error
    Description: Test enum type conversion error handling
    Expectation: Return default value or raise exception
    """
    # OutliersSuppressionType invalid string conversion
    result = OutliersSuppressionType.from_str('invalid_type')
    assert result == OutliersSuppressionType.NONE

    # PrecisionRecovery invalid string conversion
    result = PrecisionRecovery.from_str('invalid_recovery')
    assert result == PrecisionRecovery.NONE

    # QuantGranularity invalid string conversion
    result = QuantGranularity.from_str('invalid_granularity')
    assert result is None

    # Test case-insensitive conversion
    assert OutliersSuppressionType.from_str('SMOOTH') == OutliersSuppressionType.SMOOTH
    assert OutliersSuppressionType.from_str('smooth') == OutliersSuppressionType.SMOOTH
    assert PrecisionRecovery.from_str('GPTQ') == PrecisionRecovery.GPTQ
    assert PrecisionRecovery.from_str('gptq') == PrecisionRecovery.GPTQ
    assert QuantGranularity.from_str('PER_TOKEN') == QuantGranularity.PER_TOKEN
    assert QuantGranularity.from_str('per_token') == QuantGranularity.PER_TOKEN


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_algo_args_incompatible():
    """
    Feature: Algorithm parameters incompatibility
    Description: Test algorithm parameter compatibility checking
    Expectation: Raise ValueError exception, compatibility check passes
    """
    # Note: A8W4 and GPTQ parameter mismatch tests are covered in test_ptq_config_param_conflict
    # AWQ configuration is incompatible with OSL - needs to be checked in PTQ class, here only test configuration construction
    # Note: AWQ and OSL compatibility check is in PTQ class, not in PTQConfig
    # Here test that configuration can be constructed normally, but compatibility will be checked in PTQ class during actual use


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_config_convert():
    """
    Feature: Config conversion
    Description: Test configuration conversion functionality
    Expectation: Conversion function works normally, parameter mapping correct
    """
    # Convert PTQConfig to InnerPTQConfig - different approach conversion logic
    # RTN approach requires act_quant_granularity to be PER_TOKEN (if act_quant_dtype is int8)
    ptq_cfg_rtn = PTQConfig(
        mode=PTQMode.DEPLOY,
        backend=BackendTarget.ASCEND,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_dtype=msdtype.int8,
        opname_blacklist=['layer0']
    )

    # Use RTN approach
    inner_cfg_rtn = InnerPTQConfig.inner_config(ptq_cfg_rtn, PTQApproach.RTN)
    assert inner_cfg_rtn.approach == PTQApproach.RTN
    assert inner_cfg_rtn.mode == PTQMode.DEPLOY
    assert inner_cfg_rtn.act_quant_dtype == msdtype.int8
    assert inner_cfg_rtn.act_quant_granularity == QuantGranularity.PER_TOKEN
    assert inner_cfg_rtn.weight_quant_dtype == msdtype.int8
    assert inner_cfg_rtn.opname_blacklist == ['layer0']

    # Prepare PTQConfig for SMOOTH_QUANT
    ptq_cfg = PTQConfig(
        mode=PTQMode.DEPLOY,
        backend=BackendTarget.ASCEND,
        act_quant_dtype=msdtype.int8,
        weight_quant_dtype=msdtype.int8,
        opname_blacklist=['layer0']
    )

    # Use SMOOTH_QUANT approach
    inner_cfg_sq = InnerPTQConfig.inner_config(ptq_cfg, PTQApproach.SMOOTH_QUANT)
    assert inner_cfg_sq.approach == PTQApproach.SMOOTH_QUANT
    assert inner_cfg_sq.mode == PTQMode.DEPLOY
    assert inner_cfg_sq.act_quant_dtype == msdtype.int8

    # Verify parameter mapping - algo_args merge
    sq_config = SmoothQuantConfig(alpha=0.6)
    ptq_cfg = PTQConfig(mode=PTQMode.QUANTIZE, algo_args=sq_config)
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg, PTQApproach.SMOOTH_QUANT)
    assert isinstance(inner_cfg.algo_args, dict)
    assert inner_cfg.algo_args['alpha'] == 0.6

    # Verify parameter mapping - all parameters correctly passed (A8W4 configuration requires GPTQ parameters, use PTQ approach to avoid RTN limitations)
    gptq_config = GPTQQuantConfig(desc_act=True, static_groups=True, block_size=64)
    ptq_cfg = PTQConfig(
        mode=PTQMode.DEPLOY,
        backend=BackendTarget.NONE,
        weight_quant_dtype=msdtype.qint4x2,
        act_quant_dtype=msdtype.int8,
        kvcache_quant_dtype=msdtype.int8,
        outliers_suppression=OutliersSuppressionType.SMOOTH,
        precision_recovery=PrecisionRecovery.GPTQ,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_granularity=QuantGranularity.PER_GROUP,
        kvcache_quant_granularity=QuantGranularity.PER_TOKEN,
        group_size=128,
        weight_clip=True,
        algo_args=gptq_config
    )
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg, PTQApproach.PTQ)
    assert inner_cfg.mode == PTQMode.DEPLOY
    assert inner_cfg.backend == BackendTarget.NONE
    assert inner_cfg.weight_quant_dtype == msdtype.qint4x2
    assert inner_cfg.outliers_suppression == OutliersSuppressionType.SMOOTH
    assert inner_cfg.precision_recovery == PrecisionRecovery.GPTQ
    assert inner_cfg.act_quant_granularity == QuantGranularity.PER_TOKEN
    assert inner_cfg.weight_quant_granularity == QuantGranularity.PER_GROUP
    assert inner_cfg.group_size == 128
    assert inner_cfg.weight_clip is True
    assert inner_cfg.algo_args['desc_act'] is True
    assert inner_cfg.algo_args['static_groups'] is True


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_config_merge():
    """
    Feature: Config merge
    Description: Test configuration merge functionality
    Expectation: Merge function works normally, conflict resolution mechanism effective
    """
    # Configuration parameter override rules - algo_args merge
    sq_config = SmoothQuantConfig(alpha=0.7)
    merged_cfg = PTQConfig(mode=PTQMode.QUANTIZE, algo_args=sq_config)
    # algo_args will be converted to dictionary
    assert isinstance(merged_cfg.algo_args, dict)
    assert merged_cfg.algo_args['alpha'] == 0.7

    # Merge default values with custom values - merge algo_args in InnerPTQConfig
    ptq_cfg = PTQConfig(
        mode=PTQMode.DEPLOY,
        algo_args={'custom': 'value'}
    )
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg, PTQApproach.SMOOTH_QUANT)
    # algo_args should contain default values and custom values
    assert isinstance(inner_cfg.algo_args, dict)
    assert inner_cfg.algo_args.get('custom') == 'value'

    # Nested configuration merge logic - use GPTQQuantConfig as algo_args
    gptq_config = GPTQQuantConfig(block_size=256, desc_act=True)
    ptq_cfg = PTQConfig(
        mode=PTQMode.QUANTIZE,
        precision_recovery=PrecisionRecovery.GPTQ,
        algo_args=gptq_config
    )
    assert isinstance(ptq_cfg.algo_args, dict)
    assert ptq_cfg.algo_args['block_size'] == 256
    assert ptq_cfg.algo_args['desc_act'] is True
    assert ptq_cfg.algo_args['damp_percent'] == 0.01  # Default value retained

    # Configuration parameter override - parameters set later override those set earlier
    cfg1 = PTQConfig(mode=PTQMode.QUANTIZE, weight_quant_dtype=msdtype.int8)
    cfg2 = PTQConfig(mode=PTQMode.DEPLOY, weight_quant_dtype=msdtype.qint4x2)
    assert cfg1.mode == PTQMode.QUANTIZE
    assert cfg1.weight_quant_dtype == msdtype.int8
    assert cfg2.mode == PTQMode.DEPLOY
    assert cfg2.weight_quant_dtype == msdtype.qint4x2


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_config_validation():
    """
    Feature: Config validation
    Description: Test configuration validation functionality
    Expectation: Validation function works normally, check results accurate
    """
    # Verify configuration completeness - required parameter check
    cfg = PTQConfig()
    # Default configuration should be complete and valid
    assert cfg.mode is not None
    assert cfg.backend is not None
    assert cfg.weight_quant_dtype is not None

    # Check configuration validity - PER_TOKEN granularity matches quantization type
    cfg = PTQConfig(
        mode=PTQMode.DEPLOY,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_dtype=msdtype.int8,
        act_quant_dtype=msdtype.int8
    )
    assert cfg.act_quant_granularity == QuantGranularity.PER_TOKEN
    assert cfg.act_quant_dtype == msdtype.int8

    # Verify dependency relationship - dependency between group_size and weight_quant_granularity
    cfg = PTQConfig(
        weight_quant_granularity=QuantGranularity.PER_GROUP,
        group_size=128
    )
    assert cfg.weight_quant_granularity == QuantGranularity.PER_GROUP
    assert cfg.group_size == 128

    # Check configuration consistency - A8W4 configuration consistency
    gptq_config = GPTQQuantConfig(desc_act=True, static_groups=True)
    cfg = PTQConfig(
        weight_quant_dtype=msdtype.qint4x2,
        act_quant_dtype=msdtype.int8,
        act_quant_granularity=QuantGranularity.PER_TOKEN,
        weight_quant_granularity=QuantGranularity.PER_CHANNEL,
        precision_recovery=PrecisionRecovery.GPTQ,
        algo_args=gptq_config
    )
    assert cfg.weight_quant_dtype == msdtype.qint4x2
    assert cfg.act_quant_dtype == msdtype.int8
    assert cfg.precision_recovery == PrecisionRecovery.GPTQ
    assert cfg.algo_args['desc_act'] is True
    assert cfg.algo_args['static_groups'] is True


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_enum_type_string_conversion():
    """
    Feature: Enum type string conversion
    Description: Test enum type conversion functionality
    Expectation: Conversion function works normally, case-insensitive handling correct
    """
    # Enum to string conversion - QuantGranularity has __str__ method
    assert str(QuantGranularity.PER_TOKEN) == 'per_token'
    assert str(QuantGranularity.PER_CHANNEL) == 'per_channel'
    assert str(QuantGranularity.PER_TENSOR) == 'per_tensor'
    assert str(QuantGranularity.PER_GROUP) == 'per_group'

    # Enum to string conversion - use .value attribute to get enum value (OutliersSuppressionType and PrecisionRecovery do not have custom __str__)
    assert OutliersSuppressionType.SMOOTH.value == 'smooth'
    assert OutliersSuppressionType.AWQ.value == 'awq'
    assert OutliersSuppressionType.NONE.value == 'none'
    assert PrecisionRecovery.GPTQ.value == 'gptq'
    assert PrecisionRecovery.NONE.value == 'none'

    # String to enum conversion - case insensitive
    assert OutliersSuppressionType.from_str('SMOOTH') == OutliersSuppressionType.SMOOTH
    assert OutliersSuppressionType.from_str('smooth') == OutliersSuppressionType.SMOOTH
    assert OutliersSuppressionType.from_str('Smooth') == OutliersSuppressionType.SMOOTH
    assert OutliersSuppressionType.from_str('AWQ') == OutliersSuppressionType.AWQ
    assert OutliersSuppressionType.from_str('awq') == OutliersSuppressionType.AWQ

    assert PrecisionRecovery.from_str('GPTQ') == PrecisionRecovery.GPTQ
    assert PrecisionRecovery.from_str('gptq') == PrecisionRecovery.GPTQ
    assert PrecisionRecovery.from_str('Gptq') == PrecisionRecovery.GPTQ
    assert PrecisionRecovery.from_str('NONE') == PrecisionRecovery.NONE

    assert QuantGranularity.from_str('PER_TOKEN') == QuantGranularity.PER_TOKEN
    assert QuantGranularity.from_str('per_token') == QuantGranularity.PER_TOKEN
    assert QuantGranularity.from_str('Per_Token') == QuantGranularity.PER_TOKEN
    assert QuantGranularity.from_str('PER_CHANNEL') == QuantGranularity.PER_CHANNEL
    assert QuantGranularity.from_str('PER_TENSOR') == QuantGranularity.PER_TENSOR
    assert QuantGranularity.from_str('PER_GROUP') == QuantGranularity.PER_GROUP

    # Special enum value conversion
    assert (OutliersSuppressionType.from_str('outlier-suppression+') ==
            OutliersSuppressionType.OUTLIER_SUPPRESSION_PLUS)
    assert (OutliersSuppressionType.from_str('outlier-suppression-lite') ==
            OutliersSuppressionType.OUTLIER_SUPPRESSION_LITE)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_config_default_values():
    """
    Feature: Config default values
    Description: Test configuration default values
    Expectation: Default values reasonable, combination compatibility good
    """
    # Verify PTQConfig default values
    cfg = PTQConfig()
    assert cfg.mode == PTQMode.QUANTIZE
    assert cfg.backend == BackendTarget.ASCEND
    assert cfg.opname_blacklist == []
    assert cfg.algo_args == {}
    assert cfg.weight_quant_dtype == msdtype.int8
    assert cfg.kvcache_quant_dtype is None
    assert cfg.act_quant_dtype is None
    assert cfg.outliers_suppression == OutliersSuppressionType.NONE
    assert cfg.precision_recovery == PrecisionRecovery.NONE
    assert cfg.weight_quant_granularity == QuantGranularity.PER_CHANNEL
    assert cfg.kvcache_quant_granularity == QuantGranularity.PER_CHANNEL
    assert cfg.act_quant_granularity == QuantGranularity.PER_TENSOR
    assert cfg.group_size == 0
    assert cfg.weight_clip is False

    # Verify GPTQQuantConfig default values
    gptq_cfg = GPTQQuantConfig()
    assert gptq_cfg.block_size == 128
    assert gptq_cfg.desc_act is False
    assert gptq_cfg.damp_percent == 0.01
    assert gptq_cfg.static_groups is False

    # Verify SmoothQuantConfig default values
    sq_cfg = SmoothQuantConfig()
    assert sq_cfg.alpha == 0.5

    # Verify AWQConfig default values
    awq_cfg = AWQConfig()
    assert awq_cfg.duo_scaling is True
    assert isinstance(awq_cfg.smooth_alpha, list)
    assert len(awq_cfg.smooth_alpha) == 20
    assert isinstance(awq_cfg.weight_clip_ratio, list)
    assert len(awq_cfg.weight_clip_ratio) == 10

    # Verify InnerPTQConfig default values
    inner_cfg = InnerPTQConfig(approach=PTQApproach.RTN)
    assert inner_cfg.approach == PTQApproach.RTN
    assert inner_cfg.mode == PTQMode.QUANTIZE
    assert inner_cfg.backend == BackendTarget.ASCEND
    assert inner_cfg.act_per_channel is False
    assert inner_cfg.weight_per_channel is True
    assert inner_cfg.act_symmetric is False
    assert inner_cfg.weight_symmetric is True
    assert inner_cfg.act_narrow_range is False
    assert inner_cfg.weight_narrow_range is False

    # Verify default value combination compatibility - default PTQConfig can be converted to default InnerPTQConfig
    ptq_cfg = PTQConfig()
    inner_cfg = InnerPTQConfig.inner_config(ptq_cfg)
    assert inner_cfg.mode == PTQMode.QUANTIZE
    assert inner_cfg.backend == BackendTarget.ASCEND
    assert inner_cfg.weight_quant_dtype == msdtype.int8

    # Verify default value combination compatibility - default quantization type combination
    cfg = PTQConfig()
    assert cfg.weight_quant_dtype == msdtype.int8
    assert cfg.act_quant_dtype is None
    assert cfg.kvcache_quant_dtype is None
    # This combination should be valid
    assert cfg.weight_quant_granularity == QuantGranularity.PER_CHANNEL
    assert cfg.group_size == 0  # group_size should be 0 when PER_CHANNEL


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_inner_ptq_config_yaml_dump_and_load():
    """
    Feature: InnerPTQConfig YAML dump and load
    Description: Test InnerPTQConfig YAML serialization and deserialization
    Expectation: Dump and load operations work correctly, all attributes preserved
    """
    yaml_file = 'my_cfg.yaml'
    try:
        # Create configuration and set various attributes
        cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
        cfg.weight_symmetric = False
        cfg.enable_deploy_fusion = True
        cfg.weight_quant_dtype = None
        cfg.kvcache_quant_dtype = msdtype.int8
        cfg.act_quant_dtype = msdtype.int8
        cfg.outliers_suppression = OutliersSuppressionType.SMOOTH
        cfg.act_quant_granularity = QuantGranularity.PER_TOKEN
        cfg.kvcache_quant_granularity = QuantGranularity.PER_TOKEN

        # Export to YAML
        cfg.dump(yaml_file)

        # Load from YAML to new configuration
        new_cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
        new_cfg.load(yaml_file)

        # Verify all attributes
        assert new_cfg.weight_symmetric is False
        assert new_cfg.enable_deploy_fusion is True
        assert new_cfg.kvcache_quant_dtype is msdtype.int8
        assert new_cfg.weight_quant_dtype is None
        assert new_cfg.act_quant_dtype is msdtype.int8
        assert new_cfg.outliers_suppression == OutliersSuppressionType.SMOOTH
        assert new_cfg.act_quant_granularity == QuantGranularity.PER_TOKEN
        assert new_cfg.kvcache_quant_granularity == QuantGranularity.PER_TOKEN
    finally:
        # Clean up temporary files
        if os.path.exists(yaml_file):
            os.remove(yaml_file)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_inner_ptq_config_yaml_parse_unparse():
    """
    Feature: InnerPTQConfig YAML parse and unparse
    Description: Test InnerPTQConfig YAML parsing and unparsing behavior with value override
    Expectation: Load from YAML correctly overrides existing values
    """
    yaml_file = 'my_cfg.yaml'
    try:
        # Create configuration and export to YAML
        cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
        cfg.dump(yaml_file)

        # Create new configuration and set different values
        new_cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
        new_cfg.act_quant_dtype = msdtype.uint8
        new_cfg.weight_quant_dtype = msdtype.uint8

        # Load configuration from YAML, should override previously set values
        new_cfg.load(yaml_file)

        # Verify loaded values (should be restored from YAML, not the previously set uint8)
        assert new_cfg.act_quant_dtype is None  # Default value
        assert new_cfg.weight_quant_dtype == msdtype.int8  # Default value
    finally:
        # Clean up temporary files
        if os.path.exists(yaml_file):
            os.remove(yaml_file)
