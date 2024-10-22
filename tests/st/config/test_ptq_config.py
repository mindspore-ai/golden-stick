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
"""test for ptq related config"""
import os
import sys

import pytest
from mindspore import dtype as msdtype

from mindspore_gs.ptq.ptq_config import (PTQConfig, SmoothQuantConfig, InnerPTQConfig,
                                         PTQApproach, PTQMode, OutliersSuppressionType, QuantType)
from mindspore_gs.common.gs_enum import BackendTarget

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../'))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sq_config():
    """
    Feature: smooth quant algo config.
    Description: Feed invalid param to SmoothQuantConfig to raise type error.
    Expectation: Except error.
    """
    with pytest.raises(TypeError):
        _ = SmoothQuantConfig(alpha='0.5')

    with pytest.raises(TypeError):
        _ = SmoothQuantConfig(alpha=0.5, is_deploy=1)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ptq_config_construct():
    """
    Feature: config for customer for post training quant
    Description: Feed valid and invalid param to ptq_config to test constructor
    Expectation: as expectation
    """
    cfg = PTQConfig()
    assert cfg.mode == PTQMode.QUANTIZE
    assert cfg.backend == BackendTarget.ASCEND

    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.NONE)
    assert cfg.mode == PTQMode.DEPLOY
    assert cfg.backend == BackendTarget.NONE

    with pytest.raises(ValueError):
        _ = PTQConfig(mode='none')

    with pytest.raises(ValueError):
        _ = PTQConfig(backend=PTQMode.QUANTIZE)

    with pytest.raises(TypeError):
        _ = PTQConfig(opname_blacklist=1)
    with pytest.raises(TypeError):
        _ = PTQConfig(opname_blacklist="1")
    with pytest.raises(TypeError):
        _ = PTQConfig(opname_blacklist=["1", 1])
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_dtype=["1"])
    with pytest.raises(ValueError):
        _ = PTQConfig(weight_quant_dtype=msdtype.float16)
    with pytest.raises(ValueError):
        _ = PTQConfig(kvcache_quant_dtype=["1"])
    with pytest.raises(ValueError):
        _ = PTQConfig(kvcache_quant_dtype=msdtype.float16)
    with pytest.raises(ValueError):
        _ = PTQConfig(act_quant_dtype=["1"])
    with pytest.raises(ValueError):
        _ = PTQConfig(act_quant_dtype=msdtype.float16)
    with pytest.raises(TypeError):
        _ = PTQConfig(outliers_suppression=1)
    with pytest.raises(TypeError):
        _ = PTQConfig(outliers_suppression='awq')
    with pytest.raises(ValueError):
        _ = PTQConfig(algo_args="1")
    with pytest.raises(ValueError):
        _ = PTQConfig(algo_args=["1", 1])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_inner_ptq_config():
    """
    Feature: config for post training quant
    Description: Feed invalid param to ptq_config to raise value error.
    Expectation: Except error.
    """
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(approach='no_such_approach')
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(approach=PTQApproach.RTN, act_quant_dtype=msdtype.int8)
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(approach=PTQApproach.RTN, weight_quant_dtype=None, kvcache_quant_dtype=None)
    with pytest.raises(ValueError):
        _ = InnerPTQConfig(approach=PTQApproach.RTN, weight_quant_dtype=msdtype.int8, kvcache_quant_dtype=msdtype.int8)

    cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    assert cfg.approach == PTQApproach.SMOOTH_QUANT
    with pytest.raises(TypeError):
        cfg.weight_symmetric = 1
        cfg.__post_init__()

    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.NONE)
    icfg = InnerPTQConfig.inner_config(cfg, PTQApproach.SMOOTH_QUANT)
    assert icfg.approach == PTQApproach.SMOOTH_QUANT


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ptq_algo_config():
    """
    Feature: config for post training quant
    Description: Feed invalid param to ptq_config to raise value error.
    Expectation: all value is consistent with default
    """
    cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    assert cfg.algo_args.get('alpha') == 0.5

    cfg = InnerPTQConfig(approach=PTQApproach.RTN)
    assert cfg.mode == PTQMode.QUANTIZE
    assert cfg.backend == BackendTarget.ASCEND
    assert cfg.act_per_channel is False
    assert cfg.act_symmetric is False
    assert cfg.weight_symmetric is True
    assert cfg.act_narrow_range is False
    assert cfg.weight_narrow_range is False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_inner_ptq_func():
    """
    Feature: test convert PTQConfig to InnerPTQConfig
    Description: convert PTQConfig to InnerPTQConfig
    Expectation: as expect
    """
    inner_cfg = InnerPTQConfig()
    inner_cfg.mode = PTQMode.DEPLOY
    inner_cfg.backend = BackendTarget.ASCEND
    inner_cfg.act_quant_dtype = msdtype.int8
    inner_cfg.act_weight_quant_type = QuantType.A8W8

    ptq_cfg = PTQConfig(mode=PTQMode.DEPLOY,
                        backend=BackendTarget.ASCEND, act_quant_dtype=msdtype.int8)
    convert_inner_cfg = inner_cfg.inner_config(ptq_cfg)
    assert convert_inner_cfg == inner_cfg

    with pytest.raises(TypeError):
        inner_cfg.inner_config('none')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ptq_yaml_dump_and_load():
    """
    Feature: test load and dump api for gs config
    Description: dump config to yaml and then load it with yaml
    Expectation: dump and load file success
    """
    cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    cfg.weight_symmetric = False
    cfg.enable_deploy_fusion = True
    cfg.weight_quant_dtype = None
    cfg.kvcache_quant_dtype = msdtype.int8
    cfg.act_quant_dtype = msdtype.int8
    cfg.outliers_suppression = OutliersSuppressionType.SMOOTH
    cfg.act_dynamic_quant = True
    cfg.act_weight_quant_type = QuantType.A16W8
    cfg.dump('my_cfg.yaml')
    new_cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    new_cfg.load('my_cfg.yaml')
    assert new_cfg.weight_symmetric is False
    assert new_cfg.enable_deploy_fusion is True
    assert new_cfg.kvcache_quant_dtype is msdtype.int8
    assert new_cfg.weight_quant_dtype is None
    assert new_cfg.act_quant_dtype is msdtype.int8
    assert new_cfg.outliers_suppression == OutliersSuppressionType.SMOOTH
    assert new_cfg.act_dynamic_quant is True
    assert new_cfg.act_weight_quant_type == QuantType.A16W8


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ptq_yaml_parse_unparse():
    """
    Feature: test load and dump api for ptq config
    Description: dump config to yaml and then load it with yaml
    Expectation: dump and load file success
    """
    cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    cfg.dump('my_cfg.yaml')
    new_cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    new_cfg.act_quant_dtype = msdtype.uint8
    new_cfg.weight_quant_dtype = msdtype.uint8
    new_cfg.load('my_cfg.yaml')
    assert new_cfg.act_quant_dtype is None
    assert new_cfg.weight_quant_dtype == msdtype.int8


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_algo_args():
    """
    Feature: test algo_args of PTQConfig.
    Description: input algo_args to PTQConfig and check
    Expectation: as expect
    """
    # use dataclass as algo_args
    sq_args = SmoothQuantConfig(alpha=0.8)
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, algo_args=sq_args)
    assert cfg.algo_args
    assert isinstance(cfg.algo_args, dict)
    assert "alpha" in cfg.algo_args
    assert cfg.algo_args["alpha"] == 0.8

    inner_cfg = InnerPTQConfig.inner_config(cfg)
    assert inner_cfg.algo_args
    assert isinstance(inner_cfg.algo_args, dict)
    assert "alpha" in inner_cfg.algo_args
    assert inner_cfg.algo_args["alpha"] == 0.8

    inner_cfg = InnerPTQConfig(approach=PTQApproach.SMOOTH_QUANT)
    assert inner_cfg.algo_args
    assert isinstance(inner_cfg.algo_args, dict)
    assert "alpha" in inner_cfg.algo_args
    assert inner_cfg.algo_args["alpha"] == SmoothQuantConfig().alpha

    # use dict as algo_args
    cfg = PTQConfig(mode=PTQMode.DEPLOY, backend=BackendTarget.ASCEND, algo_args={'a': 1})
    assert cfg.algo_args
    assert isinstance(cfg.algo_args, dict)
    assert "a" in cfg.algo_args
    assert cfg.algo_args["a"] == 1

    inner_cfg = InnerPTQConfig.inner_config(cfg)
    assert inner_cfg.algo_args
    assert isinstance(inner_cfg.algo_args, dict)
    assert "a" in inner_cfg.algo_args
    assert inner_cfg.algo_args["a"] == 1
