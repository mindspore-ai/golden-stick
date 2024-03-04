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
"""test interfaces of rtn."""
import os
import sys

import pytest
from mindspore import nn
from mindspore_gs.ptq import RoundToNearestPTQ as RTN
from mindspore_gs.ptq import RTNConfig
from mindspore_gs.ptq.ptq_config import PTQConfig
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../../'))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor():
    """
    Feature: RoundToNearestPTQ algorithm.
    Description: Call constructor of RoundToNearestPTQ and check config.
    Expectation: RTNConfig is updated according to argument `config` of constructor.
    """

    config = PTQConfig()
    ptq = RTN(config)
    # pylint: disable=W0212
    quant_config: RTNConfig = ptq._config
    assert config == quant_config

    ptq_default = RTN()
    # pylint: disable=W0212
    quant_defalut_config = ptq_default._config
    assert quant_defalut_config == config


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor_error():
    """
    Feature: RoundToNearestPTQ algorithm.
    Description: Feed invalid config to constructor of RoundToNearestPTQ and except error.
    Expectation: Except error.
    """
    config = {"per_channel": (1, 1)}
    with pytest.raises(TypeError):
        RTN(config)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_convert_error():
    """
    Feature: simulated quantization convert function.
    Description: Feed invalid type of bn_fold to convert function.
    Expectation: Except TypeError.
    """
    network = nn.Conv2d(6, 5, kernel_size=2)
    ptq = RTN()
    new_network = ptq.apply(network)
    with pytest.raises(TypeError, match="The parameter `net_opt` must be isinstance of Cell"):
        ptq.convert(100)

    with pytest.raises(TypeError, match="The parameter `ckpt_path` must be isinstance of str"):
        ptq.convert(new_network, 100)

    with pytest.raises(ValueError, match="The parameter `ckpt_path` can only be empty or a valid file"):
        ptq.convert(new_network, "file_path")
