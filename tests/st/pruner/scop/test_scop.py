# Copyright 2022 Huawei Technologies Co., Ltd
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
"""test qat."""

import pytest
from mindspore_gs.pruner.scop.scop_pruner import PrunerFtCompressAlgo


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor():
    """
    Feature: SCOP algorithm.
    Description: Call constructor of SCOP and check config.
    Expectation: prune_rate is updated according to argument `config` of constructor.
    """

    scop = PrunerFtCompressAlgo({"prune_rate": 0.8})
    assert scop._config.prune_rate == 0.8

    scop = PrunerFtCompressAlgo(None)
    assert scop._config.prune_rate == 0.0

    scop = PrunerFtCompressAlgo({})
    assert scop._config.prune_rate == 0.0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_constructor_error():
    """
    Feature: SCOP algorithm.
    Description: Feed invalid config to constructor of SCOP and except error.
    Expectation: Except error.
    """

    config = {"prune_rate": 1}
    with pytest.raises(TypeError, match="For 'PrunerFtCompressAlgo', the 'prune_rate' must be 'float',  but got "
                                        "'int'."):
        PrunerFtCompressAlgo(config)

    with pytest.raises(TypeError, match="For 'PrunerFtCompressAlgo', the type of 'config' should be 'dict', but got "
                                        "type 'int'."):
        PrunerFtCompressAlgo(1)

    config = {"prune_rate": 1.1}
    with pytest.raises(ValueError, match="For 'PrunerFtCompressAlgo', the 'prune_rate' must be in range of "
                                         "\\[0.0, 1.0\\), but got 1.1 with type 'float'."):
        PrunerFtCompressAlgo(config)
