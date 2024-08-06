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
"""test fake-quantizer cells."""

import os
import pytest
from mindspore import Parameter, context, GRAPH_MODE, PYNATIVE_MODE
from mindspore.common.initializer import initializer
from mindspore import dtype as mstype
from mindspore_gs.ptq.fake_quantizer import MinMaxPerChannel


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_minmaxperchannel(mode):
    """
    Feature: MinMaxPerChannel.
    Description: Call forward function of MinMaxPerChannel.
    Expectation: Forward successful.
    """
    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    shape = (5, 6)
    rank = 2
    axis = 1
    channels = 6
    weight = Parameter(initializer('ones', shape, mstype.float32), name="weight")
    fq = MinMaxPerChannel(symmetric=True, data_rank=rank, quant_dtype=mstype.int8, narrow_range=False, axis=axis,
                          output_channel=channels)
    fq(weight)
    assert fq.float_min.shape == (6,)
    assert fq.float_max.shape == (6,)
    mins = fq.float_min.value().asnumpy()
    for min_ in mins:
        assert min_ == 1
    maxs = fq.float_max.value().asnumpy()
    for max_ in maxs:
        assert max_ == 1
    os.environ.pop('GRAPH_OP_RUN')

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [GRAPH_MODE, PYNATIVE_MODE])
def test_minmaxperchannel_3d(mode):
    """
    Feature: MinMaxPerChannel.
    Description: Call forward function of MinMaxPerChannel.
    Expectation: Forward successful.
    """
    os.environ['GRAPH_OP_RUN'] = "1"
    context.set_context(device_target="Ascend", mode=mode)
    shape = (5, 6, 7)
    rank = 3
    axis = 1
    channels = 6
    weight = Parameter(initializer('ones', shape, mstype.float32), name="weight")
    fq = MinMaxPerChannel(symmetric=True, data_rank=rank, quant_dtype=mstype.int8, narrow_range=False, axis=axis,
                          output_channel=channels)
    fq(weight)
    assert fq.float_min.shape == (6,)
    assert fq.float_max.shape == (6,)
    mins = fq.float_min.value().asnumpy()
    for min_ in mins:
        assert min_ == 1
    maxs = fq.float_max.value().asnumpy()
    for max_ in maxs:
        assert max_ == 1
    os.environ.pop('GRAPH_OP_RUN')
