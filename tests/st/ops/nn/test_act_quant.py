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
"""test_act_quant"""
import pytest
import numpy as np

import mindspore
from mindspore import Tensor, nn
from mindspore_gs.ops.nn import ActQuant
from .nn_utils import create_quant_config


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_act_quant():
    """
    Feature: Test nn ops ActQuant.
    Description: Test nn ops ActQuant.
    Expectation: Success.
    """
    qconfig = create_quant_config()
    act_quant = ActQuant(nn.ReLU(), quant_config=qconfig)
    x = Tensor(np.array([[1, 2, -1], [-2, 0, -1]]), mindspore.float32)
    expect_output = np.array([[0.9882355, 1.9764705, 0.], [0., 0., 0.]]).astype(np.float32)
    result = act_quant(x).asnumpy()
    assert np.allclose(expect_output, result, 0.001, 0.001)
