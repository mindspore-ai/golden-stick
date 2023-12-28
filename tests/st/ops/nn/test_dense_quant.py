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
"""test_dense_quant"""
import pytest
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore.nn import Dense
from mindspore_gs.quantization.simulated_quantization.quant_cells import DenseQuant
from .nn_utils import TestLayerPolicy


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dense_quant():
    """
    Feature: Test nn ops DenseQuant.
    Description: Test nn ops DenseQuant.
    Expectation: Success.
    """
    policy = TestLayerPolicy(1, True, False)
    dense = Dense(2, 1, weight_init='ones')
    dense_quant = DenseQuant(dense, policy, quant_config=policy.get_quantizer())
    x = Tensor(np.array([[1, 5], [3, 4]]), mindspore.float32)
    result = dense_quant(x).asnumpy()
    expect_output = np.array([[5.929413], [6.9176483]]).astype(np.float32)
    assert np.allclose(expect_output, result, 0.001, 0.001)
