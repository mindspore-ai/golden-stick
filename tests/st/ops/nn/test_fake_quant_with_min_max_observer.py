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
"""test_fake_quant_with_min_max_observer"""
import pytest
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore_gs.quantization.simulated_quantization.quant_cells import FakeQuantWithMinMaxObserver


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_fake_quant_with_min_max_observer():
    """
    Feature: Test nn ops FakeQuantWithMinMaxObserver.
    Description: Test nn ops FakeQuantWithMinMaxObserver.
    Expectation: Success.
    """
    fake_quant = FakeQuantWithMinMaxObserver()
    x = Tensor(np.array([[1, 2, 1], [-2, 0, -1]]), mindspore.float32)
    result = fake_quant(x).asnumpy()
    expect_output = np.array([[0.9882355, 1.9764705, 0.9882355],
                              [-1.9764705, 0., -0.9882355]]).astype(np.float32)
    assert np.allclose(expect_output, result, 0.001, 0.001)
