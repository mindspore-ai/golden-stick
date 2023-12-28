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
"""test_conv2d_bn_fold_quant_one_conv"""
import pytest
import numpy as np

import mindspore
from mindspore import Tensor
from mindspore_gs.ops.nn import Conv2dBnFoldQuantOneConv
from mindspore_gs.quantization.simulated_quantization.combined import Conv2dBn
from .nn_utils import TestLayerPolicy


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_conv2d_bn_fold_quant_one_conv():
    """
    Feature: Test nn ops Conv2dBnFoldQuantOneConv.
    Description: Test nn ops Conv2dBnFoldQuantOneConv.
    Expectation: Success.
    """
    policy = TestLayerPolicy(1, True, False)
    convbn: Conv2dBn = Conv2dBn(1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid", weight_init="ones")
    conv2d_bnfold = Conv2dBnFoldQuantOneConv(convbn, policy, 1, 1, kernel_size=(2, 2), stride=(1, 1), pad_mode="valid",
                                             weight_init="ones", quant_config=policy.get_quant_config())
    x = Tensor(np.array([[[[1, 0, 3], [1, 4, 7], [2, 5, 2]]]]), mindspore.float32)
    result = conv2d_bnfold(x).asnumpy()
    expect_output = np.array([[5.9296875, 13.8359375], [11.859375, 17.78125]]).astype(np.float32)
    assert np.allclose(expect_output, result, 0.001, 0.001)
