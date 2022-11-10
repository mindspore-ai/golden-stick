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
"""Test FakeQuantPerChannel ops."""
import pytest
import numpy as np

from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops.operations import _quant_ops as ms_Q
import mindspore_gs.ops.operations as custom_Q


class FakeQuantPerChannelNet(Cell):
    """Net."""

    def __init__(self):
        """Init."""
        super(FakeQuantPerChannelNet, self).__init__()
        self.program = custom_Q.FakeQuantPerChannel()

    def construct(self, x, min_val, max_val):
        """Construct."""
        return self.program(x, min_val, max_val)


class MSNet(Cell):
    """MS net."""

    def __init__(self):
        """Init."""
        super(MSNet, self).__init__()
        self.program = ms_Q.FakeQuantPerChannel()

    def construct(self, x, min_val, max_val):
        """Construct."""
        return self.program(x, min_val, max_val)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE])
def test_fqpc_gpu(mode):
    """
    Feature: Test ops FakeQuantPerChannel.
    Description: Test ops FakeQuantPerChannel.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='GPU')
    x = np.array([10.0, 10.0, 20.0, 20.0]).astype(np.float32)
    min_val = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    max_val = np.array([25.0, 25.0, 25.0, 25.0]).astype(np.float32)

    net = FakeQuantPerChannelNet()
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))[0].asnumpy()

    ms_net = MSNet()
    expect = ms_net(Tensor(x), Tensor(min_val), Tensor(max_val))[0].asnumpy()
    assert np.allclose(expect, output, 0.001, 0.001)

    sens = np.array([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    bprop_out = ops.GradOperation(sens_param=True, get_all=True)(net)(
        Tensor(x), Tensor(min_val), Tensor(max_val), Tensor(sens))[0].asnumpy()
    expect_bprop_out = ops.GradOperation(sens_param=True, get_all=True)(ms_net)(
        Tensor(x), Tensor(min_val), Tensor(max_val), Tensor(sens))[0].asnumpy()
    assert np.allclose(expect_bprop_out, bprop_out, 0.001, 0.001)
