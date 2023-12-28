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
"""Test MinmaxUpdateLayer ops."""
import pytest
import numpy as np

from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops.operations import _quant_ops as ms_Q
import mindspore_gs.quantization.ops.operations as custom_Q


class MinmaxUpdateLayerNet(Cell):
    """Net."""

    def __init__(self):
        """Init."""
        super(MinmaxUpdateLayerNet, self).__init__()
        self.program = custom_Q.MinMaxUpdatePerLayer()

    def construct(self, x, min_val, max_val):
        """Construct."""
        res = self.program(x, min_val, max_val)
        return res[0] + res[1]


class MSNet(Cell):
    """MS net."""

    def __init__(self):
        """Init."""
        super(MSNet, self).__init__()
        self.program = ms_Q.MinMaxUpdatePerLayer()

    def construct(self, x, min_val, max_val):
        """Construct."""
        res = self.program(x, min_val, max_val)
        return res[0] + res[1]


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [context.GRAPH_MODE])
def test_mupl_gpu(mode):
    """
    Feature: Test ops MinMaxUpdatePerLayer.
    Description: Test ops MinMaxUpdatePerLayer.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target='GPU')
    x = np.array([-0.1, 0.0, 63.75, 63.8]).astype(np.float32)
    min_val = np.array([-0.1]).astype(np.float32)
    max_val = np.array([63.65]).astype(np.float32)
    sens = np.array([1.0]).astype(np.float32)
    out = MinmaxUpdateLayerNet()(Tensor(x), Tensor(min_val), Tensor(max_val))
    out_expect = MSNet()(Tensor(x), Tensor(min_val), Tensor(max_val))
    assert np.allclose(out.asnumpy(), out_expect.asnumpy(), 0.00001, 0.00001)
    net = MinmaxUpdateLayerNet()
    ms_net = MSNet()
    out_x, out_min, out_max = ops.GradOperation(sens_param=True, get_all=True)(net)(
        Tensor(x), Tensor(min_val), Tensor(max_val), Tensor(sens))
    out_x_ms, out_min_ms, out_max_ms = ops.GradOperation(sens_param=True, get_all=True)(ms_net)(
        Tensor(x), Tensor(min_val), Tensor(max_val), Tensor(sens))
    assert np.allclose(out_x_ms.asnumpy(), out_x.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(out_min_ms.asnumpy(), out_min.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(out_max_ms.asnumpy(), out_max.asnumpy(), 0.00001, 0.00001)
